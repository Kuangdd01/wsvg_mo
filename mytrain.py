
# ==================================================================================================================== #
#                                           make slrum work with local import                                          #
#                              refer to https://stackoverflow.com/a/39574373/16534997                                  #
import sys, os

sys.path.append(os.getcwd())
# ==================================================================================================================== #
os.environ['CUDA_LAUNCH_BLOCKING']='1'

import json
import os
import pathlib
import shutil
import time
import random
import argparse
import numpy as np
import torch
from torch import einsum, nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import build_glove_vocab, build_dataloader
from utils.debugger import set_writer
import utils.debugger as cutils
from utils.utils_algo import AverageMeter, ProgressMeter, accuracy, adjust_learning_rate, len2mask
from models import build_model
from utils.utils_loss import ConLoss, finf, feps, WSLoss, ConLossMask
from einops import rearrange, repeat
from flow.mask_duplicate_image_inbatch import get_duplicate_mask, get_same_label_mask, get_visualization_masked_label_query
from flow.inner_similarity import get_inner_similarity
from utils.utils_algo import update_pseudo_label_temperature
#debug
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(
    description='mat use em refine confidence adapted from PiCO',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seed', default=4008, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--exp-dir', default='experiment/mat', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')

parser.add_argument('--task', default='phrase')
parser.add_argument('--glove_file', default='')
parser.add_argument('--dataroot', default='')
parser.add_argument('--referoot', default='')
parser.add_argument('--boxfile', default='')
parser.add_argument('--features_path', default='')
parser.add_argument('--dataset_name', default='refcoco+')
parser.add_argument('--mat_root', default='')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--prefetch_factor', default=2, type=int)

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--threshold', default=0.8, type=float, help='inner sim threshold')

parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay_epochs', type=str, default='40,70',
                    help='where to decay lr, can be a list')
parser.add_argument('--p_t_range', type=str, default='0.5, 1',
                    help='range of pseudo label temperature')
parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    dest='weight_decay')
parser.add_argument('--clip_norm', type=float, default=0)

parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')

parser.add_argument('--soft_init', action='store_true')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--base_temperature', default=1, type=float)
parser.add_argument('--update_conf_start', default=1, type=int,
                    help='Start Prototype Updating')

parser.add_argument('--do_topk', action='store_true')
parser.add_argument('--topk_values', type=str, default='30,15,8,4')
parser.add_argument('--topk_epochs', type=str, default='15,30,50,80')

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--v_feature_dropout_prob', type=float, default=0.0)
parser.add_argument('--normalize_feature', action='store_true')
parser.add_argument('--arch', default='mat')
parser.add_argument('--lin_rnn_lr', type=float)
parser.add_argument('--sum_scale', type=float, default=12.0)
parser.add_argument('--reduce_method', type=str, default='sum')
parser.add_argument('--no_contrastive', action='store_true')
parser.add_argument('--neg_num', type=int, default=None)

parser.add_argument('--var_t', action='store_true')
parser.add_argument('--mask', type=str, default='label')
parser.add_argument('--p_temperature', type=float, default=1.0)

parser.add_argument('--momentum_m', type=float, default=0.995)

parser.add_argument('--fixed_pt', action='store_true')
parser.add_argument('--convert', action='store_true')



def main():
    args = parser.parse_args()
    model_path = '{task}_{arch}lr_{lr}_ep_{ep}_sd_{seed}_{soft_init}{hard}_ema_{ema}_t_{t}_bt_{bt}_update_start_{upd}_dp_{dp}_bs_{bs}_th{th}'.format(
        task=args.task,
        arch=args.arch,
        lr=args.lr,
        ep=args.epochs,
        seed=args.seed,
        soft_init='softinit_' if args.soft_init else '',
        hard='hard' if args.hard else 'soft',
        ema=args.conf_ema_range,
        t=args.temperature,
        bt=args.base_temperature,
        upd=args.update_conf_start,
        dp=args.dropout,
        bs=args.batch_size,
        th=args.threshold
    )
    print(model_path)
    # mkdir
    #breakpoint()
    save_root = os.path.join(args.exp_dir, model_path)
    i = 0
    while pathlib.Path(f"{save_root}_{i}").exists():
        i += 1
    exp_dir = f"{save_root}_{i}"
    print("expDIR:{}".format(exp_dir))
    pathlib.Path(exp_dir).mkdir(parents=True)
    # save input args
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    # renew args
    args.exp_dir = exp_dir
    args.conf_ema_range = [float(item) for item in str(args.conf_ema_range).split(',')]
    iterations = str(args.lr_decay_epochs).split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # topk
    args.p_t_range = [float(it) for it in str(args.p_t_range).split(',') ] #st ed
    print("****",args.p_t_range[0], args.p_t_range[1])
    args.topk_values = [int(item) for item in str(args.topk_values).split(',')]
    args.topk_epochs = [int(item) for item in str(args.topk_epochs).split(',')]
    assert len(args.topk_epochs) == len(args.topk_values)
    # writer
    args.writer = SummaryWriter(args.exp_dir)
    set_writer(args.writer)
    # seed
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # device
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    # show final args
    print(args)
    #breakpoint()
    # main_worker
    main_worker(args)


def get_target_topk(args, epoch):
    if not args.do_topk:
        return None
    i = 0
    while i < len(args.topk_epochs) and args.topk_epochs[i] <= epoch:
        i += 1
    if i == 0:
        return None
    return args.topk_values[i - 1]


def main_worker(args):
    print('=> build glove Tokenizer >>>>')
    gloveTokenizer = build_glove_vocab(args.glove_file)

    print("=> creating model")
    model = build_model(gloveTokenizer.embeddings_ori, args)
    model.float()
    print(model)

    print("=> creating optimizer")
    # select different structures
    if args.arch in ['matrnn', 'dual_zero', 'dual_zero_atten','dual_zero_nov','dual_zero_nov_att', 'dual_mom'] and args.lin_rnn_lr is not None:
        print(f"==> use different lr for lin_rnn: {args.lin_rnn_lr}. other params lr is {args.lr}")

        rnn_params = list(model.linear_rnn.parameters())
        rnn_params_id = list(map(id, rnn_params))
        rnn_params_name = [n for n, p in model.named_parameters() if id(p) in rnn_params_id]
        print(f"rnn params : {rnn_params_name}")

        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in rnn_params_id]
        other_params_id = list(map(id, other_params))
        other_params_name = [n for n, p in model.named_parameters() if id(p) in other_params_id]
        print(f"other params : {other_params_name}")

        if args.lin_rnn_lr == 0:
            model_parameters = [
                {'params': other_params},
            ]
        else:
            model_parameters = [
                {'params': rnn_params, 'lr': args.lin_rnn_lr},
                {'params': other_params},
            ]
        optimizer = torch.optim.SGD(model_parameters, args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    print(optimizer)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            assert False

    # model to device
    model = model.to(args.device)

    # dataloader
    train_loader, eval_loader, test_loader = build_dataloader(args, gloveTokenizer)
    if args.epochs <= 0:
        print('epochs <= 0, only do test')
        # eval
        acc_eval = val(model, eval_loader, args, args.epochs, args.writer, device=args.device, prefix='eval')
        if isinstance(test_loader, list):
            acc_test = []
            for test_dl in test_loader:
                acc_dl = val(model, test_dl, args, args.epochs, args.writer, device=args.device, prefix='test')
                acc_test.append(acc_dl)
        else:
            acc_test = val(model, test_loader, args, args.epochs, args.writer, device=args.device, prefix='test')
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write(f'Epoch {"final"}: '
                    f'Eval Acc {acc_eval}'
                    f'Test Acc {acc_test}\n')
        return

    print('Calculating uniform targets...')
    n = len(train_loader.dataset)
    max_phrase_num = train_loader.dataset.max_phrase_num
    max_region_num = train_loader.dataset.max_region_num
    phrase_mask = len2mask(train_loader.dataset.num_phrase, (n, max_phrase_num)).float()
    region_confidence = len2mask(train_loader.dataset.num_obj, (n, max_region_num)).float() / train_loader.dataset.num_obj.unsqueeze(-1)
    confidence = einsum('b q, b k -> b q k', phrase_mask, region_confidence)


    loss_fn = WSLoss(conf_ema_m=args.conf_ema_range[0],
                      hard=args.hard,
                      temperature=args.temperature,
                      base_temperature=args.base_temperature,
                      no_contrastive=args.no_contrastive,
                      neg_num=args.neg_num,
                      args=args)

    if args.soft_init:
        print('\nSoft init\n')
        soft_init(model, train_loader, args, loss_fn, args.writer, args.device)
    print('\nStart Training\n')
    best_eval_acc = 0
    best_eval_test_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train
        update_conf = epoch >= args.update_conf_start
        adjust_learning_rate(args, optimizer, epoch)
        if update_conf:
            target_topk = get_target_topk(args, epoch)
        else:
            target_topk = None
        train(model, train_loader, loss_fn, optimizer, epoch, args, args.writer, update_conf=update_conf,
              target_topk=target_topk, device=args.device, glove_tk=gloveTokenizer)
        acc_eval = val(model, eval_loader, args, epoch, args.writer, device=args.device, prefix='eval')
        if isinstance(test_loader, list):
            acc_test = []
            for test_dl in test_loader:
                acc_dl = val(model, test_dl, args, epoch, args.writer, device=args.device, prefix='test')
                acc_test.append(acc_dl)
        else:
            acc_test = val(model, test_loader, args, epoch, args.writer, device=args.device, prefix='test')
        # mmc = loss_fn.confidence.amax(dim=-1).sum() / train_loader.dataset.num_phrase.sum()
        mmc = 0
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write(f'Epoch {epoch}: '
                    f'Eval Acc {acc_eval}, Best Eval Acc {best_eval_acc}. '
                    f'Test Acc {acc_test}. Best Eval Test Acc {best_eval_test_acc}'
                    f'(lr {optimizer.param_groups[0]["lr"]}, MMC {mmc}).\n')
        is_best = False
        if acc_eval > best_eval_acc:
            best_eval_acc = acc_eval
            best_eval_test_acc = acc_test
            is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))


def train(model, train_loader, loss_cont_fn, optimizer, epoch, args, writer, update_conf=True, target_topk=None, device='cuda', glove_tk=None):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_atten = AverageMeter('Acc@Atten', ':2.2f')
    acc_target = AverageMeter('Acc@Target', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    mask_time = AverageMeter('MaskTime', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_atten, acc_target, loss_cont_log, mask_time],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if type(loss_cont_fn) == ConLossMask:
        print("using mask loss!")
    if type(loss_cont_fn) == ConLoss:
        print("using default policy")
    if args.mask =="label":
            print("same label mask!")
            # print("do not use x_mask-dot")
    if args.mask == "duplicate":
            print("duplicate mask!")
    else:
            print("zero label policy")
    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # move to device
        batch = move_to_device(batch, device)
        (index,
         image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
         caption, caption_ids, span_mask, length,
         phrases, mix_phrases_id, num_phrase,
         ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
         ) = batch
        # word2idx = glove_tk.vocab
        # idx2word = {id: word for id, word in word2idx.items()}
        # encode
        #breakpoint()
        if args.arch == 'dual_zero_nov' or args.arch == "dual_zero_nov_att":
            phrase, region, label_emb = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
            pseudo_tgt_ = model.get_pseudo_target(phrase, region)
            # pseudo_tgt = torch.randn(256,32,100).cuda()
        elif args.arch == 'dual_mom':
            phrase, region, label_emb, momentum_out = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
            pseudo_tgt_ = model.get_pseudo_target(momentum_out[0], momentum_out[1])
        elif args.arch != 'mat':
            phrase, region = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
        else:
            phrase, region = model(mix_phrases_id, mix_objs_id, mix_features_pad)
        if args.normalize_feature:
            phrase = phrase / (phrase.norm(dim=-1, keepdim=True) + feps(phrase.dtype))
            region = region / (region.norm(dim=-1, keepdim=True) + feps(region.dtype))
            # if type(loss_cont_fn) == WSLoss:
            #     # print("using psl")
            #     pseudo_tgt_ = model.get_pseudo_target(phrase, region)
        # print(phrase.shape, region.shape)
        atten = einsum('b q d, a k d -> b a q k', phrase, region)
        #init_confidence 
        # mask
        phrase_mask = len2mask(num_phrase, phrase.shape[:2]).bool()
        region_mask = len2mask(mix_num_boxes, region.shape[:2]).bool()
        x_mask = einsum('b q, b k -> b q k', phrase_mask, region_mask)
        # pseudo_tgt = torch.softmax(pseudo_tgt)

        # reshape this pseudo label reconstruct the compute graph
        pseudo_tgt = pseudo_tgt_.masked_fill(~x_mask, -finf(atten.dtype))
        if args.fixed_pt:
            p_t = args.p_temperature
        else:
            p_t = update_pseudo_label_temperature(epoch=epoch,initial_temperature=args.p_t_range[1],
                                                  min_temperature=args.p_t_range[0],total_epochs=args.epochs)
        pseudo_tgt = pseudo_tgt / p_t
        pseudo_tgt = torch.softmax(pseudo_tgt, dim=-1)

        if args.mask == "duplicate":
            # print("duplicate mask!")
            mask = get_duplicate_mask(image_id=image_id, output=atten)
        elif args.mask =="label":
            # print("same label mask!")
            mask = get_same_label_mask(label_feature=label_emb, label_id=mix_objs_id, query_embeddings=phrase, device=region.device)
            # visual_dict = get_visualization_masked_label_query(mask=mask, x_mask=x_mask, phrase_mask=phrase_mask, bbox=boxes_ori_tensor,
            #  phrases=phrases, label_id=mix_objs_id, idx2word=idx2word) 
            # print("using x_mask")
            mask = rearrange(mask, 'b q (a k) -> b a q k', b=mask.shape[0], q=mask.shape[1], k=x_mask.shape[-1], a=mask.shape[0])
            mask = mask * x_mask.unsqueeze(0)
            mask = rearrange(mask, 'b a q k -> b q (a k)', b=mask.shape[0], q=x_mask.shape[1], k=x_mask.shape[-1])
            # print("avg mask for one phrase:{}".format(mask.sum()/phrase_mask.sum()))
        else:
            # print("no label policy")
            b,q,k = atten.shape[0],atten.shape[2],atten.shape[-1]
            mask = torch.zeros((b,q,b*k), device=atten.device, dtype=bool) 

        
        tst = time.time()
        from flow.inner_similarity import get_mask_similarity_matrix_by_threshold
        norm_mix_features_pad = mix_features_pad / (mix_features_pad.norm(dim=-1, keepdim=True) + feps(phrase.dtype))
        region_similarity = torch.einsum('b q d, a k d -> b a q k',norm_mix_features_pad, norm_mix_features_pad)
        atten, origin_attn, inner_mask = get_mask_similarity_matrix_by_threshold(region_similarity, atten, args.threshold,
                                                                                   -torch.finfo(atten.dtype).max, atten.device)
        ted = time.time()
        mask_time.update(ted-tst)

        if type(loss_cont_fn) == ConLossMask:
            loss, other = loss_cont_fn(atten, mask, index, topk=target_topk, x_mask=x_mask,
                                        update_conf=update_conf, attn_mask=inner_mask, origin_attn=origin_attn)
        elif type(loss_cont_fn) == WSLoss:
            if args.convert:
                loss, other = loss_cont_fn(atten, index, topk=target_topk, 
                                       x_mask=x_mask, update_conf=update_conf, 
                                       pseudo_target=pseudo_tgt, origin_att=origin_attn, inner_mask=inner_mask)
            else:
                loss, other = loss_cont_fn(atten, index, topk=target_topk, 
                                       x_mask=x_mask, update_conf=update_conf, 
                                       pseudo_target=pseudo_tgt)
        else:
            loss, other = loss_cont_fn(atten,index, topk=target_topk, x_mask=x_mask, update_conf=update_conf)
        # log loss
        loss_cont_log.update(loss.item())
        # log accuracy
        acc = accuracy(other['score'], boxes_ori_tensor, union_ref_boxes, phrase_mask)[0]
        acc_atten.update(acc * 100, num_phrase.sum().item())
        acc = accuracy(other['target'], boxes_ori_tensor, union_ref_boxes, phrase_mask)[0]
        acc_target.update(acc * 100, num_phrase.sum().item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss = loss.item() + loss_prefix.item()
        loss.backward()

       
# ==================================================================================================================== #
#                                           calculate inner similarity                                                  #
# ==================================================================================================================== #

        # score = get_inner_similarity(mix_features_pad, temperature=0.025)
        # score = score.cpu().numpy()
        # from matplotlib import pyplot as plt
        # plt.cla()
        # plt.clf()
        # plt.imshow(score,cmap='hot',interpolation='nearest')
        # plt.colorbar()
        # plt.savefig("heat_{}_{}.png".format(epoch * len(train_loader) + i, args.batch_size))

        # from flow.inner_similarity import get_inner_similarity_by_unciom
        # score_ = get_inner_similarity_by_unciom(image_id, device=atten.device)
        # score_ = score_.cpu().numpy()
        # plt.cla()
        # plt.clf()
        # plt.imshow(score_,cmap='hot',interpolation='nearest')
        # plt.colorbar()
        # plt.savefig("heat_{}_{}_unicom.png".format(epoch * len(train_loader) + i, args.batch_size))

        # html_help_image_list = image_id * atten.shape[0]
        # score_list_2d = score.tolist()
        # score_unicom_list_2d = score_.tolist()
        # score_list = [elx for el in score_list_2d for elx in el]
        # score_unicom = [elx for el in score_unicom_list_2d for elx in el]
        # assert len(score_unicom) == len(score_list)
        # tuple_score = [(x,y) for x,y in zip(score_list, score_unicom)]
        # from flow.write_html_cropped_image import html_helper_inner_batch
        # html_helper_inner_batch(tuple_score, html_help_image_list, "diff_com.html", atten.shape[0])

        cutils.writer('add_scalars', "[model grad norm]", {'before clip': cutils.grad_norm_p(model.parameters())})
        if args.clip_norm > 0:
            nn.utils.clip_grad_norm_(parameters=[p for p in model.parameters() if p.requires_grad and p.grad is not None],
                                     max_norm=args.clip_norm, error_if_nonfinite=True)
            cutils.writer('add_scalars', "[model grad norm]", {'after clip': cutils.grad_norm_p(model.parameters())})
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            cutils.writer_var_norm(phrase, 'encoder', 'phrase', global_step=epoch * len(train_loader) + i)
            cutils.writer_grad_norm(phrase, 'encoder', 'phrase', global_step=epoch * len(train_loader) + i)
            cutils.writer_var_norm(region, 'encoder', 'region', global_step=epoch * len(train_loader) + i)
            cutils.writer_grad_norm(region, 'encoder', 'region', global_step=epoch * len(train_loader) + i)
            cutils.writer_region_inner_similarity(region, 'encoder','region', global_step=epoch * len(train_loader) + i, region_mask=region_mask, image_id=image_id)
            progress.display(i)
            # print("mask similarity item timing consum:",ted-tst)
            # if inner_mask is not None:
            #     print("the fuzzy pairs nums is:{}".format(inner_mask.sum() / inner_mask.shape[0]))
            #

    writer.add_scalars('acc', {
        f'Train Top1 Acc atten': acc_atten.avg,
        f'Train Top1 Acc target': acc_target.avg,
    }, global_step=epoch)


@torch.no_grad()
def val(model, test_loader, args, epoch, writer, device='cuda', prefix='eval'):
    print('==> Evaluation...')
    model.eval()
    top1_acc = AverageMeter("Top1")
    top5_acc = AverageMeter("Top5")
    for i, batch in enumerate(test_loader):
        # move to device
        batch = move_to_device(batch, device)
        (index,
         image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
         caption, caption_ids, span_mask, length,
         phrases, mix_phrases_id, num_phrase,
         ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
         ) = batch
        # encode
        if args.arch == 'dual_zero_nov' or args.arch == 'dual_zero_nov_att':
            # phrase, region, label_emb = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad, mix_num_boxes)
            phrase, region, label_emb = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
        elif args.arch == 'dual_mom':
            phrase, region, label_emb, momentum_out = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
        elif args.arch != 'mat':
            phrase, region = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
            #print(caption_ids)
        else:
            phrase, region = model(mix_phrases_id, mix_objs_id, mix_features_pad)
        # if args.normalize_feature:
        #     phrase = phrase / (phrase.norm(dim=-1, keepdim=True) + feps(phrase.dtype))
        #     region = region / (region.norm(dim=-1, keepdim=True) + feps(region.dtype))
        atten = einsum('b q d, b k d -> b q k', phrase, region)
        # mask
        phrase_mask = len2mask(num_phrase, phrase.shape[:2]).bool()
        region_mask = len2mask(mix_num_boxes, region.shape[:2]).bool()
        x_mask = einsum('b q, b k -> b q k', phrase_mask, region_mask)
        atten = atten.masked_fill_(~x_mask, -finf(atten.dtype))
        acc1, acc5 = accuracy(atten, boxes_ori_tensor, union_ref_boxes, phrase_mask, topk=(1, 5))
        top1_acc.update(acc1 * 100, num_phrase.sum().item())
        top5_acc.update(acc5 * 100, num_phrase.sum().item())

    print(f'{epoch} Accuracy is {top1_acc.avg:.2f} {top5_acc.avg:.2f}')
    writer.add_scalars('acc', {
        f'{prefix} Top1 Acc': top1_acc.avg,
        f'{prefix} Top5 Acc': top5_acc.avg,
    }, global_step=epoch)
    return top1_acc.avg


@torch.no_grad()
def soft_init(model, train_loader, args, loss_cont_fn, writer, device='cuda'):
    """
    init loss_fn confidence by phrase-label similarity
    """
    print('==> soft init...')
    model.eval()
    top1_acc = AverageMeter("Top1")
    top5_acc = AverageMeter("Top5")
    for i, batch in enumerate(tqdm(train_loader)):
        # move to device
        batch = move_to_device(batch, device)
        (index,
         image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
         caption, caption_ids, span_mask, length,
         phrases, mix_phrases_id, num_phrase,
         ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
         ) = batch
        # encode
        if args.arch == 'dual_zero_nov' or args.arch == 'dual_zero_nov_att':
            phrase, region, label_emb = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
            phrase = phrase / (phrase.norm(dim=-1, keepdim=True) + feps(phrase.dtype))
            region = region / (region.norm(dim=-1, keepdim=True) + feps(region.dtype))
            label_emb = label_emb / (label_emb.norm(dim=-1, keepdim=True) + feps(label_emb.dtype))
        elif args.arch == 'dual_mom':
            phrase, region, label_emb, momentum_out = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
        elif args.arch != 'mat':
            phrase, region = model(caption_ids, span_mask, length, mix_objs_id, mix_features_pad)
        else:
            phrase, region = model(mix_phrases_id, mix_objs_id, mix_features_pad)
        atten_0 = einsum('b q d, b k d -> b q k', phrase, label_emb)
        atten = einsum('b q d, b k d -> b q k', phrase, region)


        # breakpoint()
        # mask
        phrase_mask = len2mask(num_phrase, phrase.shape[:2]).bool()
        region_mask = len2mask(mix_num_boxes, region.shape[:2]).bool()
        x_mask = einsum('b q, b k -> b q k', phrase_mask, region_mask)
        atten_0 = atten_0.masked_fill_(~x_mask, -finf(atten_0.dtype))
        atten = atten.masked_fill_(~x_mask, -finf(atten.dtype))
        conf_0 = atten_0.softmax(dim=-1)
        conf_0 = conf_0.masked_fill_(~x_mask, 0)
        conf = atten.softmax(dim=-1)
        conf = conf.masked_fill_(~x_mask, 0)

        
        # loss_cont_fn.confidence_update(conf, index, x_mask, hard=False, conf_ema_m=0.)
        loss_cont_fn.set_batch_confidence(conf_0, index)
        # breakpoint()
        acc1, acc5 = accuracy(conf, boxes_ori_tensor, union_ref_boxes, phrase_mask, topk=(1, 5))
        top1_acc.update(acc1 * 100, num_phrase.sum().item())
        top5_acc.update(acc5 * 100, num_phrase.sum().item())

    mmc = loss_cont_fn.confidence.amax(dim=-1).sum() / train_loader.dataset.num_phrase.sum()
    print(f'soft_init Accuracy is {top1_acc.avg:.2f} {top5_acc.avg:.2f} mmc is {mmc:.4f}')
    writer.add_scalars('acc', {
        f'soft_init Top1 Acc': top1_acc.avg,
        f'soft_init Top5 Acc': top5_acc.avg,
    }, global_step=0)
    return top1_acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


def move_to_device(batch, device):
    if device == 'cpu':
        return batch
    (index,
     image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
     caption, caption_ids, span_mask, length,
     phrases, mix_phrases_id, num_phrase,
     ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
     ) = batch

    should_stack = (index,
                    mix_features_pad, mix_objs_id, mix_num_boxes,
                    caption_ids, span_mask,
                    mix_phrases_id, num_phrase,
                    union_ref_boxes, boxes_ori_tensor,
                    )
    stacked = map(lambda x: x.cuda(non_blocking=True), should_stack)
    (index,
     mix_features_pad, mix_objs_id, mix_num_boxes,
     caption_ids, span_mask,
     mix_phrases_id, num_phrase,
     union_ref_boxes, boxes_ori_tensor,
     ) = stacked

    batch = (index,
             image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
             caption, caption_ids, span_mask, length,
             phrases, mix_phrases_id, num_phrase,
             ref_boxes, boxes_ori, union_ref_boxes, boxes_ori_tensor,
             )
    return batch


if __name__ == '__main__':
    main()
