# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:57:58 2022

"""
import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from torch import einsum
from einops import rearrange, repeat

from . import debugger as cutils

def exist(x):
    return x is not None


def default(val, d):
    return val if exist(val) else d


def feps(dtype):
    return torch.finfo(dtype).eps


def finf(dtype):
    return torch.finfo(dtype).max


def log_softmax(x, dim=-1):
    maxv = x.amax(dim=dim, keepdim=True)
    x = x - maxv
    x = x - torch.logsumexp(x, dim=dim, keepdim=True)
    return x

#all masked 1
def get_negative_mask(batchsize: int, neg: int = None, device='cpu'):
    """
    only 'neg' negative samples are masked as 1, others are 0.
    """
    if (not exist(neg)) or neg >= batchsize - 1:
        return torch.ones(batchsize, batchsize, dtype=torch.bool, device=device)
    identity = torch.eye(batchsize, batchsize, dtype=torch.bool, device=device)
    mask = identity.clone().detach()
    for st in range(1, neg + 1):
        mask += torch.roll(identity, shifts=-1 * st, dims=0)
    return mask

def get_negative_mask_(batchsize: int, neg: int = None, device='cpu'):
    msk = torch.eye(batchsize, device=device)
    msk = ~msk.bool()
    msk = msk.float()
    return msk

class ConLoss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1., no_contrastive=False, neg_num=None, args=None):
        super().__init__()
        rearrange(confidence, 'n q k -> n q k') #assert shape?
        self.n, self.q, self.k = confidence.shape
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m
        self.hard = hard
        self.temperature = temperature
        # if args.var_t:
        #     self.temperature = nn.Parameter(torch.FloatTensor(1),requires_grad=True).cuda()
        #     self.temperature.data.fill_(0.1)
        self.base_temperature = base_temperature
        
        self.no_contrastive = no_contrastive
        self.neg_num = neg_num
        if exist(neg_num):
            assert no_contrastive is False

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start
    
    def set_batch_confidence(self, att, index):
        self.confidence[index] = att

    @torch.no_grad()
    def confidence_update(self, temp_conf, batch_index, x_mask=None, hard=None, conf_ema_m=None):
        """
        Parameters
        ----------
        temp_conf :
            conf should be p not logit
        batch_index :
        x_mask :in this bathc, q,k both exist where confidence diff
        hard:
        conf_ema_m:
        Returns
        -------
        None.

        """
        rearrange(temp_conf, 'b q k -> b q k', q=self.q, k=self.k)


        hard = default(hard, self.hard)
        if hard:
            #b,q,k
            temp_conf = F.one_hot(
                temp_conf.argmax(dim=-1),
                temp_conf.shape[-1],
            )
            #max postion for 1
        if exist(x_mask):
            temp_conf = temp_conf.masked_fill_(~x_mask, 0)
        conf_ema_m = default(conf_ema_m, self.conf_ema_m)
        self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :] + (1 - conf_ema_m) * temp_conf

    def forward(self, output, batch_index, topk=None, x_mask=None, update_conf=True,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True):
        b, _, _, _ = output.shape
        cutils.breakpoint_if_find_debug_file()
        output = output / self.temperature
        cutils.breakpoint_if_nan_or_inf(output)
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            # only use the sim on the Diagonal
            logit = log_softmax(output, dim=-1)
        else:
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            # shape = [b, q, b * k]
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)  #sim score[b q (b2 k)]
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
            neg_masked_output = log_softmax(neg_masked_output, dim=-1) #[b, q, bk]
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            logit = einsum('i i q k -> i q k', neg_masked_output)
            
            cutils.breakpoint_if_nan_or_inf(logit)
        # loss for CL
        pseudo_target = self.confidence[batch_index]
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        loss = -(pseudo_target * logit).sum(-1)
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
            if update_conf:
                self.confidence_update(conf, batch_index, x_mask)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }

#reproduce
class ConLoss_ori(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1., no_contrastive=False, neg_num=None):
        super().__init__()
        rearrange(confidence, 'n q k -> n q k') #assert shape?
        self.n, self.q, self.k = confidence.shape
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m
        self.hard = hard
        self.temperature = temperature
        self.base_temperature = base_temperature
#default = None
        self.no_contrastive = no_contrastive
        self.neg_num = neg_num
#default = None
        if exist(neg_num):
            assert no_contrastive is False

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    @torch.no_grad()
    def confidence_update(self, temp_conf, batch_index, x_mask=None, hard=None, conf_ema_m=None):
        """
        Parameters
        ----------
        temp_conf :
            conf should be p not logit
        batch_index :
        x_mask :in this batch, q,k both exist where confidence diff
        hard:
        conf_ema_m:
        Returns
        -------
        None.

        """
        rearrange(temp_conf, 'b q k -> b q k', q=self.q, k=self.k)
        hard = default(hard, self.hard)
        if hard:
            temp_conf = F.one_hot(
                temp_conf.argmax(dim=-1),
                temp_conf.shape[-1],
            )
            #max postion for 1
        if exist(x_mask):
            temp_conf = temp_conf.masked_fill_(~x_mask, 0)
        conf_ema_m = default(conf_ema_m, self.conf_ema_m)
        #confidence[bid,:] = [256:32:100]
        # self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :] + (1 - conf_ema_m) * temp_conf
        self.confidence[batch_index, :] = self.confidence[batch_index, :] 
    def forward(self, output, batch_index, topk=None, x_mask=None, update_conf=True,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True):
        """
        logit:
        target:
        score:
        """
        b, _, _, _ = output.shape
        cutils.breakpoint_if_find_debug_file()
        output = output / self.temperature
        cutils.breakpoint_if_nan_or_inf(output)
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            # only use the sim on the Diagonal
            logit = log_softmax(output, dim=-1)
        else:
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            # shape = [b, q, b * k]
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
            neg_masked_output = log_softmax(neg_masked_output, dim=-1)
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            
            logit = einsum('i i q k -> i q k', neg_masked_output)
            #breakpoint()
            cutils.breakpoint_if_nan_or_inf(logit)
        # loss for CL
        pseudo_target = self.confidence[batch_index]
        # pseudo_target = torch.ones_like(self.confidence[batch_index]) / 20
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        loss = -(pseudo_target * logit).sum(-1)
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)

        #breakpoint()
        #update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
            if update_conf:
                self.confidence_update(conf, batch_index, x_mask)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }


@torch.no_grad()
def top_k_mask(target, k, dim=-1):
    k = min(k, target.shape[dim])
    _, indices = torch.topk(target, k=k, dim=dim)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask.scatter_(dim, indices, 1)
    return mask


class ConLossCoLabel(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1., no_contrastive=False, neg_num=None, args=None):
        super().__init__()
        rearrange(confidence, 'n q k -> n q k') #assert shape?
        self.n, self.q, self.k = confidence.shape
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m
        self.hard = hard
        self.temperature = temperature
        self.base_temperature = base_temperature
        
#default = None
        self.no_contrastive = no_contrastive
        self.neg_num = neg_num
#default = None
        if exist(neg_num):
            assert no_contrastive is False

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start
    
    def set_batch_confidence(self, att, index):
        self.confidence[index] = att

    @torch.no_grad()
    def confidence_update(self, temp_conf, batch_index, x_mask=None, det_labels=None, hard=None, conf_ema_m=None):
       
        rearrange(temp_conf, 'b q k -> b q k', q=self.q, k=self.k)
        # rearrange(det_labels, 'b k -> b q k',q=self.q)
        #label idx [b, q, k]
        det_labels = rearrange(det_labels.repeat(1,self.q), 'b (q k) -> b q k', q=self.q, k=self.k)
        hard = default(hard, self.hard)
        if hard:
            temp_conf = F.one_hot(
                temp_conf.argmax(dim=-1),
                temp_conf.shape[-1],
            )
            #max postion for 1
        if exist(x_mask):
            temp_conf = temp_conf.masked_fill_(~x_mask, 0)
        co_label = det_labels * temp_conf#[bqk]
        co_label, _ = co_label.max(dim=-1, keepdim=True)
        temp_conf = (co_label == det_labels).to(dtype=torch.int64)
        conf_ema_m = default(conf_ema_m, self.conf_ema_m)
        self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :] + (1 - conf_ema_m) * temp_conf

    def forward(self, output, batch_index, det_labels=None, topk=None, x_mask=None, update_conf=True,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True):
        b, _, _, _ = output.shape
        cutils.breakpoint_if_find_debug_file()
        output = output / self.temperature
        cutils.breakpoint_if_nan_or_inf(output)
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            # only use the sim on the Diagonal
            logit = log_softmax(output, dim=-1)
        else:
            # breakpoint()
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            # shape = [b, q, b * k]
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
            neg_masked_output = log_softmax(neg_masked_output, dim=-1)
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            logit = einsum('i i q k -> i q k', neg_masked_output)
            #breakpoint()
            cutils.breakpoint_if_nan_or_inf(logit)
        # loss for CL
        pseudo_target = self.confidence[batch_index]
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        loss = -(pseudo_target * logit).sum(-1)
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
            if update_conf:
                self.confidence_update(conf, batch_index, x_mask, det_labels)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }

class ConLossMask(ConLoss):
    def __init__(self, confidence, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1, no_contrastive=False, neg_num=None, args=None):
        super().__init__(confidence, conf_ema_m, hard, temperature, base_temperature, no_contrastive, neg_num, args)
    
    def forward(self, output, duplicate_mask, batch_index, topk=None, x_mask=None, update_conf=True, return_atten=True, 
                return_logit=True, return_target=True, return_score=True, attn_mask=None, origin_attn=None):
        b, _, _, _ = output.shape
        cutils.breakpoint_if_find_debug_file()
        # duplicate_mask = duplicate_mask.bool().cuda()
        # duplicate_mask = repeat(duplicate_mask, 'b1 b2 -> b1 b2 q k',b1=b,b2=b, q=self.q, k=self.k)
        # duplicate_mask = rearrange(duplicate_mask, 'b1 b2 q k -> b1 q (b2 k)',b1=b, b2=b, q=self.q, k=self.k)
        output = output / self.temperature
        # cutils.breakpoint_if_nan_or_inf(output)
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            # only use the sim on the Diagonal
            logit = log_softmax(output, dim=-1)
        else:
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            # shape = [b, q, b * k]
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)

            output.masked_fill_(duplicate_mask, -finf(output.dtype))
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))


            neg_masked_output = log_softmax(neg_masked_output, dim=-1) #[b, q, bk]
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            logit = einsum('i i q k -> i q k', neg_masked_output)
            cutils.breakpoint_if_nan_or_inf(logit)
            # ipdb.set_trace()
        # loss for CL
        pseudo_target = self.confidence[batch_index]
        # set meaningless pairs sim to zero
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)            
        loss = -(pseudo_target * logit).sum(-1)

        if isinstance(attn_mask, torch.Tensor):
            masked_maxv = origin_attn.amax(dim=-1, keepdim=True)
            masked_atten = (origin_attn - masked_maxv) * attn_mask #[4 dim]
            pseudo_att = masked_atten * pseudo_target.unsqueeze(1) #[b1,b2,q,k]
            pseudo_loss = -(pseudo_att.sum(1).sum(-1))
            cutils.breakpoint_if_nan_or_inf(pseudo_loss)
            loss += pseudo_loss
            # ipdb.set_trace()
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        # ipdb.set_trace()
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
            if update_conf:
                pass
                # self.confidence_update(conf, batch_index, x_mask)
                #pass
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }
    

from flow.inner_similarity import converting
# TODO no pseudo label Matrix
class WSLoss(nn.Module):
    def __init__(self, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1., 
                 no_contrastive=False, neg_num=None, args=None):
        super().__init__()
        # rearrange(confidence, 'n q k -> n q k') #assert shape?
        # self.n, self.q, self.k = confidence.shape
        # self.confidence = confidence
        self.q = 32
        self.k = 100
        self.conf_ema_m = conf_ema_m
        # self.RMmodel = pseudo_generator
        self.hard = hard
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.no_contrastive = no_contrastive
        self.neg_num = neg_num
        if exist(neg_num):
            assert no_contrastive is False

    
    def forward(self, output, batch_index, topk=None, 
                x_mask=None, update_conf=True, 
                origin_att=None, inner_mask=None,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True,
                pseudo_target=None):
        b, _, q_, k_ = output.shape
        self.q = q_
        self.k = k_
        cutils.breakpoint_if_find_debug_file()
        # print("loss in",pseudo_target._version)
        output = output / self.temperature
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            logit = log_softmax(output, dim=-1)
        else:
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)  #sim score[b q (b2 k)]
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
            neg_masked_output = log_softmax(neg_masked_output, dim=-1) #[b, q, bk]
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            logit = einsum('i i q k -> i q k', neg_masked_output)
            
            cutils.breakpoint_if_nan_or_inf(logit)
        # loss for CL
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            # do it again
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        loss = -(pseudo_target * logit).sum(-1)
        if origin_att is not None:
            fake_loss = converting(inner_mask, origin_att, logit.clone(), logit.device)
            loss = (loss.sum() + 1 * fake_loss) / (phrase_mask.sum() + feps(loss.dtype))
        else:
            loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
            # if update_conf:
            #     self.confidence_update(conf, batch_index, x_mask)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }