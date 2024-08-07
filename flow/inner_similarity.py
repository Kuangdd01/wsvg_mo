import torch
from typing import List
from einops import rearrange, repeat


@torch.no_grad()
def get_inner_similarity(region_feature: torch.Tensor, temperature):
    # norm backbone features
    region_feature = region_feature / (region_feature.norm(dim=-1, keepdim=True) + 
                                       torch.finfo(region_feature.dtype).eps)
    inner_sim = torch.einsum('b k d, a q d -> b a k q',region_feature,region_feature)
    inner_sim /= temperature
    inner_sim_k = inner_sim.max(dim=-1)[0] #[b a k]
    inner_sim_b = inner_sim_k.mean(dim=-1) #[b a]
    score = torch.softmax(inner_sim_b, dim=-1) #[b,b]
    return score

def image_io(image_id, preprocess, device="cpu"):
    from PIL import Image
    tensor_list = []
    flickr_image_root = "your_root"
    image_path_list = [flickr_image_root + str(img) + ".jpg" for img in image_id]
    for path in image_path_list:
        image = Image.open(path).convert('RGB')
        transform_image = preprocess(image)
        tensor_list.append(transform_image)
    image_tensor = torch.stack(tensor_list).to(device=device)
    return image_tensor

@torch.no_grad()
def get_inner_similarity_by_unciom(image_id: List, device="cpu") -> torch.Tensor:
    import unicom
    model, preprocess = unicom.load("ViT-B/16")
    model.to(device=device)
    input_image_tensors = image_io(image_id, preprocess, device)
    assert input_image_tensors.device == next(model.parameters()).device
    unicom_out = model(input_image_tensors) #[B,DIM]
    sim = torch.einsum('b d, a d -> b a', unicom_out, unicom_out)
    score = torch.softmax(sim, dim=-1)
    return score


# @torch.no_grad()
def get_mask_similarity_matrix_by_threshold(inner_sim: torch.Tensor,
                                            atten: torch.Tensor, threshold: float, fill_value: float,
                                              device='cpu', )-> torch.tensor:
    atten_back = atten.clone()
    b, k = atten.shape[0], atten.shape[-1]
    eye_mask = torch.eye(b,dtype=bool,device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, k, k)
    inner_sim.masked_fill_(eye_mask, 0.)
    bool_mask = (inner_sim >= threshold).to(device=device) #[b,b,k,k]

    # position_tensor = bool_mask.nonzero()  #[nums, 4]
    # atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    
    pos = bool_mask.nonzero(as_tuple=True)
    atten_mask = torch.zeros(atten.shape, dtype=bool, device=device)
    a, b, c, d = pos
    # import ipdb
    # ipdb.set_trace()
    atten_mask[a, b, :10, d] = True
    atten_mask[b, a, :10, c] = True
    # print(atten_mask.sum())
    # mask_logits = atten * atten_mask #[b,b,q,k]

    atten.masked_fill_(atten_mask, fill_value)
    return atten, atten_back, atten_mask

    
# TODO 
"""
input: masked_logits [b1 b2 q k] non-diagonal sparse matrix here
=> for [: b2 q k] [a b q k] [a q b k] q from a k from b
=> pseudo label := [k_e * qs_e] #shape=[n_k, q_num]
=> partial loss = sum(pseudo label * logits(qk)) / sum(log_neg_logits))
calculate: sparse matrix [a b q k] => [a q (b k)]
total matrix: [a1 b1 q1 k1] => [i i q k]

output = output / self.temperature
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            logit = log_softmax(output, dim=-1)  ===for every logits div / logsum(negative logits)
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
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature

        masked_atten_pos = masked_atten.nonzero()

"""
def converting(atten_mask: torch.Tensor, origin_atten: torch.Tensor, 
               diag_logits: torch.Tensor, device=None):
    # masked_pos = masked_atten.nonzero()
    # shape of origin_atten:[b q bk]
    # mask[b a q k]
    # import ipdb
    # ipdb.set_trace()
    def log_softmax(x, dim=-1):
        maxv = x.amax(dim=dim, keepdim=True)
        x = x - maxv
        x = x - torch.logsumexp(x, dim=dim, keepdim=True)
        return x
    b, a, q, k = origin_atten.shape

    reshape_origin_att = rearrange(origin_atten, 'b a q k -> b q (a k)',b=b, q=q, a=a, k=k)

    msk_output = log_softmax(reshape_origin_att, dim=-1)
    #TODO reshape it to b q b k Then dot mul mask =>
    
    msk_output = rearrange(msk_output, 'b q (a k) -> b q a k',b=b, q=q, a=a, k=k)
    atten_mask = rearrange(atten_mask, 'b a q k -> b q a k',b=b, q=q, a=a, k=k)
    msk_output = msk_output * atten_mask #[b q b k] 
    
    # import ipdb
    # ipdb.set_trace()
    """
    diag_logits : [b q k]
    """
    eye_mask = torch.eye(b,dtype=bool,device=device).unsqueeze(-1).unsqueeze(-1)
    eye_mask = eye_mask.expand(-1, -1, q, k) #[b b q k]
    origin_att_logit = diag_logits * eye_mask
    origin_att_logit = rearrange(origin_att_logit, 'b a q k -> b q a k',b=b, q=q, a=a, k=k)

    #add diag elements
    tmp_logits = origin_att_logit + msk_output

    msk_zero_mask = (msk_output == 0)

    msk_output_logit = torch.masked_fill(tmp_logits, msk_zero_mask, -torch.finfo(msk_output.dtype).max).to(msk_output.device)
    msk_output_tmp = torch.masked_fill(msk_output, msk_zero_mask, -torch.finfo(msk_output.dtype).max).to(msk_output.device)
    # msk_output_logit = rearrange(msk_output_logit, 'b a q k -> b q a k',b=b, q=q, a=a, k=k)
    msk_output_logit = rearrange(msk_output_logit, 'b q a k -> b q (a k)',b=b, q=q, a=a, k=k)
    msk_output_tmp = rearrange(msk_output_tmp, 'b q a k -> b q (a k)',b=b, q=q, a=a, k=k)

    
    # limited the number of qk pairs in calculating
    topk_mask = torch.zeros_like(msk_output_logit, dtype=torch.bool, device=device)
    _, indices = msk_output_tmp.topk(50, dim=-1)
    topk_mask.scatter_(2, indices, 1)
    msk_output_logit = msk_output_logit * topk_mask

    converted_pseudo_weight = torch.softmax(msk_output_logit, dim=-1) #[b q ak]
    converted_pseudo_weight = rearrange(converted_pseudo_weight, 'b q (a k) -> b q a k',b=b, q=q, a=a, k=k)
    converted_pseudo_weight = rearrange(converted_pseudo_weight, 'b q a k -> b a q k',b=b, q=q, a=a, k=k)

    converted_pseudo_weight = converted_pseudo_weight.masked_fill(eye_mask, 0.)
    converted_pseudo_weight = rearrange(converted_pseudo_weight, 'b a q k -> b q a k',b=b, q=q, a=a, k=k)


    converted_loss = converted_pseudo_weight * msk_output * (-1) #b q b k
    # c_loss = converted_loss.sum(dim=1).sum()
    c_loss = converted_loss.sum(dim=2).sum()  
    # import ipdb
    # ipdb.set_trace()
    return c_loss
#TODO concatenate the negative samples to origin positive samples := converted_pseudo_weight has some issues.
#TODO test this workaround

