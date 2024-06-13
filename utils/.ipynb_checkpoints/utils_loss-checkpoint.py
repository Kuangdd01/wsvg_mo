# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:57:58 2022

@author: chenkq
"""
import torch
from torch import nn
from torch.nn import functional as F

from torch import einsum
from einops import rearrange

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


class ConLoss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1.):
        super().__init__()
        rearrange(confidence, 'n q k -> n q k')
        self.n, self.q, self.k = confidence.shape
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m
        self.hard = hard
        self.temperature = temperature
        self.base_temperature = base_temperature

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
        x_mask :
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
        output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
        output = log_softmax(output, dim=-1)
        output = rearrange(output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
        cutils.breakpoint_if_nan_or_inf(output)
        logit = einsum('i i q k -> i q k', output)
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
        }


@torch.no_grad()
def top_k_mask(target, k, dim=-1):
    k = min(k, target.shape[dim])
    _, indices = torch.topk(target, k=k, dim=dim)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask.scatter_(dim, indices, 1)
    return mask
