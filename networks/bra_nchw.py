"""
Refactored Bi-level Routing Attention that takes NCHW input.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
import torch
import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x
    
from typing import Optional, Tuple

class nchwBRA(nn.Module):
    """Bi-Level Routing Attention that takes nchw input

    Compared to legacy version, this implementation:
    * removes unused args and components
    * uses nchw input format to avoid frequent permutation

    When the size of inputs is not divisible by the region size, there is also a numerical difference
    than legacy implementation, due to:
    * different way to pad the input feature map (padding after linear projection)
    * different pooling behavior (count_include_pad=False)

    Current implementation is more reasonable, hence we do not keep backward numerical compatiability
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4,  side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 # NOTE: to be consistent with old models.

        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ regional routing setting #################
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        ##########################################

        self.qkv_linear = nn.Conv2d(self.dim, 3*self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x:Tensor, ret_attn_mask=False, glass_mask=None):
        """
        Args:
            x: NCHW tensor, better to be channel_last (https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
        Return:
            NCHW tensor
        """
        x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
        
        N, C, H, W = x.size()
        region_size = (H//self.n_win, W//self.n_win)

        # STEP 1: linear projection
        qkv = self.qkv_linear.forward(x) # ncHW
        q, k, v = qkv.chunk(3, dim=1) # ncHW
       
        # STEP 2: region-to-region routing
        # NOTE: ceil_mode=True, count_include_pad=False = auto padding
        # NOTE: gradients backward through token-to-token attention. See Appendix A for the intuition.
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) # nchw
        q_r:Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2) # n(hw)c
        k_r:Tensor = k_r.flatten(2, 3) # nc(hw)
        a_r = q_r @ k_r # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) # n(hw)k long tensor
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        # STEP 3: token to token attention (non-parametric function)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size,
                                        glass_mask=glass_mask
                                       )
        
        output = output + self.lepe(v) # ncHW
        output = self.output_linear(output) # ncHW

        if ret_attn_mask:
            return output, attn_mat

        output = output.permute(0, 2, 3, 1)

        return output

def _grid2seq(x:Tensor, region_size:Tuple[int], num_heads:int):
    """
    Args:
        x: BCHW tensor
        region size: int
        num_heads: number of attention heads
    Return:
        out: rearranged x, has a shape of (bs, nhead, nregion, reg_size, head_dim)
        region_h, region_w: number of regions per col/row
    """
    B, C, H, W = x.size()
    region_h, region_w =  H//region_size[0],  W//region_size[1]
    x = x.view(B, num_heads, C//num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2) # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w


def _seq2grid(x:Tensor, region_h:int, region_w:int, region_size:Tuple[int]):
    """
    Args: 
        x: (bs, nhead, nregion, reg_size^2, head_dim)
    Return:
        x: (bs, C, H, W)
    """
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    # print(x.shape, (bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim))
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead*head_dim,
        region_h*region_size[0], region_w*region_size[1])
    return x

def get_correspondence(mask):
    one = torch.ones_like(mask)
    mask = torch.where(mask < 0.1, one, one*2)
    mm = mask @ mask.transpose(-2, -1)
    one = torch.ones_like(mm)
    mm = torch.where(mm==2, one, one*0.2)
    # mm = torch.unsqueeze(mm, dim=1)
    return mm


def regional_routing_attention_torch(
    query:Tensor, key:Tensor, value:Tensor, scale:float,
    region_graph:LongTensor, region_size:Tuple[int],
    kv_region_size:Optional[Tuple[int]]=None,
    auto_pad=True, glass_mask=None)->Tensor:
    """
    Args:
        query, key, value: (B, C, H, W) tensor
        scale: the scale/temperature for dot product attention
        region_graph: (B, nhead, h_q*w_q, topk) tensor, topk <= h_k*w_k
        region_size: region/window size for queries, (rh, rw)
        key_region_size: optional, if None, key_region_size=region_size
        auto_pad: required to be true if the input sizes are not divisible by the region_size
    Return:
        output: (B, C, H, W) tensor
        attn: (bs, nhead, q_nregion, reg_size, topk*kv_region_size) attention matrix
    """
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()
    
    # Auto pad to deal with any input size 
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b)) # zero padding
        
        if glass_mask is not None:
            ## NOTE: padding glass mask here
            glass_mask = F.pad(glass_mask, (0, q_pad_r, 0, q_pad_b)) # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b)) # zero padding
    
    # print(query.shape, key.shape, value.shape)
    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)
    # print(query.shape, key.shape, value.shape, '---')

    # gather key and values.
    # TODO: is seperate gathering slower than fused one (our old version) ?
    # torch.gather does not support broadcasting, hence we do it manually
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1).\
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim).\
        expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
        index=broadcasted_region_graph) # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    
    # print(key_g.shape, value_g.shape, '...')
    # import pdb 
    # pdb.set_trace()

    # token-to-token attention
    # (bs, nhead, q_nregion, region_size, head_dim) @ (bs, nhead, q_nregion, head_dim, topk*kv_region_size)
    # -> (bs, nhead, q_nregion, region_size, topk*kv_region_size)
    # TODO: mask padding region
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)

    #####
    if glass_mask is not None:
        attn = _seq2grid(attn, region_h=q_region_h, region_w=q_region_w, region_size=region_size)
        attn = attn + glass_mask 
        attn,_,_ = _grid2seq(attn, region_size=region_size, num_heads=nhead)
    #####

    attn = torch.softmax(attn, dim=-1)

    # (bs, nhead, q_nregion, region_size, topk*kv_region_size) @ (bs, nhead, q_nregion, topk*kv_region_size, head_dim)
    # -> (bs, nhead, q_nregion, region_size, head_dim)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    # remove paddings if needed
    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class deform_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        self.deform = nchwBRA(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x, glass_mask=None):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.deform(self.norm1(x), glass_mask=glass_mask)) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                x = x + self.drop_path(self.deform(self.norm1(x), glass_mask=glass_mask)) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.deform(x, glass_mask=glass_mask))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.deform(x, glass_mask=glass_mask))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


if __name__ == '__main__':
    print('000')
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    x = torch.randn(2, 64, 32, 32).cuda()
    model = nchwBRA(dim=64, num_heads=1).cuda()
    
    print(model(x).shape, '----')
    


