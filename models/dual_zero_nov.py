import torch
import torch.nn as nn
from .layer import DynamicLSTM, REDUCE
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .cross_att import CrossAttention, CrossAttnProcessor
# helpers
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # zero init
        self.fc1.weight.data = torch.zeros(hidden_size, input_size)
        self.fc2.weight.data = torch.zeros(hidden_size, hidden_size)
        self.fc3.weight.data = torch.zeros(output_size,hidden_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            return self.norm(x)




class Dual(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, scale=12, reduce_method='sum', emb_dim=300, feature_dim=2048):
        super(Dual, self).__init__()
        self.eps = 1e-5
        self.scale = scale
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.linear_f = nn.Linear(feature_dim, emb_dim)

        self.rnn = DynamicLSTM(emb_dim, emb_dim, num_layers=1, bias=True, batch_first=True, dropout=0.,
                               bidirectional=True, only_use_last_hidden_state=False, rnn_type='LSTM')
        self.linear_rnn = nn.Linear(emb_dim * 2, emb_dim)
        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear_rnn.weight.data = torch.zeros(emb_dim, emb_dim * 2)
        self.linear_p.weight.data = torch.eye(emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)
        # self.mlp = SimpleMLP(feature_dim, hidden_size=768, output_size=emb_dim)
        # self.transforms_block = Transformer(dim=emb_dim, depth=6, heads=8, dim_head=64, mlp_dim=768, dropout=0.1)

        # query->   cross_att_dim-> 
        # self.cross_att = CrossAttention(query_dim=emb_dim, cross_attention_dim=emb_dim, bias=True, dropout=0.1)
        self.reduce_func = REDUCE[reduce_method]

    def encode_k(self, label, feature):
        label_emb = self.wv(label)
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        f_emb = self.linear_f(feature)
        # f_emb = self.mlp(feature)
        # f_emb = self.transforms_block(f_emb)
        ks_emb = k_emb + f_emb
        ks_emb = self.dropout(ks_emb)
        # k_emb += self.transforms_block(f_emb)
        # k_emb += f_emb
        # k_emb = self.dropout(k_emb)
        # k_emb = self.transforms_block(k_emb)
        return ks_emb, label_emb

    def encode_p(self, caption_id, phrase_span_mask, length):
        caption = self.wv(caption_id)
        hidden, _ = self.rnn(caption, length)
        hidden = self.linear_rnn(hidden)
        hidden = caption + hidden
        p_emb = self.reduce_func(phrase_span_mask, hidden)
        p_emb = p_emb / self.scale
        p_emb = self.linear_p(p_emb) + self.eps * self.linear_mini(p_emb)
        p_emb = self.dropout(p_emb)
        return p_emb

    def get_pseudo_target(self, p_emb, k_emb):
        # crs = torch.clone(self.cross_att.parameters())
        # p_s, k_s = p_emb.clone().detach(), k_emb.clone().detach()
        # cross_k_emb = self.cross_att(k_s, p_s)
        # print(cross_k_emb.grad)
        # cross_k_emb.retain_grad()
        # ps_emb = p_emb.detach().clone()
        # cross_similarity = torch.einsum('b q d, b k d -> b q k', ps_emb, cross_k_emb)
        # cross_similarity = torch.einsum('b q d, b k d -> b q k', p_emb, k_emb)
        
        cross_similarity = torch.einsum('b q d, b k d -> b q k', p_emb.detach(), k_emb.detach())
        return cross_similarity
    
    def forward(self, caption_id, phrase_span_mask, length, label, feature):
        p_emb = self.encode_p(caption_id, phrase_span_mask, length)
        k_emb, label_emb = self.encode_k(label, feature)
        return p_emb, k_emb, label_emb