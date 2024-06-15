import os
import torch
from torch import nn, Tensor, einsum
from transformers import BertConfig
from flash_attn.models.bert import create_mixer_cls, create_mlp_cls
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
from torchvision.ops import StochasticDepth

class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls=None,
        mlp_cls=None,
        norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
        fused_dropout_add_ln=False,
        return_residual=False,
    ):
        super().__init__()
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode="row")
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert layer_norm_fn is not None, "Triton is not installed"
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(
                self.dropout1, nn.Dropout
            )
            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        x: Tensor,
        x_kv: Tensor = None,
        mixer_kwargs=None,
    ):
        mixer_out = self.mixer(
            x, x_kv=x_kv, **(mixer_kwargs if mixer_kwargs is not None else {})
        )
        if self.return_residual:  # mixer out is actually a pair here
            mixer_out, x = mixer_out
        if not self.fused_dropout_add_ln:
            x = self.norm1(
                (self.drop_path1(self.dropout1(mixer_out)) + x).to(
                    dtype=self.norm1.weight.dtype
                )
            )
        else:
            if self.drop_path1.p == 0 or not self.training:
                rowscale1 = None
            else:
                rowscale1 = self.drop_path1(
                    torch.ones(
                        mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype
                    )
                )
            x = layer_norm_fn(
                mixer_out,
                self.norm1.weight,
                self.norm1.bias,
                residual=x,
                eps=self.norm1.eps,
                dropout_p=self.dropout1.p if self.training else 0.0,
                rowscale=rowscale1,
                prenorm=False,
                is_rms_norm=isinstance(self.norm1, RMSNorm)
            )
        if not isinstance(self.mlp, nn.Identity):
            mlp_out = self.mlp(x)
            if self.return_residual:  # mlp out is actually a pair here
                mlp_out, x = mlp_out
            if not self.fused_dropout_add_ln:
                x = self.norm2(
                    (self.drop_path2(self.dropout2(mlp_out)) + x).to(
                        dtype=self.norm2.weight.dtype
                    )
                )
            else:
                if self.drop_path2.p == 0 or not self.training:
                    rowscale2 = None
                else:
                    rowscale2 = self.drop_path2(
                        torch.ones(
                            mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype
                        )
                    )
                x = layer_norm_fn(
                    mlp_out,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=x,
                    eps=self.norm2.eps,
                    dropout_p=self.dropout2.p if self.training else 0.0,
                    rowscale=rowscale2,
                    prenorm=False,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            return x
        
def create_block(config, layer_idx=None, cross_attn=False):
    mixer_cls = create_mixer_cls(config, cross_attn=cross_attn, return_residual=True)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=True)
    norm_cls = RMSNorm # partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=True,
    )
    return block

class Encoder(nn.Module):
    def __init__(self, config: BertConfig, cross_attn=False):
        super().__init__()
        self.use_flash_attn = getattr(config, "use_flash_attn", False)
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i, cross_attn=cross_attn) for i in range(config.num_hidden_layers)]
        )

    def forward(self, x, max_seqlen, cu_seqlens, x_kv_dict=None):
        mixer_kwargs = {"max_seqlen": max_seqlen, "cu_seqlens": cu_seqlens}
        if x_kv_dict is not None:
            mixer_kwargs.update({'max_seqlen_k': x_kv_dict['max_seqlen'], 'cu_seqlens_k': x_kv_dict['cu_seqlens']})
        x_kv = x_kv_dict["x"] if x_kv_dict is not None else None
        for layer in self.layers:
            x = layer(x, x_kv=x_kv, mixer_kwargs=mixer_kwargs)
        return x
    
exists = lambda val: val is not None
    
class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x_dict):
        device = x_dict["x"].device
        max_seqlen, cu_seqlens, indices = [x_dict[k] for k in ["max_seqlen", "cu_seqlens", "indices"]]
        batch = len(cu_seqlens) - 1

        pos = torch.arange(max_seqlen, device=device)
        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        emb = emb*self.scale
        emb = emb.repeat(batch, 1)
        return emb[indices]
    
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_out: int,
        depth: int,
        heads: int,
        emb_dropout: float = 0.,
        hidden_dropout: float = 0.,
        attn_dropout: float = 0.,
        head_dropout: float = 0.,
    ):
        super().__init__()
        config = BertConfig(hidden_size=dim,
                            num_hidden_layers=depth,
                            num_attention_heads=heads,
                            intermediate_size=dim*4,
                            hidden_act='gelu',
                            hidden_dropout_prob=hidden_dropout,
                            attention_probs_dropout_prob=attn_dropout,
                            use_flash_attn=True,
                            fused_dropout_add_ln=True,
                           )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.cross_encoder = Encoder(config, cross_attn=True)
        self.encoder = Encoder(config)
        
        self.head = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Tanh(),
            nn.Dropout(head_dropout),
            nn.Linear(dim,dim_out),
        )
        
    def forward_single(self, x_dict, x_kv_dict):
        x, max_seqlen, cu_seqlens, indices = [x_dict[k] for k in ["x", "max_seqlen", "cu_seqlens", "indices"]]
        batch = len(cu_seqlens)-1
        x = self.emb_dropout(x)
        x = self.cross_encoder(x, max_seqlen, cu_seqlens, x_kv_dict)
        x = self.encoder(x, max_seqlen, cu_seqlens)        
        x = pad_input(x, indices, batch, max_seqlen)
        x = x.mean(dim=1)
        return x
    
    def forward(self, xs_dict):
        x1 = self.forward_single(xs_dict["1"], xs_dict["2"])
        x2 = self.forward_single(xs_dict["2"], xs_dict["1"])
        x = (x1+x2) / 2
        x = self.head(x)
        return x