import math
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from utils.registry import MODEL_REGISTRY

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SimpleDown(nn.Module):
    def __init__(self, in_channels, padding_mode="reflect"):
        super().__init__()
        self.out = nn.Conv2d(
            in_channels,in_channels * 2,kernel_size=2,stride=2,padding=0,bias=False,padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.out(x)

class SimpleUp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.out = nn.ConvTranspose2d(in_channels,in_channels // 2,kernel_size=2,stride=2,padding=0,bias=False,)

    def forward(self, x):
        return self.out(x)
'''
class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode="reflect"):
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(
            hidden_features, hidden_features * 2, 1, 1, 0, bias=bias
        )
        self.dwconv = nn.Conv2d(hidden_features * 2,hidden_features * 2,3,1,1,bias=bias,padding_mode=padding_mode,groups=hidden_features * 2,)
        self.project_out = nn.Conv2d(
            hidden_features, in_channels, kernel_size=1, bias=bias
        )
        self.mlp = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        shortcut = inp_feats[0]
        x = torch.cat(inp_feats, dim=1)
        x = self.pwconv(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return self.mlp(x + shortcut)
'''

class DAF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.convs = nn.Sequential(
            nn.Conv2d(2*in_channels, 4*in_channels, 3, padding=1, groups=2*in_channels),
            ChannelAttention(num_feat=4*in_channels),
            nn.Conv2d(4*in_channels, in_channels, 1)     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GELU()
        )
        self.out = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, cur, pre):
        x = torch.cat([cur, pre], dim=1)
        pre = self.conv1(pre) #c
        x = self.convs(x) * pre
        x = self.conv2(x) + cur
        x = self.out(x)
        return x

class ADZMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs,),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=8, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    @staticmethod
    def dt_init(dt_rank,d_inner,dt_scale=1.0,dt_init="random",dt_min=0.001,dt_max=0.1,dt_init_floor=1e-4,**factory_kwargs,):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def scan(self, x, rows, cols, mask):
        b, c, h, w = x.shape
        y = torch.zeros((b, c, rows * cols), device=x.device)
        y[:, :, :h * w] = x.view(b, c, -1)
        y = y.view(b, c, rows, cols)
        y = y.permute(0, 1, 3, 2)
        y = y.masked_select(mask).view(b,c,h*w)
        return y

    def merge(self, x, h, w, rows, cols, mask):
        b,c,_ = x.shape
        y = torch.zeros((b,c,cols,rows),device=x.device)
        y.masked_scatter_(mask, x.view(b,c,h*w))
        y = y.permute(0,1,3,2).contiguous().view(b,c,rows*cols)
        y = y[:,:,:h*w].view(b,c,h,w)
        return y
    
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 8

        rows, cols = math.ceil(H * W / (W + 1)), W + 1
        mask = torch.zeros((B, C, rows * cols), dtype=torch.bool, device=x.device)
        mask[:, :, :H * W] = True
        mask = mask.view(B, C, rows, cols).permute(0, 1, 3, 2)#w,h

        row_indices = torch.arange(x.size(2)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2
        col_indices = torch.arange(x.size(3)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2
        xs = torch.stack([
            torch.where(row_indices == 0,x,torch.flip(x, dims=[3])).view(B, -1, L),
            torch.where(col_indices == 0,torch.transpose(x, dim0=2, dim1=3).contiguous(),torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous(), dims=[3])).view(B, -1, L)
        ],dim=1).view(B, 2, -1, L)


        #-----------------
        oblique_x, anti_oblique_x = self.scan(x, rows, cols, mask).contiguous().view(B, -1, L), self.scan(torch.flip(x, dims=[-1]), rows, cols, mask).contiguous().view(B, -1, L)
        oblique_x = torch.stack([oblique_x,anti_oblique_x], dim=1).view(B, 2, -1, L)
        xs = torch.stack([xs,oblique_x],dim=1).view(B, 4, -1, L)
        #-----------------

        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (B, 8, C, L)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(xs,dts,As,Bs,Cs,Ds,z=None,delta_bias=dt_projs_bias,delta_softplus=True,return_last_state=False,
                                    ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y1 = out_y[:, 0].view(B,-1,H,W)
        y2 = out_y[:, 1].view(B,-1,W,H)
        y1 = torch.where(row_indices == 0, y1, torch.flip(y1, dims=[3])).view(B, -1, L)
        y2 = torch.transpose(torch.where(col_indices == 0, y2, torch.flip(y2, dims=[3])),dim0=2,dim1=3).contiguous().view(B,-1,L)

        y1_inv = torch.flip(out_y[:, 4], dims=[-1]).view(B,-1,H,W)
        y1_inv = torch.where(row_indices == 0, y1_inv, torch.flip(y1_inv, dims=[3])).view(B, -1, L)
        y2_inv = torch.flip(out_y[:, 5], dims=[-1]).view(B,-1,W,H)
        y2_inv = torch.transpose(torch.where(col_indices == 0, y2_inv, torch.flip(y2_inv, dims=[3])),dim0=2,dim1=3).contiguous().view(B,-1,L)


        y3 = self.merge(out_y[:, 2], H, W, rows, cols, mask).view(B, -1, L)
        y3_inv = self.merge(torch.flip(out_y[:, 6], dims=[-1]), H, W, rows, cols, mask).view(B, -1, L)

        y4 = self.merge(out_y[:, 3], H, W, rows, cols, mask).view(B, -1, H, W)
        y4 = torch.flip(y4, dims=[-1]).view(B, -1, L)

        y4_inv = self.merge(torch.flip(out_y[:, 7], dims=[-1]), H, W, rows, cols, mask).view(B, -1, H, W)
        y4_inv = torch.flip(y4_inv, dims=[-1]).view(B, -1, L)

        return y1,y2,y3,y4,y1_inv,y2_inv,y3_inv,y4_inv

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: tensor with shape (b,c,h,w)
        output: tensor with shape (b,c,h,w)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (b,h,w,c)

        xz = self.in_proj(x)  # (b,h,w,2*e*c)
        x, z = xz.chunk(2, dim=-1)  # both (b,h,w,e*c)

        x = x.permute(0, 3, 1, 2).contiguous()  # (b,e*c,h,w)
        x = self.act(
            self.conv2d(x)
        )  # main branch that will be selective scaned by SS2D

        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)#four directions scans
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8# merge the four scaned info (b,e*c,hw)
        y = (
            torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        )  # (b,h,w,e*c)
        y = self.out_norm(y)  # (b,h,w,e*c)
        y = y * F.silu(z)  # mutiplied by the other info flow after SiLU
        out = self.out_proj(y)  # (b,h,w,e*c) -> #(b,h,w,c)
        out = out.permute(0, 3, 1, 2)  # (b,c,h,w)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class NaiveMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W= x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y).permute(0, 3, 1, 2).contiguous()
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class RMamba(nn.Module):
    def __init__(self, in_channels, d_state):
        super().__init__()
        self.layer_norm_1 = LayerNorm(in_channels, data_format="channels_first")
        self.mamba = ADZMamba(d_model=in_channels, d_state=d_state)
        self.skip_scale = nn.Parameter(torch.ones(in_channels))
        self.layer_norm_2 = LayerNorm(in_channels, data_format="channels_first")
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.GELU())
        self.ca = ChannelAttention(num_feat=in_channels)
        self.skip_scale2 = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        shortcut = x  # (b,c,h,w)
        x = self.layer_norm_1(x)
        x = self.mamba(x)
        x = (shortcut.permute(0, 2, 3, 1) * self.skip_scale).permute(0, 3, 1, 2) + x
        shortcut = x  # (b,c,h,w)
        x = self.layer_norm_2(x)
        x = self.conv(x)
        x = self.ca(x)
        x = (shortcut.permute(0, 2, 3, 1) * self.skip_scale2).permute(0, 3, 1, 2) + x
        return x

class TowDGE(nn.Module):
    def __init__(self, in_channels, d_state):
        super().__init__()
        self.dn_branch = SDB(in_channels=in_channels)
        self.cc_branch = RMamba(in_channels=in_channels,d_state=d_state)
    def forward(self, x, state):
        assert state == 'dn' or state == 'cc'
        if state == 'dn':
            return self.dn_branch(x)
        else:
            return self.cc_branch(x)

class RDM(nn.Module):
    def __init__(self,in_channels=4, middle_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels+1, middle_channels, kernel_size=1, bias=True)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(middle_channels, 2*middle_channels, kernel_size=5, padding=2, bias=True, groups=middle_channels),
            nn.Conv2d(2*middle_channels, middle_channels, 3, padding=1, bias=True, padding_mode='reflect'),
            nn.GELU()
        )
        self.conv2 = nn.Conv2d(middle_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)  # (b, 1, h, w) prior
        input = torch.cat([img, mean_c], dim=1)#(b,5,h,w)
        x_1 = self.conv1(input)#(b,c,h,w)
        illu_fea = self.depth_conv(x_1)#(b,c,h,w)
        illu_map = self.conv2(illu_fea)#(b,4,h,w)
        return illu_fea, illu_map

class SDB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sdb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, padding_mode='reflect', groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )
    def forward(self, x):
        return  x + self.sdb(x)

@MODEL_REGISTRY.register()
class TSCDNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        base_channels=32,
        base_d_state=4,
    ):
        super().__init__()

        self.sqrt_inchannels = int(math.sqrt(in_channels))

        self.rdm = RDM(in_channels=in_channels, middle_channels=base_channels)
        self.light_feature_downs = nn.ModuleList(
            [SimpleDown(in_channels=base_channels * (2**idx)) for idx in range(3)]
        )

        #1,2,4,8
        self.dn_down_retinex_fuses = nn.ModuleList([
            DAF(in_channels=base_channels*(2**i))
            #GFM(in_channels=base_channels*(2**i))
            for i in range(4)
        ])
        self.cc_down_retinex_fuses = nn.ModuleList([
            DAF(in_channels=base_channels*(2**i))
            #SimpleFuse(in_channels=base_channels*(2**i))
            for i in range(4)
        ])

        self.padding_mode = "reflect"

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, 1, 2, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 5, 1, 2, padding_mode="reflect"),
        )
        self.dn_downs = nn.ModuleList(
            [SimpleDown(in_channels=base_channels * (2**idx)) for idx in range(3)]
        )
        self.cc_downs = nn.ModuleList(
            [SimpleDown(in_channels=base_channels * (2**idx)) for idx in range(3)]
        )

        self.towdge_blocks = nn.ModuleList([
            TowDGE(base_channels * (2**i),d_state=base_d_state * (2**i))
            for i in range(4)
        ])

        # four
        self.dn_decoder = nn.ModuleList([
            SDB(in_channels=base_channels*(2**i))
            for i in range(3, -1, -1)
        ])

        # three
        self.dn_ups = nn.ModuleList(
            [SimpleUp(in_channels=base_channels * (2**idx)) for idx in range(3, 0, -1)]
        )

        # three
        self.dn_fuses = nn.ModuleList([
            DAF(in_channels=base_channels*(2**i))
            for i in range(2, -1, -1)
        ])

        self.raw_out = nn.Sequential(
            nn.Conv2d(base_channels,base_channels,3,padding=1,padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 1),
        )

        self.re_fuses = nn.ModuleList([
            DAF(in_channels=base_channels*(2**i))
            for i in range(3)
        ])

        # four
        self.ccs = nn.ModuleList([
            RMamba(in_channels=base_channels * (2**i),d_state=base_d_state * (2**i))
            for i in range(3, -1, -1)
        ])

        # three
        self.cc_ups = nn.ModuleList(
            [SimpleUp(in_channels=base_channels * (2**idx)) for idx in range(3, 0, -1)]
        )

        # four
        self.cc_fuses = nn.ModuleList([
            DAF(in_channels=base_channels*(2**i))
            for i in range(2, -1, -1)
        ])

        self.rgb_out = nn.Sequential(
            nn.Conv2d(base_channels,base_channels,5,padding=2,padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(base_channels, 3*in_channels, 1),
            nn.PixelShuffle(self.sqrt_inchannels)
        )

    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = 2**3

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w + left_pad, top_pad, h + top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor

    def _check_and_crop(self, x, raw):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top * self.sqrt_inchannels : bottom * self.sqrt_inchannels, left * self.sqrt_inchannels : right * self.sqrt_inchannels]
        raw = raw[:, :, top:bottom, left:right] if raw is not None else None
        return x, raw

    def forward(self, x):
        x = self._check_and_padding(x)

        #reintex
        light_feature, light_map = self.rdm(x)#(b,in_channels,h,w),(b,c,h,w)
        x = x * light_map

        x = self.conv_in(x)
        temp = x

        light_features = [light_feature]# 1,2,4,8
        encoder_features = []  # 1,2,4
        for process, retinex_fuse, dn_down, illuminate_down in zip(
            self.towdge_blocks[:-1], self.dn_down_retinex_fuses[:-1], self.dn_downs, self.light_feature_downs
        ):
            retinex_fuse(x, light_feature)
            x = process(x, 'dn')
            encoder_features.append(x)
            x = dn_down(x)
            light_feature = illuminate_down(light_feature)
            light_features.append(light_feature)
        x = self.dn_down_retinex_fuses[-1](x, light_feature)
        x = self.towdge_blocks[-1](x, 'dn')
        

        #denoising decoding
        decoder_features = []# 4,2,1
        encoder_features.reverse()  # 1,2,4 -> 4,2,1
        
        x = self.dn_decoder[0](x)
    
        for dn_up, dn_fuse, denoise, encoder_feature in zip(
            self.dn_ups, self.dn_fuses,self.dn_decoder[1:], encoder_features
        ):
            x = dn_up(x)
            x = dn_fuse(x, encoder_feature)
            x = denoise(x)
            decoder_features.append(x)

        x = x + temp
        x = self.raw_out(x)
        raw = x

        x = temp

        #cc encoding
        encoder_features = []
        decoder_features.reverse()  # 4,2,1 -> 1,2,4
        
        for (retinex_fuse, process, refuse, cc_down, light_feature, decoder_feature) in zip(
            self.cc_down_retinex_fuses[:-1],
            self.towdge_blocks[:-1],
            self.re_fuses,
            self.cc_downs,
            light_features[:-1],
            decoder_features
        ):
            x = retinex_fuse(x, light_feature)
            x = process(x,'cc')
            encoder_features.append(x)  # 1,2,4
            x = refuse(x, decoder_feature)
            x = cc_down(x)
        x = self.cc_down_retinex_fuses[-1](x, light_features[-1])
        x = self.towdge_blocks[-1](x,'cc')
        

        ## cc decoding
        encoder_features.reverse()  # 1,2,4 -> 4,2,1

        for cc, cc_up, cc_fuse, encoder_feature in zip(
            self.ccs[:-1], self.cc_ups, self.cc_fuses, encoder_features
        ):
            x = cc(x)
            x = cc_up(x)
            x = cc_fuse(x, encoder_feature)
        x = self.ccs[-1](x)

        x = x + temp
        x = self.rgb_out(x)

        x, raw = self._check_and_crop(x, raw)

        return x, raw

'''
FLOPs: 113.640979008 G
Params: 6.220296 M
'''

def cal_model_complexity():
    import thop
    model = TSCDNet(in_channels=4, base_channels=32, base_d_state=4).cuda()
    x = torch.rand(1, 4, 512, 512).cuda()
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == "__main__":
    cal_model_complexity()
