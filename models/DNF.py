import torch
import torch.nn.functional as F
from abc import abstractmethod
from einops import rearrange
from torch import nn
from utils.registry import MODEL_REGISTRY

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
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
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResidualSwitchBlock(nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.block = block
        
    def forward(self, x, residual_switch):
        return self.block(x) + residual_switch * x

class SimpleDownsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0, bias=False, padding_mode=padding_mode)

    def forward(self, x):
        return self.body(x)

class SimpleUpsample(nn.Module):
    def __init__(self, dim, *, padding_mode='reflect'):
        super().__init__()
        self.body = nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        return self.body(x)

class MCC(nn.Module):
    def __init__(self, f_number, num_heads, padding_mode, bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(f_number, eps=1e-6, data_format='channels_first')
        self.f_number = f_number
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(f_number, f_number * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(f_number * 3, f_number * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=f_number * 3)
        self.project_out = nn.Conv2d(f_number, f_number, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(f_number, f_number, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(f_number, f_number, 3, 1, 1, bias=bias, groups=f_number, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):
        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.feedforward(out + x)
        return out
# CI
class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.dconv(x)

# Post-CI
class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

class CID(nn.Module):
    def __init__(self, f_number, padding_mode) -> None:
        super().__init__()
        self.channel_independent = DConv7(f_number, padding_mode)
        self.channel_dependent = MLP(f_number, excitation_factor=2)

    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))

class PDConvFuse(nn.Module):
    def __init__(self, in_channels=None, f_number=None, feature_num=2, bias=True, **kwargs) -> None:
        super().__init__()
        if in_channels is None:
            assert f_number is not None
            in_channels = f_number
        self.feature_num = feature_num
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(feature_num * in_channels, in_channels, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels, padding_mode='reflect')

    def forward(self, *inp_feats):
        assert len(inp_feats) == self.feature_num
        return self.dwconv(self.act(self.pwconv(torch.cat(inp_feats, dim=1))))

class GFM(nn.Module):
    def __init__(self, in_channels, feature_num=2, bias=True, padding_mode='reflect', **kwargs) -> None:
        super().__init__()
        self.feature_num = feature_num

        hidden_features = in_channels * feature_num
        self.pwconv = nn.Conv2d(hidden_features, hidden_features * 2, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, in_channels, kernel_size=1, bias=bias)
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




@MODEL_REGISTRY.register()
class DNF(nn.Module):
    def __init__(self, f_number=32,
                block_size=2,
                layers=4,
                denoising_block='CID',
                color_correction_block='MCC',
                feedback_fuse='GFM'
                ) -> None:
        super().__init__()
        def get_class(class_or_class_str):
            return eval(class_or_class_str) if isinstance(class_or_class_str, str) else class_or_class_str

        self.denoising_block_class = get_class(denoising_block)
        self.color_correction_block_class = get_class(color_correction_block)
        self.downsample_class = SimpleDownsample
        self.upsample_class = SimpleUpsample
        self.decoder_fuse_class = PDConvFuse

        self.padding_mode = 'reflect'
        self.act = nn.GELU()
        self.layers = layers

        head = [2 ** layer for layer in range(layers)]
        self.block_size = block_size
        inchannel = 3 if block_size == 1 else block_size * block_size
        outchannel = 3 * block_size * block_size

        self.feature_conv_0 = nn.Conv2d(inchannel, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)
        self.feature_conv_1 = nn.Conv2d(f_number, f_number, 5, 1, 2, bias=True, padding_mode=self.padding_mode)

        self.downsamples = nn.ModuleList([
            self.downsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(layers - 1)
        ])

        self.upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])

        self.denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )
            )
            for idx in range(layers)
        ])
        self.feedback_fuse_class = get_class(feedback_fuse)

        self.feedback_fuses = nn.ModuleList([
            self.feedback_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers)
        ])

        aux_outchannel = 3 if block_size == 1 else block_size * block_size
        self.aux_denoising_blocks = nn.ModuleList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2**idx),
                    padding_mode=self.padding_mode
                )   
            )
            for idx in range(layers)
        ])
        self.aux_upsamples = nn.ModuleList([
            self.upsample_class(
                f_number * (2**idx), 
                padding_mode=self.padding_mode
            )
            for idx in range(1, layers)
        ])
        self.denoising_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.aux_conv_fuse_0 = nn.Conv2d(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.aux_conv_fuse_1 = nn.Conv2d(f_number, aux_outchannel, 1, 1, 0, bias=True)

            
        self.color_correction_blocks = nn.ModuleList([
            self.color_correction_block_class(
                f_number=f_number * (2 ** idx),
                num_heads=head[idx],
                padding_mode=self.padding_mode,
            )
            for idx in range(layers)
        ])

        self.color_decoder_fuses = nn.ModuleList([
            self.decoder_fuse_class(in_channels=f_number * (2 ** idx)) for idx in range(layers - 1)
        ])

        self.conv_fuse_0 = nn.Conv2d(f_number, f_number, 3, 1, 1, bias=True, padding_mode=self.padding_mode)
        self.conv_fuse_1 = nn.Conv2d(f_number, outchannel, 1, 1, 0, bias=True)

        if block_size > 1:
            self.pixel_shuffle = nn.PixelShuffle(block_size)
        else:
            self.pixel_shuffle = nn.Identity()

    @abstractmethod
    def _pass_features_to_color_decoder(self, x, f_short_cut, encoder_features):
        ## denoising decoder
        denoise_decoder_features = []
        for denoise, up, fuse, encoder_feature in reversed(list(zip(
            self.aux_denoising_blocks[1:], 
            self.aux_upsamples, 
            self.denoising_decoder_fuses,
            encoder_features    
        ))):
            x = denoise(x, 1)
            denoise_decoder_features.append(x)
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.aux_denoising_blocks[0](x, 1)
        denoise_decoder_features.append(x)
        x = x + f_short_cut
        x = self.act(self.aux_conv_fuse_0(x))
        x = self.aux_conv_fuse_1(x)
        res1 = x

        ## feedback, local residual switch on
        encoder_features = []
        denoise_decoder_features.reverse()
        x = f_short_cut
        for fuse, denoise, down, decoder_feedback_feature in zip(
            self.feedback_fuses[:-1], 
            self.denoising_blocks[:-1], 
            self.downsamples,
            denoise_decoder_features[:-1]
        ):
            x = fuse(x, decoder_feedback_feature)
            x = denoise(x, 1)  # residual switch on
            encoder_features.append(x)
            x = down(x)
        x = self.feedback_fuses[-1](x, denoise_decoder_features[-1])
        x = self.denoising_blocks[-1](x, 1)  # residual switch on
        
        return x, res1, encoder_features

    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = (2 ** (self.layers - 1))

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor
        
    def _check_and_crop(self, x, res1):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top*self.block_size:bottom*self.block_size, left*self.block_size:right*self.block_size]
        res1 = res1[:, :, top:bottom, left:right] if res1 is not None else None
        return x, res1

    def forward(self, x):
        x = self._check_and_padding(x)
        x = self.act(self.feature_conv_0(x))
        x = self.feature_conv_1(x)
        f_short_cut = x

        ## encoder, local residual switch off
        encoder_features = []
        for denoise, down in zip(self.denoising_blocks[:-1], self.downsamples):
            x = denoise(x, 0)  # residual switch off
            encoder_features.append(x)
            x = down(x)
        x = self.denoising_blocks[-1](x, 0)  # residual switch off

        x, res1, refined_encoder_features = self._pass_features_to_color_decoder(x, f_short_cut, encoder_features) 

        ## color correction
        for color_correction, up, fuse, encoder_feature in reversed(list(zip(
            self.color_correction_blocks[1:], 
            self.upsamples, 
            self.color_decoder_fuses,
            refined_encoder_features
        ))):
            x = color_correction(x)
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.color_correction_blocks[0](x)

        x = self.act(self.conv_fuse_0(x))
        x = self.conv_fuse_1(x)
        x = self.pixel_shuffle(x)
        x, raw = self._check_and_crop(x, res1)

        return x, raw

'''
FLOPs: 65.616740352 G
Params: 2.834128 M
'''
def cal_model_complexity():
    import thop
    model = DNF(f_number=32,block_size=2,layers=4,denoising_block='CID',feedback_fuse='GFM').cuda()
    x = torch.rand(1,4,512,512).cuda()
    flops, params = thop.profile(model, inputs=(x,), verbose=False)  
    print(f"FLOPs: {flops / 1e9} G") 
    print(f"Params: {params / 1e6} M")



if __name__ == '__main__':
    cal_model_complexity()
