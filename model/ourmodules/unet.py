import math
from typing import NoReturn, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torchlibrosa.stft import ISTFT, STFT, magphase


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

# Added for after conv process
class ICB_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        activation: str,
        momentum: float,
    ):
        r"""Convolutional block."""
        super(ICB_Block, self).__init__()

        self.activation = activation
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.init_weights()

    def init_layer(self, layer: nn.Module) -> NoReturn:
        r"""Initialize a Linear or Convolutional layer."""
        nn.init.xavier_uniform_(layer.weight)

        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)

    def init_bn(self, bn: nn.Module) -> NoReturn:
        r"""Initialize a Batchnorm layer."""
        bn.bias.data.fill_(0.0)
        bn.weight.data.fill_(1.0)
        bn.running_mean.data.fill_(0.0)
        bn.running_var.data.fill_(1.0)

    def act(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        if activation == "relu":
            return F.relu_(x)

        elif activation == "leaky_relu":
            return F.leaky_relu_(x, negative_slope=0.01)

        elif activation == "swish":
            return x * torch.sigmoid(x)

        else:
            raise Exception("Incorrect activation!")

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        self.init_layer(self.conv1)
        self.init_layer(self.conv2)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.
        Args:
            input_tensor: (batch_size, in_feature_maps, time_steps, freq_bins)
        Returns:
            output_tensor: (batch_size, out_feature_maps, time_steps, freq_bins)
        """
        x = self.act(self.bn1(self.conv1(input_tensor)), self.activation)
        x = self.act(self.bn2(self.conv2(x)), self.activation)
        output_tensor = x

        return output_tensor


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()
        
        # UNet Added
        self.input_channels = 1
        self.target_sources_num = 1
        self.output_sources_num = 1

        window_size = 2048
        hop_size = window_size//4
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "relu"
        momentum = 0.01

        self.subbands_num = 1
        self.K = 4  # outputs: |M|, cos∠M, sin∠M
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # End

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel*self.K, groups=norm_groups)
    
    def spectrogram_phase(
        self, input: torch.Tensor, eps: float = 0.0
    ):
        r"""Calculate the magnitude, cos, and sin of the STFT of input.

        Args:
            input: (batch_size, segments_num)
            eps: float

        Returns:
            mag: (batch_size, time_steps, freq_bins)
            cos: (batch_size, time_steps, freq_bins)
            sin: (batch_size, time_steps, freq_bins)
        """
        (real, imag) = self.stft(input)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin
    
    def wav_to_spectrogram_phase(
        self, input: torch.Tensor, eps: float = 1e-10
    ):
        r"""Convert waveforms to magnitude, cos, and sin of STFT.

        Args:
            input: (batch_size, channels_num, segment_samples)
            eps: float

        Outputs:
            mag: (batch_size, channels_num, time_steps, freq_bins)
            cos: (batch_size, channels_num, time_steps, freq_bins)
            sin: (batch_size, channels_num, time_steps, freq_bins)
        """
        batch_size, channels_num, segment_samples = input.shape

        # Reshape input with shapes of (n, segments_num) to meet the
        # requirements of the stft function.
        x = input.reshape(batch_size * channels_num, segment_samples)

        mag, cos, sin = self.spectrogram_phase(x, eps=eps)
        # mag, cos, sin: (batch_size * channels_num, 1, time_steps, freq_bins)

        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(batch_size, channels_num, time_steps, freq_bins)
        cos = cos.reshape(batch_size, channels_num, time_steps, freq_bins)
        sin = sin.reshape(batch_size, channels_num, time_steps, freq_bins)

        return mag, cos, sin
    
    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.
        Args:
            input_tensor: (batch_size, target_sources_num * input_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            sin_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            cos_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
        Outputs:
            waveform: (batch_size, target_sources_num * input_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, input_channles, K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        linear_mag = x[:, :, :, 3, :, :]
        # mask_cos, mask_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Reformat shape to (n, 1, time_steps, freq_bins) for ISTFT.
        shape = (
            batch_size * self.target_sources_num * self.input_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * input_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.input_channels, audio_length
        )
        # (batch_size, target_sources_num * input_channels, segments_num)

        return waveform

    def forward(self, x, time):
        # x = (B, 2, 1, sample_num)
        x = x.squeeze(-2)
        audio_length = x.shape[-1]
        
        # Get Magnitude and phase individually
        mag_with_cond, _, _ = self.wav_to_spectrogram_phase(x)
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(x[:, 1, :].unsqueeze(-2))

        # Add Pad for even num of convolutions
        x = mag_with_cond
        origin_w = x.shape[-1]
        origin_h = x.shape[-2]
        pad_right = (int(np.ceil(origin_w / 32)) * 32 - origin_w)
        pad_top = (int(np.ceil(origin_h / 32)) * 32 - origin_h)
        x = F.pad(x, pad=(0, pad_right, pad_top, 0))
        
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        x = self.final_conv(x)
        
        # Recover shape
        x = F.pad(x, pad=(0, 0, 0, origin_h - x.shape[-2]))
        x = x[...,:origin_w]

        separated_audio = self.feature_maps_to_wav(x, mag, sin_in, cos_in, audio_length)
        # separated_audio: (batch_size, output_sources_num * input_channels, segments_num)
        
        separated_audio = separated_audio.unsqueeze(-2)
        return separated_audio
