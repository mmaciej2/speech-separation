import numpy as np
import scipy.signal
import torch
from torch.nn import Module
from torch.nn.modules.conv import ConvTranspose1d

tconv_pad = ConvTranspose1d(1,1,1)

def generate_STFT_bases(window, shift):
    freqs = window // 2 + 1
    t = torch.tensor(range(window)).float().unsqueeze(0).expand((freqs, window))
    f = torch.tensor(range(freqs)).float().unsqueeze(1).expand((freqs, window))
    real = torch.cos(2 * np.pi * t * f / window)
    imag = -torch.sin(2 * np.pi * t[1:-1] * f[1:-1] / window)
    return torch.cat([real, imag], 0).unsqueeze(1)

def analysis_to_synthesis(basis):
    synth_basis = basis/basis.shape[2]
    synth_basis[1:basis.shape[0]//2] *= 2
    synth_basis[basis.shape[0]//2+1:] *= 2
    return synth_basis

def stft_mag(spec):
    comp_mag = torch.sqrt(spec[..., 1:spec.shape[-2]//2, :] ** 2 +
                          spec[..., spec.shape[-2]//2+1:, :] ** 2)
    return torch.cat([torch.abs(spec[..., :1, :]), comp_mag, torch.abs(spec[..., -1:, :])], dim=-2)

def apply_mask(spec, mask):
    if spec.shape[-2]//2+1 == mask.shape[-2]:
        mask = torch.cat([mask, mask[..., 1:-1, :]], dim=-2)
    return spec * mask

class ConvSTFT(Module):
    def __init__(self, window, shift):
        super(ConvSTFT, self).__init__()
        assert(not window%shift)

        self.window = window
        self.stride = shift

        stft_window = torch.tensor(np.sqrt(scipy.signal.hann(window, sym=False))).float().unsqueeze(0).expand((window, window)).unsqueeze(1)
        cola_factor = window//shift//2

        analysis_bases = generate_STFT_bases(window, shift)
        synth_bases = analysis_to_synthesis(analysis_bases)

        self.register_buffer('analysis_bases', analysis_bases * stft_window)
        self.register_buffer('synth_bases', synth_bases * stft_window /cola_factor)

    def encode(self, x):
        return torch.nn.functional.conv1d(x, self.analysis_bases, None, self.stride, self.window//2, 1, 1)

    def decode(self, x):
        output_padding = tconv_pad._output_padding(x, None, self.stride, self.window//2, self.window)
        return torch.nn.functional.conv_transpose1d(x, self.synth_bases, None, self.stride, self.window//2, output_padding, 1, 1)

