import numpy as np
from scipy.fft import fft, ifft

def to_latent_fourier_space(x, k=1):
    fx = fft(x)
    return np.real(fx[:k])

def from_latent_fourier_space(z, d=5000):
    fx_full = np.zeros(d, dtype=complex)
    fx_full[:len(z)] = z
    return np.real(ifft(fx_full))
