import numpy as np


def _omega_coef(x: np.ndarray, N: int) -> np.ndarray:
    """Calculate omega coefficient for fourier transform

    Args:
        x (np.ndarray): signal
        N (int): n-th frequency

    Returns:
        np.ndarray: partial FT
    """
    return np.exp(-2j * np.pi * x / N)


def dft(x: np.ndarray) -> np.ndarray:
    """Basic implementation of Discrete Fourier Transform.
       Complexity: O(n^2)

    Args:
        x (np.ndarray): signal of shape (N,)
    Returns:
        np.ndarray: vector of Fourier coefficients of shape (N,)
    """
    N = x.shape[0]
    k = np.arange(N)
    D = _omega_coef(k * k.reshape(N, 1), N)
    return D @ x


def fft(x: np.ndarray) -> np.ndarray:
    """Simple implementation of Fast Fourier Transform via Cooley-Tuckey algorithm. Requires x to be power of 2.
       Complexity: O(nlog2(n))
        

    Args:
        x (np.ndarray): signal of shape (N,)

    Returns:
        np.ndarray: vector of Fourier coefficients of shape (N,)
    """
    assert np.ceil(np.log2(x.shape[0])) == np.log2(x.shape[0]), f'Fast fourier transform require len of signal to be the power of 2!'
    def _fft(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n < 32: # at this point fft is slower than regular dft
            return dft(x)
        else:
            x_even = fft(x[::2])
            x_odd = fft(x[1::2])
            D_half = _omega_coef(np.arange(n), n)
            return np.concatenate([
                x_even + D_half[:(n // 2)] * x_odd,
                x_even + D_half[(n // 2):] * x_odd
            ])
    return _fft(x)


def hann_window(k: int) -> np.ndarray:
    """Creates hann window on domain [0, k - 1]

    Args:
        k (int): len of frame

    Returns:
        np.ndarray: hann window function applied on domain
    """
    domain = np.arange(0, k, step=1)
    return 0.5 * (1 - np.cos((2 * np.pi * domain)/ (k - 1)))


def stft(x: np.ndarray, frame_size=2048, hop_size=512, window_function='hann') -> np.ndarray:
    """Implementation of Short-Time Fourier Transform. This version always uses a padding to a frame_size.

    Args:
        x (np.ndarray): signal of shape (N,)
        frame_size (int, optional): Size of frame. Must be power of 2. Defaults to 2048.
        hop_size (int, optional): Size of hop. Defaults to 512.
        window_function (str, optional): Function to convolve partial signal with. Defaults to 'hann'.

    Raises:
        ValueError: If such window function is not supported.

    Returns:
        np.ndarray: spectrogram of size (freq_bins, win_size) = (frame_size // 2 + 1, (n_samples - frame_size) // hop_size + 1)
    """

    # frame size must be power of 2
    assert np.ceil(np.log2(frame_size)) == np.log2(frame_size), 'Frame size must be power of 2!'

    n_samples = x.shape[0]
    # pad signal
    pad_size = n_samples % frame_size
    x = np.concatenate([x, np.zeros(pad_size)])

    # calculate parameters and create result matrix 
    freq_bins = frame_size // 2 + 1
    win_len = (n_samples - frame_size) // hop_size + 1
    spectrogram = np.zeros((freq_bins, win_len), dtype=np.complex64)

    # create window vector
    if window_function == 'hann':
        window_vector = hann_window(k=frame_size)
    else:
        raise ValueError(f'Window function "{window_function}" is not supported')
    
    # starting point and hann window const vector
    for m in range(win_len):
        signal_short = x[m * hop_size : m * hop_size + frame_size] * window_vector
        dft = fft(
            x=signal_short
        )[:freq_bins]
        spectrogram[:, m] = dft

    return spectrogram

