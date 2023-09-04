import numpy as np
from scipy.special import hankel2
from ..sources import full_ricker_wavelet


def nodal_homogeneous_analytical(Wave_object, offset, c_value, n_extra=5000):

    # Generating extended ricker wavelet
    dt = Wave_object.dt
    final_time = Wave_object.final_time

    extended_final_time = n_extra * final_time

    frequency = Wave_object.frequency
    delay = Wave_object.delay
    amplitude = Wave_object.amplitude
    delay_type = Wave_object.delay_type

    ricker_wavelet = full_ricker_wavelet(
        dt=dt,
        final_time=extended_final_time,
        frequency=frequency,
        delay=delay,
        amplitude=amplitude,
        delay_type=delay_type,
    )

    return analytical_solution(ricker_wavelet, c_value, final_time, offset)


def analytical_solution(ricker_wavelet, c_value, final_time, offset):
    num_t = len(ricker_wavelet)

    # Constantes de Fourier
    nf = int(num_t/2 + 1)
    frequency_axis = (1.0/final_time) * np.arange(nf)

    # FOurier tranform of ricker wavelet
    fft_rw = np.fft.fft(ricker_wavelet)
    fft_rw = fft_rw[0:nf]

    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf - 1):
        k = 2 * np.pi * frequency_axis[a] / c_value
        tmp = k * offset
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * fft_rw[a]

    U_t = 1.0 / (2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], num_t))

    return np.real(U_t)
