import numpy as np
import matplotlib.pyplot as plt

import auto_tv_denoise
from auto_tv_denoise.signal import NoiseSignal
from auto_tv_denoise.signal_tv_denoiser import TVDenoiser
from auto_tv_denoise.tv_weight_estimator import ManualWeightEstimator, SureWeightEstimator, AUTWeightEstimator, \
    AutoWeightEstimator, SemiAutoWeightEstimator
from auto_tv_denoise.utils import plot_signals


def main():
    n = 200
    t = np.arange(0, n)
    u0 = np.zeros(t.size)

    u0[5:20] = 1
    u0[20:35] = 0
    u0[35:50] = -1
    #
    u0[50:100] = u0[:50]
    u0[100:150] = u0[:50]
    u0[150:200] = u0[:50]

    u0 = u0 * 2
    um = u0 + np.random.normal(0, 0.5, n)
    print(um)

    t = np.arange(n)
    s = NoiseSignal(um, t, name='Noised_signal')
    s.pre_processing()
    weight_estimator = ManualWeightEstimator(manual_weight=1, name='Manual')
    print(weight_estimator.selected_weight)
    denoiser = TVDenoiser(signal=s, weight_estimator=weight_estimator)
    _, signal_restored = denoiser.denoise_signal()

    # sure_weight_estimator = SureWeightEstimator(s, name='SURE')
    # print(sure_weight_estimator.selected_weight)
    # sure_denoiser = TVDenoiser(signal=s, weight_estimator=sure_weight_estimator)
    # sure_weight, sure_signal_restored = sure_denoiser.denoise_signal()
    #
    # aut_weight_estimator = AUTWeightEstimator(s, name='AUT')
    # print(aut_weight_estimator.selected_weight)
    # aut_denoiser = TVDenoiser(signal=s, weight_estimator=aut_weight_estimator)
    # aut_weight, aut_signal_restored = aut_denoiser.denoise_signal()

    auto_weight_estimator = AutoWeightEstimator(s, name='ours')
    print(auto_weight_estimator.selected_weight)
    auto_denoiser = TVDenoiser(signal=s, weight_estimator=auto_weight_estimator)
    auto_weight, auto_signal_restored = auto_denoiser.denoise_signal()

    semi_auto_weight_estimator = SemiAutoWeightEstimator(s, name='semi_auto')
    print(semi_auto_weight_estimator.selected_weight)
    semi_auto_weight, semi_auto_signal_restored = TVDenoiser(
        signal=s,
        weight_estimator=semi_auto_weight_estimator
    ).denoise_signal()

    plot_signals([s, signal_restored, auto_signal_restored, semi_auto_signal_restored])


if __name__ == "__main__":
    main()
