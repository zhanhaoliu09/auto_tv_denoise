from typing import Tuple

import numpy as np

from auto_tv_denoise import math_utils
from auto_tv_denoise.signal import NoiseSignal, Signal
from auto_tv_denoise.tv_weight_estimator import AbstractWeightEstimator


class TVDenoiser:
    def __init__(self, signal: NoiseSignal, weight_estimator: AbstractWeightEstimator):
        self._weight_estimator = weight_estimator
        self._signal = signal

    def denoise_signal(self) -> Tuple[float, Signal]:
        """Denoise given signal"""
        selected_weight = self._weight_estimator.selected_weight
        denoise_sigal = self._tv_denoise_with_given_weight(self._signal, selected_weight)
        return selected_weight, denoise_sigal

    @staticmethod
    def _select_elements_with_given_index(elements_for_tv_restoration, i, weight):
        # n = elements_for_tv_restoration["grand_n"][i - 1, i - 1] + 1
        t_test = elements_for_tv_restoration["grand_t"][i - 1, :i]
        beta_test = elements_for_tv_restoration["beta"][i - 1, :i]
        v_test = (
            elements_for_tv_restoration["v"][i - 1, :i]
            - (elements_for_tv_restoration["grand_lambda"][i] - weight) * beta_test
        )
        t_segment = np.array(np.cumsum(np.append(elements_for_tv_restoration["time_step"][0], t_test)), dtype=int)
        return v_test, t_segment

    def _tv_denoise_with_given_weight(self, s: NoiseSignal, weight: float) -> Signal:
        u_restored, t = math_utils.get_denoised_signal_with_given_weight(s, weight)
        return Signal(signal_value=u_restored, time_step=t, name=f"denoised by {self._weight_estimator.name}")
