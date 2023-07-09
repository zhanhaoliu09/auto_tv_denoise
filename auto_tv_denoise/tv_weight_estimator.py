import abc

import numpy as np
import scipy.stats

from auto_tv_denoise import math_utils
from auto_tv_denoise.signal import NoiseSignal


class AbstractWeightEstimator:
    _selected_weight: float
    _name: str

    def __init__(self, name):
        self._name = name

    @property
    def selected_weight(self) -> float:
        return self._selected_weight

    @property
    def name(self) -> str:
        return self._name


class ManualWeightEstimator(AbstractWeightEstimator):
    def __init__(self, manual_weight: float, **kwargs):
        super().__init__(**kwargs)
        self._selected_weight = manual_weight


class AbstractOptimizedWeightEstimator(AbstractWeightEstimator):
    _selected_weight: float

    def __init__(self, s: NoiseSignal, **kwargs):
        super().__init__(**kwargs)
        self._selected_weight = self._estimate_weight(s)

    @abc.abstractmethod
    def _estimate_weight(self, s) -> float:
        """ """


class SureWeightEstimator(AbstractOptimizedWeightEstimator):
    def _estimate_weight(self, s, **kwargs) -> float:
        test_max = np.log10(np.max(s._lam_rond))
        list_test = np.logspace(test_max - 4, test_max, num=1000)
        sigma = s.mad()
        metric_to_minimize = np.inf
        l_op = 0
        signal_value, _ = s.get_signal()
        for l1 in list_test:
            u_lambda = math_utils.get_denoised_signal_with_given_weight(s, l1)
            n_segment = np.sum(np.diff(u_lambda) != 0) + 1
            metric_of_l1 = np.sum((u_lambda - signal_value) ** 2) + sigma**2 * (2 * n_segment - s.n_sample)
            if metric_to_minimize >= metric_of_l1:
                metric_to_minimize = metric_of_l1
                l_op = l1
        return l_op


class AUTWeightEstimator(AbstractOptimizedWeightEstimator):
    def _estimate_weight(self, s, **kwargs) -> float:
        sigma = s.mad()
        lam_n = sigma * np.sqrt(s.n_sample * np.log(np.log(s.n_sample)))
        u_lambda = math_utils.get_denoised_signal_with_given_weight(s, lam_n)
        n_segment = (
            np.sum(
                np.abs(np.diff(u_lambda))
                > scipy.stats.norm.ppf(1 - 0.05 / 2 / (s.n_sample - 1)) * sigma / np.sqrt(s.n_sample - 1) * np.sqrt(2)
            )
            + 1
        )
        nbar = max(s.n_sample / n_segment, 3)
        lam_aut = sigma * np.sqrt(nbar * np.log(np.log(nbar)))
        return lam_aut


class AutoWeightEstimator(AbstractOptimizedWeightEstimator):
    def _estimate_weight(self, s, **kwargs) -> float:
        return math_utils.get_lambda_ours_auto(s, **kwargs)


class SemiAutoWeightEstimator(AbstractOptimizedWeightEstimator):
    def _estimate_weight(self, s, **kwargs) -> float:
        return math_utils.get_lambda_ours_semi_auto(s, **kwargs)
