from typing import Dict

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

from auto_tv_denoise import math_utils
from auto_tv_denoise.math_utils import preprocessing_for_tv_denoise


class Signal:
    def __init__(self, signal_value: "np.ndarray", time_step: "np.ndarray", name: str = ""):
        self._signal_value = signal_value
        self._time_step = time_step
        self._n_sample = signal_value.size
        self._signal_segment, self._time_segment = math_utils.u_egal(signal_value, time_step)
        self._name = name

    @property
    def n_sample(self):
        return self._n_sample

    def get_signal(self):
        return self._signal_value, self._time_step

    def get_signal_segment(self):
        return self._signal_segment, self._time_segment

    def mad(self):
        """[summary]
        noise Standard deviation estimation by MAD
        Returns:
            [float]: [noise Standard deviation]
        """
        return scipy.stats.median_absolute_deviation(np.diff(self._signal_value)) / np.sqrt(2)

    def plot(self, **kwargs):
        if "label" not in kwargs:
            kwargs["label"] = self._name
        plt.plot(self._time_step, self._signal_value, **kwargs)


class NoiseSignal(Signal):
    is_pre_processed: bool = False

    def __init__(self, signal_value: "np.ndarray", time_step: "np.ndarray", name: str = ""):
        super().__init__(signal_value, time_step, name)

    def pre_processing(self):
        (self._lam_rond, self._grand_lambda, self._g_lambda) = preprocessing_for_tv_denoise(
            self._signal_segment, self._time_segment
        )
        self.is_pre_processed = True

    def get_elements_for_tv_restoration(self) -> Dict:
        if self.is_pre_processed:
            return {"lam_rond": self._lam_rond, "grand_lambda": self._grand_lambda, "g_lambda": self._g_lambda}
        else:
            raise ValueError("Signal is not pre processed.")

    def plot(self, **kwargs):
        if "label" not in kwargs:
            kwargs["label"] = self._name
        if "s" not in kwargs:
            kwargs["s"] = 1
        plt.scatter(self._time_step, self._signal_value, **kwargs)
