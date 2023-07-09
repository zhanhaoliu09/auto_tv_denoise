from typing import List

import matplotlib.pyplot as plt

from auto_tv_denoise.signal import Signal


def plot_signals(list_of_signal: List[Signal]):
    plt.figure()
    for s in list_of_signal:
        s.plot()
    plt.grid()
    plt.legend()
    plt.show()
