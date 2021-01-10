import numpy as np
import matplotlib.pyplot as plt

from denoise_tv.denoise_tv import Denoise_tv_naive
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
    um = u0 + np.random.normal(0, 0.5 , n)

    t = np.arange(n)
    deno = Denoise_tv_naive(um, t)
    deno.denoise_auto()
    deno.get_lambda_sure()
    deno.get_lambda_aut()
    deno.show_denoised_signal(plot_aut = True, plot_sure = True, u0 = u0)
    plt.show()
if __name__ == "__main__":
    # execute only if run as a script
    main()
