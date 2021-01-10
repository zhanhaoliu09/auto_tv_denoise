import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def t2tau(t):
    """[summary]
    Sample time to sample period
    Args:
        t ([array]): [sampling time]

    Returns:
        [array]: [sampling period]
    """
    tau = np.zeros(t.shape)
    tau[1:] = np.diff(t)
    tau[0] = tau[1]
    return tau

def findMinMAX(u):
    """[summary]
    Find local min et max in an array
    Args:
        u ([array]): [sample vector]

    Returns:
        [array]: [max = 1, min = -1]
    """
    mm = np.zeros(u.shape)
    nb_min = (np.r_[True, u[1:] <u[:-1]] & np.r_[u[:-1] < u[1:], True])
    nb_max = (np.r_[True,u[1:] > u[:-1]] & np.r_[u[:-1] > u[1:], True])
    mm[nb_min] = -1
    mm[nb_max] = 1
    return mm

def u_egal(u, t):
    """[summary]
    regroup initial 'segments' (u[i] = u[i+1]) in the noisy signal
    Args:
        u ([array]): [sample vector]
        t ([arry]): [sampling time vector]

    Returns:
        [array, array]: [sample and sampling period vector without initial segments]
    """
    bool_egal = u[:-1] != u[1:]
    bool_egal = np.append(bool_egal, True)
    u_output = u[bool_egal]
    t_output = t[bool_egal]
    t_output = np.append((t[0] - (t[1] - t[0])), t_output)
    return u_output, t_output

def t2tau2(t):
    """
    Apply to t calculated by u_egal
    samping time to sampling period
    :param t:
    :param tau0:
    :return:
    """
    return np.diff(t)

def get_sign(um):
    """[summary]
    get s(um), c.f. equation ??
    Args:
        um ([array]): [sample vector]
    Returns:
        [array]: [s(um)]
    """
    s1 = np.ones(len(um)+1)
    s1[1:-1] = um[1:] > um[:-1]
    s1[s1==0] = -1
    s1[0] = 0
    s1[-1] = 0
    return s1

def derivative_right(x, y, p=0.5):
    """[summary]
    right derivative approximation : dy/dx
    Args:
        x ([array])
        y ([array])
        p (float, optional): [scale for approximation]. Defaults to 0.5.

    Returns:
        [array]: [dy/dx]
    """
    n = x.size
    output = np.zeros(n)
    j = 0
    for i in np.arange(n - 1):
        while j < n - 1 and x[i] + p > x[j]:
            j += 1
        output[i] = (y[j] - y[i])
    return output

def derivative_left(x, y, p=0.5):
    """[summary]
    left derivative approximation : dy/dx
    Args:
        x ([array])
        y ([array])
        p (float, optional): [scale for approximation]. Defaults to 0.5.

    Returns:
        [array]: [dy/dx]
    """
    n = x.size
    output = np.zeros(n)
    j = n - 1
    for i in np.arange(n - 1, 0, -1):
        while j > 0 and x[i] - p < x[j]:
            j -= 1
        output[i] = (y[i] - y[j])
    return output

def get_g_lambda(lam_all, nb_minmax_DP, **kwargs):
    """[summary]
    show g(lambda)
    Args:
        lam_all ([array]): [list Lambda]
        nb_minmax_DP ([array]): [list g(lambda)]
    """
    l_plot = lam_all[::-1]
    l_plot_1 = np.append(l_plot[1:] - 1e-8, l_plot[-1] + 100)
    plt.figure(figsize=(10, 5))
    for a, b, y in zip(l_plot, l_plot_1, nb_minmax_DP):
        plt.plot([a, b], [y, y], color = 'b')
    # prepare for plotting given lambda
    if 'list_l' in kwargs:
        list_l = kwargs['list_l']
        n = len(list_l)
        bool_name = False
        if 'list_name' in kwargs and len(kwargs['list_name']) >= n:
            bool_name = True
            name = kwargs['list_name']
        else:
            name = [' ']*n
        if 'list_color' in kwargs and len(kwargs['list_color']) >= n:
            color = kwargs['list_color']
        else:
            color = ['r'] * n

        for l_select, s, c in zip(list_l, name, color):
            plt.axvline(l_select, linestyle='--', color = c, label=s)
        if bool_name:
            plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$g(\lambda)$')
    if 'output_path' in kwargs:
        title = ''
        if 'title' in kwargs:
            title = kwargs['title']
        plt.savefig(os.path.join(kwargs['output_path'], r'g_lambda_%s.png' % title))
    if 'bool_show' in kwargs and kwargs['bool_show']:
        plt.show()

def show_denoised_signal(um, t, **kwargs):
    """[summary]
    Show noisy and denoised signals
    Args:
        um ([array]): [sample vector]
        t ([array]): [sampling time vector]
        list_u ([list of array]) : optional, list of denoised signal
        list_name ([list of string]) : optional, list of label
        list_color ([list of string]) : optional, list of color
        list_style ([list of string]) : optional, list of line style
        output_path ([string]) : optional, path for saving figure.
        bool_show : optional, apply plt.show() at the end if True    
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(t, um, color='r', s=2)

    if 'list_u' in kwargs:
        list_u = kwargs['list_u']
        n = len(list_u)
        bool_name = False
        if 'list_name' in kwargs and len(kwargs['list_name']) >= n:
            bool_name = True
            name = kwargs['list_name']
        else:
            name = [' '] * n
        if 'list_color' in kwargs and len(kwargs['list_color']) >= n:
            color = kwargs['list_color']
        else:
            color = ['b'] * n
        if 'list_style' in kwargs and len(kwargs['list_style']) >= n:
            style = kwargs['list_style']
        else:
            style = ['--'] * n

        for u, s, c, st in zip(list_u, name, color, style):
            plt.plot(t, u, linestyle=st, color=c, label=s)
        if bool_name:
            plt.legend()

    plt.xlabel('t')
    if 'output_path' in kwargs:
        title = ''
        if 'title' in kwargs:
            title = kwargs['title']
        plt.savefig(os.path.join(kwargs['output_path'], 'signal_%s.png' % title))
    if 'bool_show' in kwargs and not(kwargs['bool_show']):
        pass
    else:
        plt.show()

def get_lambda_ours_auto(lam_all, nb_minmax_DP, **kwargs):
    """[summary]
    compute lambda with automatic method, c.f equation ??
    Args:
        lam_all ([array]): [Lambda]
        nb_minmax_DP ([array]): [g(lambda)]
        lambda_min ([float]): optional, Infimum for lambda estimation. Default = 0.
        log ([float]): optional, parameter log_10(q). Default : automatic choice
        bool_plot ([boolean]) : plot denoise signal if True, Default : False
        output_path ([string]) : output path for saving figure if bool_plot = True
        title ([string]) : extention for figure name  
    Returns:
        [float]: our proposition of lambda
    """
    # get \lambda^g 
    selection = np.append(True, nb_minmax_DP[1:] != nb_minmax_DP[:-1])
    l_plot = lam_all[::-1]
    x, y = l_plot[selection], nb_minmax_DP[selection]
    if 'lambda_min' in kwargs:
        ind_x1 = x > kwargs['lambda_min']
        y1, x1 = y[ind_x1], np.log10(x[ind_x1])
    else:
        y1, x1 = y[1:], np.log10(x[1:])

    if 'log' in kwargs:
        log = kwargs['log']
    else:
        log = max(np.diff(x1[int(x1.size/10):]))
    # lambda estimation
    d_right = derivative_right(x1, y1, p=log)
    d_left = -derivative_left(x1, y1, p=log)
    d2 = d_left + d_right
    d3 = np.append(np.diff(d2[np.argmax(d2) - 1:]), 0)
    d4 = np.diff(d3)
    if d4.size == 0:
        l_select = 10 ** x1[np.argmax(d3)]
    else:
        l_select = 10 ** x1[np.argmin(d4) + np.argmax(d2)-1]
    # plot
    if 'bool_plot' in kwargs and kwargs['bool_plot']:
        bool_output = False
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
            bool_output = True
        if 'title' in kwargs:
            title = kwargs['title']
        plt.figure(figsize=(9, 4.5))
        plt.scatter(10 ** x1, d2, s=5)
        plt.axvline(l_select, linestyle='--', color='black', label=r'$\lambda_{ours}$')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\partial^2 g(\lambda)$')
        plt.legend()
        plt.xscale('log')
        plt.xlim([10**x1[0], 10**x1[-1]])
        if bool_output:
            plt.savefig(os.path.join(output_path, 'd2g_%s.png'% title))

    return l_select

def get_lambda_ours_semi_auto(lam_all, nb_minmax_DP, **kwargs):
    """[summary]
    compute lambda with semi-automatic method
    Args:
        lam_all ([array]): [Lambda]
        nb_minmax_DP ([array]): [g(lambda)]
        lambda_min ([float]): optional, Infimum for lambda estimation. Default = 0.
        log ([float]): parameter log_10(q). Default : automatic choice
        p ([float]) : parameter p. Default: 0.02
        bool_plot ([boolean]) : plot denoise signal if True, Default : False
        output_path ([string]) : output path for saving figure if bool_plot = True
        title ([string]) : extension for figure name  
    Returns:
        [float]: our proposition of lambda
    """
    selection = np.append(True, nb_minmax_DP[1:] != nb_minmax_DP[:-1])
    l_plot = lam_all[::-1]
    x, y = l_plot[selection], nb_minmax_DP[selection]
    if 'lambda_min' in kwargs:
        ind_x1 = x > kwargs['lambda_min']
        y1, x1 = y[ind_x1], np.log10(x[ind_x1])
    else:
        y1, x1 = y[1:], np.log10(x[1:])

    if 'log' in kwargs:
        log = kwargs['log']
    else:
        log = max(np.diff(x1[int(x1.size/10):]))

    if 'p' in kwargs:
        decalage = kwargs['p']
    else:
        decalage = 0.02
    # print('log = %.1f decalage = %.3f' %(log, decalage))

    n = lam_all.size
    d_right = derivative_right(x1, y1, p=log)
    d_left = -derivative_left(x1, y1, p=log)
    d2 = d_left + d_right
    idx = np.where(d2 >= (np.max(d2) - decalage*log*n))[0]
    l_select = 10 ** x1[idx[-1]]
    if 'bool_plot' in kwargs and kwargs['bool_plot']:
        bool_output = False
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
            bool_output = True
        if 'title' in kwargs:
            title = kwargs['title']
        plt.figure(figsize=(9, 4.5))
        plt.scatter(10 ** x1, d2, s=5)
        plt.axvline(l_select, linestyle='--', color='black', label=r'$\lambda_{ours}$')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\partial^2 g(\lambda)$')
        plt.legend()
        plt.xscale('log')
        plt.xlim([10**x1[0], 10**x1[-1]])
        if bool_output:
            plt.savefig(os.path.join(output_path, 'd2g_%s.png'% title))

    return l_select

def get_lambda_naive(um, tau):
    """[summary]
    Algorithm DP-VT
    Args:
        um ([array]): [noised signal]
        tau ([array]): [sampling periode]
    Returns:
        v ([matrix]): segment level matrix (v[i,:i] : segment level for lambda = lambda_{i+1})
        beta ([matrix]): beta matrix (beta[i,:i] = d(v[i,:i]) / d(lambda))
        lam ([array]): Lambda (lambda list of theorem 1)
        grandT ([matrix]): length of segment (grandT[i,:i] : segment length for lambda = lambda_{i+1})
        grandN ([matrix]): number of points in a segment (grandN[i,:i] : points in each segment for lambda = lambda_{i+1})
        grandMM ([matrix]): min max segment indicator (grandMM[i,:i] : indicator of each segment for lambda = lambda_{i+1})
    """
    n = um.shape[0]
    v = np.zeros((n, n))
    lam = np.zeros((n))
    grandT = np.zeros((n, n))
    grandN = np.zeros((n, n), dtype=int)
    grandMM = np.zeros((n, n), dtype=int)
    beta = np.zeros((n, n))

    grandT[-1, :] = tau

    grandN[-1, :] = np.arange(0, n)

    grandMM[-1, :] = findMinMAX(um)

    v[-1, :] = um
    for l in np.arange(n, 1, -1):
        s_n = get_sign(v[l - 1, :l])
        beta[l - 1, :l] = (s_n[1:l + 1] - s_n[:l]) / 2 / grandT[l - 1, :l]
        # gamma = beta[:-1] - beta[1:]
        gamma = beta[l - 1, :l - 1] - beta[l - 1, 1:l]
        delta_v = v[l - 1, :l - 1] - v[l - 1, 1:l]
        a = np.abs(delta_v / gamma)
        k = np.argmin(a) # find merge 
        # merge two segments
        lam[l - 2] = lam[l - 1] + a[k]
        v[l - 2, : k] = v[l - 1, :k] + a[k] * beta[l - 1, :k]
        # print(v[l, k] + a[k]*beta[k] - v[l, k+1] - a[k]*beta[k+1])
        v[l - 2, k] = v[l - 1, k] + a[k] * beta[l - 1, k]
        v[l - 2, k + 1:l - 1] = v[l - 1, k + 2:l] + a[k] * beta[l - 1, k + 2:l]

        grandT[l - 2, :k] = grandT[l - 1, : k]
        grandT[l - 2, k] = grandT[l - 1, k] + grandT[l - 1, k + 1]
        grandT[l - 2, k + 1:l - 1] = grandT[l - 1, k + 2:l]

        grandN[l - 2, :k] = grandN[l - 1, : k]
        grandN[l - 2, k] = grandN[l - 1, k + 1]
        grandN[l - 2, k + 1:l - 1] = grandN[l - 1, k + 2:l]

        grandMM[l - 2, :k] = grandMM[l - 1, : k]
        grandMM[l - 2, k] = grandMM[l - 1, k] + grandMM[l - 1, k + 1]
        grandMM[l - 2, k + 1:l - 1] = grandMM[l - 1, k + 2:l]
        # update for the first and last segment (must be an extremum)
        if grandMM[l - 2, 0] == 0:
            grandMM[l - 2, 0] = grandMM[l - 1, 1]
        if grandMM[l - 2, l - 2] == 0:
            grandMM[l - 2, l - 2] = grandMM[l - 1, l - 2]

    return v, beta, lam, grandT, grandN, grandMM

def get_u_naive(v, grandT, grandN, beta, lam, t, l_test, t0 = 0):
    """[summary]
    Get solution u^* for a given lambda
    Args:
        v ([matrix]): [output of get_lambda_dp_vt]
        grandT ([matrix]): [output of get_lambda_dp_vt]
        grandN ([matrix]): [output of get_lambda_dp_vt]
        beta ([matrix]): [output of get_lambda_dp_vt]
        lam ([matrix]): [output of get_lambda_dp_vt]
        t ([type]): [Sample tile for output]
        l_test ([type]): [the given lambda]
        t0 (int, optional): [the first sample time for training]. Defaults to 0.
    Returns:
        [array]: [solution]
    """
    # get u with known lambda
    for i, l in enumerate(lam):
        if l <= l_test:
            # print(i)
            break
    i = i + 1
    n = grandN[i - 1, i - 1] + 1
    t_test = grandT[i - 1, :i]
    beta_test = beta[i - 1, :i]
    v_test = v[i - 1, :i] - (l - l_test) * beta_test
    t_plot = np.array(np.cumsum(np.append(t0, t_test)), dtype=int)
    u_nous = np.zeros(n)
    flag = 0
    for i, t1 in enumerate(t):
        # while (t_plot[flag] <= t1):
        while (t_plot[flag] <= t1):
            flag += 1
            if flag >=t_plot.size :
                break
        u_nous[i] = v_test[flag - 1]
    return u_nous