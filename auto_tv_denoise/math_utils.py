import numpy as np


def preprocessing_for_tv_denoise(um, tau):
    """
    preprocessing method stable
    Args:
        um ([type]): [noisy signal]
        tau ([type]): [sample period vector]

    Returns:
        [array, array]: [lam^circ, delta g^circ]
    """
    n = um.shape[0]
    # init
    v = np.array(um, dtype=float)
    lam_rond = np.zeros(n)
    grandMM_rond = np.zeros(n)

    grandT = tau.copy()
    grandN = np.arange(0, n)
    beta = np.zeros(n)

    grandMM = findMinMAX(um)

    lam_old = 0
    l = n  # current segment total number

    while l > 1:
        # find index list of segment merged
        s_n = get_sign(v[:l])
        beta[:l] = (s_n[1 : l + 1] - s_n[:l]) / 2 / grandT[:l]
        gamma = beta[: l - 1] - beta[1:l]
        delta_v = v[: l - 1] - v[1:l]
        a = np.abs(delta_v / gamma)

        # index list, size(k) > 1 --> multiple merge
        k = np.where(egal_float(a, a.min()))[0]
        k1 = k[0]

        l_new = l - k.size  # segment number after merge

        # get new lambda
        lam_old += a[k1]
        lam_rond[grandN[k[:]]] = lam_old

        # save new segment value
        # v_rond[grandN[k[:]]] = v[k[:]] + a[k1] * beta[k[:]]

        # MAJ segments
        k = np.append(-2, k)
        i = 1  # current segments (index for k list)
        while i < k.size:
            # find merge of multiple segments into one segment
            i_end = i
            for j in np.arange(i + 1, k.size, 1):
                if k[j] != k[i_end] + 1:
                    break
                else:
                    i_end = j

            # update segment values
            v[k[i - 1] + 3 - i : k[i] + 1 - i] = v[k[i - 1] + 2 : k[i]] + a[k1] * beta[k[i - 1] + 2 : k[i]]
            v[k[i] + 1 - i] = v[k[i]] + a[k1] * beta[k[i]]
            # save segment period
            # grandT_rond[grandN[k[i: i_end + 1]]] = np.sum(grandT[k[i]: k[i_end] + 2])
            # update segment period
            grandT[k[i - 1] + 3 - i : k[i] + 1 - i] = grandT[k[i - 1] + 2 : k[i]]
            grandT[k[i] - i + 1] = np.sum(grandT[k[i] : k[i_end] + 2])
            # save and update # min + # max
            (new_MM, new_nb) = fusion_minmax_multiple(k[i], k[i_end], grandMM[:l])
            # save only into the first segment for the multiple merge to one segment
            grandMM_rond[grandN[k[i]]] = new_nb
            grandMM[k[i - 1] + 3 - i : k[i] + 1 - i] = grandMM[k[i - 1] + 2 : k[i]]
            grandMM[k[i] - i + 1] = new_MM

            # save segment lenth (for indexing)
            grandN[k[i - 1] + 3 - i : k[i] + 1 - i] = grandN[k[i - 1] + 2 : k[i]]
            grandN[k[i] - i + 1] = grandN[k[i_end] + 1]

            # go to the next merge
            i = i_end + 1
        # MAJ for last segments (after the last merge)
        i = i - 1

        v[k[-1] + 2 - i : l_new] = v[k[-1] + 2 : l] + a[k1] * beta[k[-1] + 2 : l]
        grandT[k[-1] + 2 - i : l_new] = grandT[k[-1] + 2 : l]

        grandN[k[-1] + 2 - i : l_new] = grandN[k[-1] + 2 : l]
        grandMM[k[-1] + 2 - i : l_new] = grandMM[k[-1] + 2 : l]

        # MAJ # segment
        l = l_new
        laminds = lam_rond.argsort()
        grand_lambda = lam_rond[laminds][::-1]
        g_lambda = -np.cumsum(np.append(0, grandMM_rond[laminds[::-1]][:-1]))[::-1] + 1

    return lam_rond, grand_lambda, g_lambda


def fusion_minmax_multiple(k, k_end, grandMM):
    """
    get sign of new segment and the nombre of min-max segments vanished
    :param k: index begin (of beta)
    :param k_end: index end (of beta)
    :param grandMM: min max segemnt vector before merge
    :return:
    """
    mm_abs = np.sum(np.abs(grandMM[k : k_end + 2]))
    mm_return = np.sum(grandMM[k : k_end + 2])
    n = k_end - k + 2  # size of merged segments
    # left side
    if k == 0:
        if mm_return == 0:
            mm_return = -grandMM[0]
            nb_return = 1 - mm_abs
        else:
            nb_return = 1 - mm_abs
    else:
        # right side
        if k == grandMM.size - n:
            if mm_return == 0:
                mm_return = -grandMM[-1]
                nb_return = 1 - mm_abs
            else:
                nb_return = 1 - mm_abs
        else:
            # middle
            if mm_return == 0:
                nb_return = 0 - mm_abs
            else:
                nb_return = 1 - mm_abs
    return mm_return, nb_return


def egal_float(array, v_test, precision=1e-8):
    """
    test egal between two array of float
    :param array: array1
    :param v_test: array2
    :param precision:
    :return:
    """
    return np.abs(array - v_test) < precision


def get_sign(um):
    """[summary]
    get s(um), c.f. equation ??
    Args:
        um ([array]): [sample vector]

    Returns:
        [array]: [s(um)]
    """
    s1 = np.ones(len(um) + 1)
    s1[1:-1] = um[1:] > um[:-1]
    s1[s1 == 0] = -1
    s1[0] = 0
    s1[-1] = 0
    return s1


def findMinMAX(u):
    """[summary]
    Find local min et max in an array
    Args:
        u ([array]): [sample vector]

    Returns:
        [array]: [max = 1, min = -1]
    """
    mm = np.zeros(u.shape)
    nb_min = np.r_[True, u[1:] < u[:-1]] & np.r_[u[:-1] < u[1:], True]
    nb_max = np.r_[True, u[1:] > u[:-1]] & np.r_[u[:-1] > u[1:], True]
    mm[nb_min] = -1
    mm[nb_max] = 1
    return mm


def get_denoised_signal_with_given_weight(s: "NoiseSignal", weight: float):
    """
    get solution of VT for a given lambda
    :param lam_rond:
    :param um: orignal signal
    :param t: orignal sampling time (with constant, for get u)
    :param tau: sampling period without constant piece
    :param l_test: lambda
    :return:
    [array] : [restoration of each sample]
    """
    # init
    param = s.get_elements_for_tv_restoration()
    lam_rond = param["lam_rond"]
    um, tau = s.get_signal_segment()

    n = lam_rond.size  # size of tau (without constant piece)
    v = np.zeros((n))
    T = np.zeros((n))
    lam = np.zeros((n))
    tau_um = um * tau

    # get segment value
    # n_et = [-1]
    l = 0  # current point
    j = 0  # current segment
    while l < n:
        # find max lambda et lambda to break
        k = l
        for i in range(l, n, 1):
            if lam_rond[i] > weight:
                break
            else:
                if lam_rond[i] > lam_rond[k]:
                    k = i

        # get current segment value
        T[j] = np.sum(tau[l : i + 1])
        v[j] = 1 / T[j] * np.sum(tau_um[l : i + 1])
        l = i + 1
        j += 1

    # MAj v with given lambda
    s_n = get_sign(v[:j])
    beta = (s_n[1 : j + 1] - s_n[:j]) / 2 / T[:j]
    v[:j] += (weight - lam[:j]) * beta

    # v2u for the given vector t
    _, t = s.get_signal()
    t_plot = np.cumsum(T[:j]) + t[0] - (t[1] - t[0])
    u_nous = np.zeros(t.size)

    flag = 0
    for i, t1 in enumerate(t):
        while t_plot[flag] < t1:
            flag += 1
        u_nous[i] = v[flag]

    return u_nous, t


def get_lambda_ours_auto(s: "NoiseSignal", **kwargs):
    """ """
    # get \lambda^g
    param = s.get_elements_for_tv_restoration()
    grand_lambda = param["grand_lambda"]
    g_lambda = param["g_lambda"]
    selection = np.append(True, g_lambda[1:] != g_lambda[:-1])
    l_plot = grand_lambda[::-1]
    x, y = l_plot[selection], g_lambda[selection]
    if "lambda_min" in kwargs:
        ind_x1 = x > kwargs["lambda_min"]
        y1, x1 = y[ind_x1], np.log10(x[ind_x1])
    else:
        y1, x1 = y[1:], np.log10(x[1:])

    if "log" in kwargs:
        log = kwargs["log"]
    else:
        log = max(np.diff(x1[int(x1.size / 10) :]))
    # lambda estimation
    d_right = derivative_right(x1, y1, p=log)
    d_left = -derivative_left(x1, y1, p=log)
    d2 = d_left + d_right
    d3 = np.append(np.diff(d2[np.argmax(d2) - 1 :]), 0)
    d4 = np.diff(d3)
    if d4.size == 0:
        l_select = 10 ** x1[np.argmax(d3)]
    else:
        l_select = 10 ** x1[np.argmin(d4) + np.argmax(d2) - 1]

    return l_select


def get_lambda_ours_semi_auto(s: "NoiseSignal", **kwargs):
    """ """
    param = s.get_elements_for_tv_restoration()
    grand_lambda = param["grand_lambda"]
    g_lambda = param["g_lambda"]
    selection = np.append(True, g_lambda[1:] != g_lambda[:-1])
    l_plot = grand_lambda[::-1]
    x, y = l_plot[selection], g_lambda[selection]
    if "lambda_min" in kwargs:
        ind_x1 = x > kwargs["lambda_min"]
        y1, x1 = y[ind_x1], np.log10(x[ind_x1])
    else:
        y1, x1 = y[1:], np.log10(x[1:])

    if "log" in kwargs:
        log = kwargs["log"]
    else:
        log = max(np.diff(x1[int(x1.size / 10) :]))

    if "p" in kwargs:
        decalage = kwargs["p"]
    else:
        decalage = 0.02
    # print('log = %.1f decalage = %.3f' %(log, decalage))

    n = grand_lambda.size
    d_right = derivative_right(x1, y1, p=log)
    d_left = -derivative_left(x1, y1, p=log)
    d2 = d_left + d_right
    idx = np.where(d2 >= (np.max(d2) - decalage * log * n))[0]
    l_select = 10 ** x1[idx[-1]]
    return l_select


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
        output[i] = y[j] - y[i]
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
        output[i] = y[i] - y[j]
    return output


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
    return u_output, np.diff(t_output)


# def t2tau2(t):
#     """
#     Apply to t calculated by u_egal
#     samping time to sampling period
#     :param t:
#     :param tau0:
#     :return:
#     """
#     return np.diff(t)
