import numpy as np
import os
import scipy.stats
import matplotlib.pyplot as plt
import warnings
import denoise_tv.utils as utils


# Abstract class
class Denoise_tv:
    _um = np.array([])
    _t = np.array([])
    _n_sample = 0

    _um_diff = np.array([])
    _tau = np.array([])

    _grandLambda = np.array([])
    _g_lambda = np.array([])

    _u_ours = np.array([])
    _l_select = 0

    _u_op = np.array([])
    _l_op = 0

    _u_sure = np.array([])
    _l_sure = 0

    _u_aut = np.array([])
    _l_aut = 0

    _method = 0
    _method_preprossing = 0

    _log, _p = 0, 0.02

    _is_updated = False

    _dict_preprocessing = {
    4 : 'naive'
    }

    _dict_method = {
    1 : 'auto method with q %s',
    2 : 'semi-auto method with q %s and p %s',
    3 : 'manuel with lambda = %.2f'
    }
        
    def __init__(self, um, t, **kwargs):
        self._um = um
        self._t = t
        self._n_sample = um.size
        self._um_diff, t_output = utils.u_egal(um, t)
        self._tau = utils.t2tau2(t_output)
    
    def _preprocessing(self):
        pass
    
    def get_noisy_signal(self):
        return(self._um, self._t)
    
    def get_restoration(self):
        return(self._u_ours, self._t)

    def get_l_select(self):
        return(self._l_select)

    def get_l_optimal(self):
        return(self._l_op)

    def get_optimal(self):
        return(self._u_op, self._t)

    def _get_method(self):
        """[summary]
        get method applied for lambda_ours estimation
        Returns:
            [string]: [method (auto, semi auto or manuel)]
        """
        m = self._dict_method.get(self._method)
        if m:
            if self._log == 0:
                q = 'auto'
            else:
                q = '= %d' %(10**self._log)
            p = '= %.2f' % self._p

            if self._method == 1:
                method = m % q
            elif self._method == 2:
                method = m % (q, p)
            else:
                method = m % (self._l_select)
        else:
            method = 'not defined'
        return method

    # Call to operator on object
    def __call__(self):
        m = self._dict_preprocessing.get(self._method_preprossing)
        if m:
            print('Computation method : %s .' % m)
        else:
            print('Computation method not defined !')
        
    # # Info
    def __repr__(self):
        return('lambda = %.2f estimated by %s' % (self._l_select, self._get_method()))

    def _get_u_dp_vt(self, l_select):
        """[summary]
        get u^*(l_select)
        Args:
            l_select ([type]): [description]
        """
        pass

    def _get_g_lambda(self):
       pass

    def denoise_auto(self, **kwargs):
        """[summary]
        Denoising with automatic method
        Args:
            lambda_min ([float]): optional, Infimum for lambda estimation. Default = 0.
            log ([float]): optional, parameter log_10(q). Default : automatic choice
            bool_plot ([boolean]) : plot denoise signal if True, Default : False
            output_path ([string]) : output path for saving figure if bool_plot = True
            title ([string]) : extention for figure name  
        """
        self._l_select = utils.get_lambda_ours_auto(self._grandLambda, self._g_lambda, **kwargs)
        self._u_ours = self._get_u_dp_vt(self._l_select)
        self._method = 1
        if 'log' in kwargs:
            self._log = kwargs['log']
        else:
            self._log = 0
        self._is_updated = True

    def denoise_semi_auto(self, **kwargs):
        """[summary]
        Denoising with semi-automatic method
        Args:
            lambda_min ([float]): optional, Infimum for lambda estimation. Default = 0.
            log ([float]): parameter log_10(q). Default : automatic choice
            p ([float]) : parameter p. Default: 0.02
            bool_plot ([boolean]) : plot denoise signal if True, Default : False
            output_path ([string]) : output path for saving figure if bool_plot = True
            title ([string]) : extension for figure name  
        """
        self._l_select = utils.get_lambda_ours_semi_auto(self._grandLambda, self._g_lambda, **kwargs)
        self._u_ours = self._get_u_dp_vt(self._l_select)
        self._method = 2
        if 'p' in kwargs :
            self._p = kwargs['p']
        else:
            self._p = 0.02
        
        if 'log' in kwargs:
            self._log = kwargs['log']
        else:
            self._log = 0
        self._is_updated = True

    def denoise_manuel(self, lam):
        """[summary]
        Denoise with a given lambda
        Args:
            lam : lambda
        """
        self._l_select = lam
        self._u_ours = self._get_u_dp_vt(lam)
        self._method = 3
        self._is_updated = True

    def show_denoised_signal(self, **kwargs):
        """
        Show noisy and denoised signal
        Args:
            u0 ([array]): optional, original signal vector (u_net)
            output_path ([string]): optional, output_path
            title ([string]): optional, figure name
            plot_op ([boolean]) : optional, plot u(lambda_op) if True
            plot_sure ([boolean]) : optional, plot u(lambda_sure) if True
            plot_aut ([boolean]) : optional, plot u(lambda_aut) if True
            plot_show ([boolean]) : optional, apply plt.show() at the end if True
        """
        bool_output = False
        if 'output_path' in kwargs:
            output_path = kwargs['output_path']
            bool_output = True
        if 'title' in kwargs:
            title = kwargs['title']

        plt.figure(figsize=(10, 5))
        plt.scatter(self._t, self._um, color='r', s=2, label = 'y')
        if self._is_updated:
            plt.plot(self._t, self._u_ours, label=r'$u^*(\lambda_{ours})$', color = 'b')

            if 'u0' in kwargs:
                u0 = kwargs['u0']
                if u0.size == self._n_sample :
                    plt.plot(self._t, u0, '--', label=r'$u_{net}$')
                    if 'plot_op' in kwargs and kwargs['plot_op']:
                        if self._l_op ==0 :
                            self.compute_lambda_op(u0)
                        plt.plot(self._t, self._u_op, ':', label=r'$u_{op}$')

                else :
                    warnings.warn("Warning : u0 should  have the same size as um.")
        else:
            warnings.warn("Warning : lambda not estimated.")
        if 'plot_sure' in kwargs and kwargs['plot_sure']:
            if self._u_sure.size != self._n_sample :
                self.get_lambda_sure()
            plt.plot(self._t, self._u_sure, '-.', label=r'SURE', color = 'black')
        if 'plot_aut' in kwargs and kwargs['plot_aut']:
            if self._u_aut.size != self._n_sample :
                self.get_lambda_aut()
            plt.plot(self._t, self._u_aut, ':', label=r'AUT', color = 'green')
        plt.legend()
        plt.xlabel('t')
        if bool_output:
            plt.savefig(os.path.join(output_path, 'u_%s.png'% title))
        if 'bool_show' in kwargs and kwargs['bool_show']:
            plt.show()

    def show_g_lambda(self, **kwargs):
        """
        Show g(lambda)
        Args:
            u0 ([array]): optional, original signal vector (u_net)
            output_path ([string]): optional, output_path
            title ([string]): optional, figure name
            plot_op ([boolean]) : optional, plot u(lambda_op) if True
            plot_sure ([boolean]) : optional, plot u(lambda_sure) if True
            plot_aut ([boolean]) : optional, plot u(lambda_aut) if True
            plot_show ([boolean]) : optional, apply plt.show() at the end if True
        """
        if 'list_l' not in kwargs:
            kwargs['list_l'] = []
            kwargs['list_name'] = []
            kwargs['list_color'] = []
        if 'plot_ours' in kwargs and kwargs['plot_ours']:
            kwargs['list_l'].append(self._l_select)
            kwargs['list_name'].append(r'$\lambda_{ours}$')
            kwargs['list_color'].append('b')
        if 'plot_op' in kwargs and kwargs['plot_op']:
            kwargs['list_l'].append(self._l_op)
            kwargs['list_name'].append(r'$\lambda_{op}$')
            kwargs['list_color'].append('orange')
        if 'plot_sure' in kwargs and kwargs['plot_sure']:
            kwargs['list_l'].append(self._l_sure)
            kwargs['list_name'].append(r'$\lambda_{sure}$')
            kwargs['list_color'].append('black')
        if 'plot_aut' in kwargs and kwargs['plot_aut']:
            kwargs['list_l'].append(self._l_aut)
            kwargs['list_name'].append(r'$\lambda_{aut}$')
            kwargs['list_color'].append('g')

        utils.get_g_lambda(self._grandLambda, self._g_lambda, **kwargs)

    def compute_lambda_op(self, u0, **kwargs):
        """[summary]
        Computation lambda op, equation ??
        Args:
            u0 ([array]): original signal vector (u_net)
            list_test ([array]) : optional, list of lambda candidate
        Returns:
            [array]: [u(lambda_op)]
            [float]: [lambda_op]
        """
        if 'list_test' in kwargs:
            list_test = kwargs['list_test']
        else:
            test_max =  np.log10(self._grandLambda[0])
            list_test = np.logspace(test_max-4, test_max, num=1000)
        e_min = np.inf
        l_op = 0
        u_op = self._um.copy()
        for l1 in list_test:
            u_lambda =  self._get_u_dp_vt(l1)
            e_net = np.mean((u_lambda - u0) ** 2)
            if e_min >= e_net:
                e_min = e_net
                l_op = l1
                u_op = u_lambda
        self._u_op, self._l_op = u_op, l_op
        return u_op, l_op

    def mad(self):
        """[summary]
        noise Standard deviation estimation by MAD
        Returns:
            [float]: [noise Standard deviation]
        """
        return scipy.stats.median_absolute_deviation(np.diff(self._um)) / np.sqrt(2)

    def get_lambda_sure(self, **kwargs):
        """[summary]
        Computation lambda sure, equation ??
        Args:
            list_test ([array]) : optional, list of lambda candidate
            sigma ([float]) : noise Standard deviation. Default: estimated by MAD
        Returns:
            [array]: [u(lambda_sure)]
            [float]: [lambda_sure]
        """
        if 'list_test' in kwargs:
            list_test = kwargs['list_test']
        else:
            test_max = np.log10(self._grandLambda[0])
            list_test = np.logspace(test_max-4, test_max, num=1000)
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = self.mad()
        e_min = np.inf
        l_op = 0
        u_op = self._um.copy()
        for l1 in list_test:
            u_lambda =  self._get_u_dp_vt(l1)
            n_segment = np.sum(np.diff(u_lambda) !=0) + 1 
            e_net = np.sum((u_lambda - self._um) ** 2) + sigma**2 *(2*n_segment - self._n_sample)
            if e_min >= e_net:
                e_min = e_net
                l_op = l1
                u_op = u_lambda
        self._u_sure, self._l_sure = u_op, l_op
        return u_op, l_op

    def get_lambda_aut(self, **kwargs):
        """[summary]
        Computation lambda aut, equation ??
        Args:
            sigma ([float]) : noise Standard deviation. Default: estimated by MAD
        Returns:
            [array]: [u(lambda_aut)]
            [float]: [lambda_aut]
        """
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
        else:
            sigma = (self.mad())
        lam_n = sigma*np.sqrt(self._n_sample*np.log(np.log(self._n_sample)))
        u_lambda =  self._get_u_dp_vt(lam_n)
        n_segment = np.sum(np.diff(u_lambda) !=0) + 1
        n_segment = np.sum(np.abs(np.diff(u_lambda))>scipy.stats.norm.ppf(1-0.05/2/(self._n_sample-1))*sigma/np.sqrt(self._n_sample-1)*np.sqrt(2)) + 1
        # sum(abs(diff(muhatN))>qnorm(1-0.05/2/(N-1))*sigma/sqrt(N-1)*sqrt(2)) + 1;
        nbar=max(self._n_sample/n_segment,3)
        lam_aut = sigma*np.sqrt(nbar*np.log(np.log(nbar)))
        u_aut =  self._get_u_dp_vt(lam_aut)
        self._u_aut, self._l_aut = u_aut, lam_aut
        return u_aut, lam_aut


class Denoise_tv_naive(Denoise_tv):

    # Constructor
    def __init__(self, um, t, **kwargs):
        super().__init__(um, t, **kwargs)
        self._method_preprossing = 4
        self._preprocessing()
        
    def _get_g_lambda(self, _grandMM_naive):
        self._g_lambda = np.sum(np.abs(_grandMM_naive), axis=1)[::-1]

    def _preprocessing(self):
        # print('naive')
        self._v_naive, self._beta_naive, self._grandLambda, self._grandT_naive, self._grandN_naive, _grandMM_naive = utils.get_lambda_naive(self._um_diff, self._tau)
        self._get_g_lambda(_grandMM_naive)

    def _get_u_dp_vt(self, l_select):
        u_ours = utils.get_u_naive(self._v_naive, self._grandT_naive, self._grandN_naive, self._beta_naive, self._grandLambda, self._t, l_select, t0 = self._t[0])
        return u_ours