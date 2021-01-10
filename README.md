# Demo code for automatic tv denoising
This repertory contains the demo code for 1D Total variation (TV) denoising method with an automatic choice of hyper-parameter $`\lambda`$ proposed in  *[Revisit 1D Total Variation restoration problem with new real-time algorithms for signal and hyper-parameter estimations](https://hal.inria.fr/hal-03079211)*. 

Let's note the sample vector $`y = (y_1, \cdots, y_n)`$ and the sample time vector $`t = (t_1, \cdots, t_n)`$ with $`n`$ the number of samples. We introduce the sampling period vector $`\tau = (\tau_1, \cdots, \tau_n)`$ with $`\tau_i = t_{i+1}- t_i`$ and $`\tau_1 = \tau_2`$. We considered the TV denoisng method:
```math
u^*(\lambda) = \arg \min_u \{\sum_{i=1}^n\tau_i(y_i-u_i)^2 + \lambda \sum_{i=1}^{n-1}|u_{i+1} - u_i|\}
```
with $`\lambda`$ a given hyper-parameter and $`u^*(\lambda)`$ the proposed restoration.

We propose to estimate $`\lambda`$ by tracking the discontinuity of the tendency of the local extremun number (noted $`g(\lambda)`$) of the restoration $`u^*(\lambda)`$ in function of $`\lambda`$. 

The key feature is to find out the end of the transitory region of $`\partial^2 g(\lambda)`$. In this repertory, we proposed 3 methods for estimating the parameter $`\lambda`$ based on $`\partial^2 g(\lambda)`$:
* Automatic method: this method is described in Section 3.3 by tracking $`\partial^4 g(\lambda)`$. The simulations show our automatic approach for $`\lambda`$ estimation has a similar performance as cross validation and SURE, and our approach is well-fitted for the real time context.
* Semi-automatic method: let's note $`\lambda_{trans} = \arg\max  \partial^2 g(\lambda)`$ giving the index of a point inside the transitory region. We propose $`\lambda_{semi}`$ as the last element of $`\{\lambda > \lambda_{trans} |\lambda\in \lambda^g\} \cup \{\partial^2 g(\lambda) \geq \partial^2 g(\lambda_{trans} - n\times p \times 10\log_{10}q)\}`$ with $`p`$ and $`q`$ two parameters. Users can adapt the variation of restoration for different applications by choosing the parameters $`p`$ and $`q`$.
* Manuel method: Users give the parameter $`\lambda`$.

This code works only under the <ins>continuous noise hypothesis</ins>. More precisely, we assume the two following conditions :
* $`y_i \neq y_{i+1}`$ for every sample.
* Only two segments are merged together for every $`\lambda \in \Lambda`$ following Remark 1 in the article.

## Parameter selection 
Some parameters are required by our methods:
* q : this parameter is used to calculate $`\partial^2 g(\lambda)`$. Typically, $`0.5\leq q\leq 1`$. An automatic choice of $`q`$ is also available, described in Page 13 of the article.
* p : the parameter for the adjustment of $`\lambda`$ in semi-automatic method. A larger value of $`p`$, giving a larger $`\lambda`$, proposes a restoration with less variation. Usually, $`\log_{10} q = 1`$ and  $`p=0.02`$ give a good restoration performance. But feel free to choose the parameter $`p`$ for your own application.

## Dependencies
It is written in pure Python and depends on the following libraries: `numpy`, `scipy`(optional) and `matplotlib` (optional).

 

## Usage
### Initialization 

```python
from denoise_tv.denoise_tv import Denoise_tv_naive
deno = Denoise_tv_naive(y, t) # y: sample vector, t: sampling time vector
```

### TV-denoisng with automatic method for $`\lambda`$ estimation
Here an example wit automatic choice of the parameter $`q`$
```python
from denoise_tv.denoise_tv import Denoise_tv_naive
deno = Denoise_tv_naive(y, t)
deno.denoise_auto()
u, t = deno.get_restoration() # u the proposed restoration
l_select = deno.get_l_select() # get lambda_ours
```

For the manuel choice of parameter $`\log_{10}(q) = 1`$ 
```python
deno.denoise_auto(log=1)
```
### TV-denoisng with semi-automatic method for $`\lambda`$ estimation
```python
deno.denoise_semi_auto(log=1, p=0.02)
u, t = deno.get_restoration() # u the proposed restoration
l_select = deno.get_l_select() # get lambda_ours
```
We can also use the automatic choice of parameter $`q`$
```python
deno.denoise_semi_auto(p=0.02)
```
### TV-denoisng with given $`\lambda`$
```python
deno.denoise_manuel(lam=10)
u, t = deno.get_restoration() # u the proposed restoration
```
### Estimation the optimal restoration in knowing the original signal $`u0`$
```python
u_op, l_op = deno.compute_lambda_op(u0=u0)
```

### Existing methods: SURE et AUT
In the cas of known noise variance $`\sigma^2 = 9`$ 
```python
u_sure, l_sure = deno.get_lambda_sure(sigma=3)
u_aut, l_aut = deno.get_lambda_aut(sigma=3)
```

In the cas of unknown noise variance, $`\sigma`$ is estimated by MAD :
```python
u_sure, l_sure = deno.get_lambda_sure()
u_aut, l_aut = deno.get_lambda_aut()
```

### Display of noisy and denoised signal 
```python
deno.show_denoised_signal(plot_aut=True, plot_sure=True, u0=u0)
```
add figure
    
### Display of the extremun number $`g(\lambda)`$ in function of $`\lambda`$
```python
deno.show_g_lambda(plot_ours=True, plot_aut=True, plot_sure=True, plot_op=True)
```
add figure

## Citation
We would appreciate citations to the following paper:
```
@unpublished{liu:hal-03079211,
  TITLE = {{Revisit 1D Total Variation restoration problem with new real-time algorithms for signal and hyper-parameter estimations}},
  AUTHOR = {Liu, Zhanhao and Perrodin, Marion and Chambrion, Thomas and Stoica, Radu S.},
  URL = {https://hal.inria.fr/hal-03079211},
  NOTE = {working paper or preprint},
  YEAR = {2020},
  MONTH = Dec,
}
```