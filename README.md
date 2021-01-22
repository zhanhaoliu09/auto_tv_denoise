# Demo code for automatic tv denoising
This repertory contains the demo code for 1D Total variation (TV) denoising method with an automatic choice of hyper-parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda"> proposed in  *[Revisit 1D Total Variation restoration problem with new real-time algorithms for signal and hyper-parameter estimations](https://hal.inria.fr/hal-03079211)*. 

Let's note the sample vector <img src="https://render.githubusercontent.com/render/math?math=y = (y_1, \cdots, y_n)"> and the sample time vector <img src="https://render.githubusercontent.com/render/math?math=t = (t_1, \cdots, t_n)"> with <img src="https://render.githubusercontent.com/render/math?math=n"> the number of samples. We introduce the sampling period vector <img src="https://render.githubusercontent.com/render/math?math=\tau = (\tau_1, \cdots, \tau_n)"> with <img src="https://render.githubusercontent.com/render/math?math=\tau_i = t_{i+1}- t_i"> and <img src="https://render.githubusercontent.com/render/math?math=\tau_1 = \tau_2">. We considered the TV denoisng method:

<img src="https://render.githubusercontent.com/render/math?math=u^*(\lambda) = \arg \min_u \{\sum_{i=1}^n\tau_i(y_i-u_i)^2 + \lambda \sum_{i=1}^{n-1}|u_{i+1} - u_i|\}"> 

with <img src="https://render.githubusercontent.com/render/math?math=\lambda"> a given hyper-parameter and <img src="https://render.githubusercontent.com/render/math?math=u^*(\lambda)"> the proposed restoration.

We propose to estimate <img src="https://render.githubusercontent.com/render/math?math=\lambda"> by tracking the discontinuity of the tendency of the local extremun number (noted <img src="https://render.githubusercontent.com/render/math?math=g(\lambda)">) of the restoration <img src="https://render.githubusercontent.com/render/math?math=u^*(\lambda)"> in function of <img src="https://render.githubusercontent.com/render/math?math=\lambda">. 

The key feature is to find out the end of the transition region of <img src="https://render.githubusercontent.com/render/math?math=\partial^2 g(\lambda)">. In this repertory, we proposed 3 methods for estimating the parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda"> based on <img src="https://render.githubusercontent.com/render/math?math=\partial^2 g(\lambda)">:
* Automatic method: this method is described in Section 3.3 by tracking <img src="https://render.githubusercontent.com/render/math?math=\partial^4 g(\lambda)">. The simulations show our automatic approach for <img src="https://render.githubusercontent.com/render/math?math=\lambda"> estimation has a similar performance as cross validation and SURE, and our approach is well-fitted for the real time context.
* Semi-automatic method: let's note <img src="https://render.githubusercontent.com/render/math?math=\lambda_{trans} = \arg\max  \partial^2 g(\lambda)"> giving the index of a point inside the transitory region. We propose <img src="https://render.githubusercontent.com/render/math?math=\lambda_{semi}"> as the last element of <img src="https://render.githubusercontent.com/render/math?math=\{\lambda > \lambda_{trans} |\lambda\in \lambda^g\} \cup \{\partial^2 g(\lambda) \geq \partial^2 g(\lambda_{trans} - n\times p \times \log_{10}q)\}"> with <img src="https://render.githubusercontent.com/render/math?math=p"> and <img src="https://render.githubusercontent.com/render/math?math=q"> two parameters. Users can adapt the variation of restoration for different applications by choosing the parameters <img src="https://render.githubusercontent.com/render/math?math=p"> and <img src="https://render.githubusercontent.com/render/math?math=q">.
* Manuel method: Users give the parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda">.

This code works only under the <ins>continuous noise hypothesis</ins>. More precisely, we assume the two following conditions :
* <img src="https://render.githubusercontent.com/render/math?math=y_i \neq y_{i+1}"> for every sample.
* Only two segments are merged together for every <img src="https://render.githubusercontent.com/render/math?math=\lambda \in \Lambda"> following Remark 1 in the article.

## Parameter selection 
Some parameters are required by our methods:
* q : this parameter is used to calculate <img src="https://render.githubusercontent.com/render/math?math=\partial^2 g(\lambda)">. Typically, <img src="https://render.githubusercontent.com/render/math?math=0.5\leq q\leq 1">. An automatic choice of <img src="https://render.githubusercontent.com/render/math?math=q"> is also available, described in Page 13 of the article.
* p : the parameter for the adjustment of <img src="https://render.githubusercontent.com/render/math?math=\lambda"> in semi-automatic method. A larger value of <img src="https://render.githubusercontent.com/render/math?math=p">, giving a larger <img src="https://render.githubusercontent.com/render/math?math=\lambda">, proposes a restoration with less variation. Usually, <img src="https://render.githubusercontent.com/render/math?math=\log_{10} q = 1"> and  <img src="https://render.githubusercontent.com/render/math?math=p=0.02"> give a good restoration performance. But feel free to choose the parameter <img src="https://render.githubusercontent.com/render/math?math=p"> for your own application.

## Dependencies
It is written in pure Python and depends on the following libraries: `numpy`, `scipy`(optional) and `matplotlib` (optional).

 

## Usage
### Initialization 

```python
from denoise_tv.denoise_tv import Denoise_tv_naive
deno = Denoise_tv_naive(y, t) # y: sample vector, t: sampling time vector
```

### TV-denoisng with automatic method for <img src="https://render.githubusercontent.com/render/math?math=\lambda"> estimation
Here an example wit automatic choice of the parameter <img src="https://render.githubusercontent.com/render/math?math=q">
```python
from denoise_tv.denoise_tv import Denoise_tv_naive
deno = Denoise_tv_naive(y, t)
deno.denoise_auto()
u, t = deno.get_restoration() # u the proposed restoration
l_select = deno.get_l_select() # get lambda_ours
```

For the manuel choice of parameter <img src="https://render.githubusercontent.com/render/math?math=\log_{10}(q) = 1">
```python
deno.denoise_auto(log=1)
```
### TV-denoisng with semi-automatic method for <img src="https://render.githubusercontent.com/render/math?math=\lambda"> estimation
```python
deno.denoise_semi_auto(log=1, p=0.02)
u, t = deno.get_restoration() # u the proposed restoration
l_select = deno.get_l_select() # get lambda_ours
```
We can also use the automatic choice of parameter <img src="https://render.githubusercontent.com/render/math?math=q">
```python
deno.denoise_semi_auto(p=0.02)
```
### TV-denoisng with given <img src="https://render.githubusercontent.com/render/math?math=\lambda">
```python
deno.denoise_manuel(lam=10)
u, t = deno.get_restoration() # u the proposed restoration
```
### Estimation the optimal restoration in knowing the original signal <img src="https://render.githubusercontent.com/render/math?math=u0">
```python
u_op, l_op = deno.compute_lambda_op(u0=u0)
```

### Existing methods: SURE et AUT
In the cas of known noise variance <img src="https://render.githubusercontent.com/render/math?math=\sigma^2 = 9"> 
```python
u_sure, l_sure = deno.get_lambda_sure(sigma=3)
u_aut, l_aut = deno.get_lambda_aut(sigma=3)
```

In the cas of unknown noise variance, <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is estimated by MAD :
```python
u_sure, l_sure = deno.get_lambda_sure()
u_aut, l_aut = deno.get_lambda_aut()
```

### Display of noisy and denoised signal 
```python
deno.show_denoised_signal(plot_aut=True, plot_sure=True, u0=u0)
```

    
### Display of the extremun number <img src="https://render.githubusercontent.com/render/math?math=g(\lambda)"> in function of <img src="https://render.githubusercontent.com/render/math?math=\lambda">
```python
deno.show_g_lambda(plot_ours=True, plot_aut=True, plot_sure=True, plot_op=True)
```


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
