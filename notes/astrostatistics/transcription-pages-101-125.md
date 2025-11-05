---
(Page 101)

(SHEET 3) see week 3 for plots of answers

### SN cosmology Reparametrisation (board)

$M_s \sim N(M_0, \sigma^2_{int})$
$M_s = M_0 + E_{int}$

$M_s = m_s + 5\log_{10}(\frac{d_L}{Mpc}) + 25$
$= m_s + 5\log_{10}(d_L) - 5\log_{10}(Mpc) + 25$

$m_s = M_0 + \mu(z_s|H_0, \vec{\Omega}) + E_{int}$

$\vec{\Omega} = (S\Omega_M, S\Omega_\Lambda, w)$

$M_0 + \mu(z|H_0, \vec{\Omega}) = M_0 + 25 + 5\log_{10}(\frac{c}{100} \frac{100}{H_0} d_L(z_s, \vec{\Omega}))$
$\uparrow 100 \text{ kms}^{-1} \text{Mpc}^{-1}$

$= M_0 - 5\log_{10}(\frac{H_0}{100}) + 25 + 5\log_{10}(\frac{c}{100} \tilde{d_L}(z_s, \vec{\Omega}))$

$M_0 - 5\log_{10}h \dots + g(z_s, \vec{\Omega})$ rename
$\mathcal{M}_0$

> why important? only have SN, here (in sheet 3 had cepheids as well for which we knew their true distances) but here, no cepheids to constraint abs. mag. so degeneracy between M_0 may arise, e.g. H_0 cannot distinguish between M_0 and H_0 w/ SN data alone.
> So can only constrain the combination.
> is reparametrise problem. We can only constrain the linear combination.

$M_0$ and $5\log_{10}h$ are degenerate!

$P(m_s|z_s; \mathcal{M}_0, \vec{\Omega}, \sigma^2_{int}) = N(m_s | \mathcal{M}_0 + g(z_s, \vec{\Omega}), \sigma^2_{int})$

Likelihood: $L(\mathcal{M}_0, \vec{\Omega}, \sigma^2_{int}) = \prod_i P(m_s|z_s; \mathcal{M}_0, \vec{\Omega}, \sigma^2_{int})$ non informative

(Indep.) priors: $\mathcal{M}_0, \Omega_\Lambda \sim U(-\infty, \infty), \Omega_M, \sigma^2_{int} \sim U(0, \infty)$

Posterior: $P(\mathcal{M}_0, \vec{\Omega}, \sigma^2_{int} | \{m_s, z_s\}) \propto L(\mathcal{M}_0, \vec{\Omega}, \sigma^2_{int}) \times P(\mathcal{M}_0) P(\vec{\Omega}) P(\sigma^2_{int})$

---
(Page 102)

(slides)

### Multiple Indep. Chains
[A plot showing multiple MCMC chains over samples. The x-axis is labeled "sample". Three chains are shown in different colors (purple, blue, green). They are all exploring the same parameter space and overlap, which is labeled "chains overlap 'mixing' = good".]

ex

### Assessing Convergence of multiple chains: Gelman-Rubin (G-R) Ratio

Suppose we have simulated $m$ parallel sequences, each of length $n$ (after discarding the first half of the simulations). For each scalar estimated $\psi$, we label the simulation draws as $\psi_{ij}$ ($i=1,...,n; j=1,...,m$), and we compute $B$ and $W$, the between- and within-sequence variances:
> over one run (or chain) / all samples in a chain

> over samples (of a given chain)

$B = \frac{n}{m-1} \sum_{j=1}^{m} (\bar{\psi}_{.j} - \bar{\psi}_{..})^2 \qquad \bar{\psi}_{.j} = \frac{1}{n} \sum_{i=1}^{n} \psi_{ij}, \quad \bar{\psi}_{..} = \frac{1}{m} \sum_{j=1}^{m} \bar{\psi}_{.j}$
> $B$ = variance of the means for all samples for all chains

$W = \frac{1}{m} \sum_{j=1}^{m} s_j^2 \qquad s_j^2 = \frac{1}{n-1} \sum_{i=1}^{n} (\psi_{ij} - \bar{\psi}_{.j})^2$
> variance of each chain, averaged over all chains

We can estimate $\text{var}(\psi|y)$, the marginal posterior variance of the estimated estimand, by a weighted average of $W$ and $B$, namely:

$$ \widehat{\text{Var}}^{+}(\psi|y) = \frac{n-1}{n} W + \frac{1}{n} B $$
> No. Note that variance of underlying distribution $\psi$ from sample is total variance which should be sum of between chain variance + within chain variance.
> $W \times B$ then scale, then sum up? has converged $m=1, \frac{1}{n} B=0$. if not may run chains for longer

G-R ratio:
$$ \hat{R} = \sqrt{\frac{\widehat{\text{Var}}^{+}(\psi|y)}{W}} $$

> if variance within chain is same as variance between chains, $\hat{R} \sim 1$
> doesn't give measure on convergence/how well this has mixed
> if not proper/ stationary reached -> good value could be > 1. mixed

---
(Page 103)

> Have proposal scale to give acceptance ratio $\approx 30\%$
> indep
> for chains for params -> want G-R ratio $\approx 1$
> cut off ~20% for burn in (rule of thumb)
> see slides / sheet 3

[A sketch of a graph with "SN mag" on the y-axis and z on the x-axis. A curve is drawn, labeled "Nobel prize 2011".]

> MCMC -> can also plot posteriors on derived quantities (target)
> see slides/ lecture -> ex sheet 3.
> can compute posterior of any derived quantity e.g. deceleration parameter.
> $q_0 = 1$
> if high prob $q_0 < 0$ universe is decelerating.

---

> B: "between" sample "chain"
> running count of means for each chain
> avg of means for each chain over all chains

$B = \frac{1}{m-1} \sum_{j=1}^{m} (\sum_{i=1}^{n} \psi_{ij} - \frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{n} \psi_{ij})^2$

thinker of $B$ as variance in the mean of each chain times $n$.
since $\text{Var}(\bar{x}) = \frac{1}{g} \text{Var}(x)$.
(this variance is calculated from a sampler)
this restores the scale to match the variance of individual samples (sheet 6 of 7)

Summary: $W$: variance within chain
$B$: variance between chains scaled to individual sample variance

$\rightarrow$ law of total variance
Not sure if this is related? but got $ \text{Var}(x) = E[\text{Var}(x|chain)] + \text{Var}(E[x|chain]) $
> $\uparrow$ (variational decomposition)

$\rightarrow$ correlated within-chain variance = $W$
$\rightarrow$ variance of chain means is $B/n$
$\rightarrow$ complicated, sample variance?

---
(Page 104)

## Lecture 19
_7.3.25_

Today: continue MCMC - applications to SN cosmology, assessing convergence & mixing, comparison of algos.

(sheet 3)

New case:
Now assume flat univ. $\Omega_k = 1-\Omega_M$ but unknown $\mathcal{M}_0, \sigma^2_{int}, \Omega_M, w$

[A small plot showing two wiggly lines for w and $\Omega_M$.]

changing $w \rightarrow$ v. small effect, so hard to constrain $w$.

now instead use Metropolis-within-gibbs
> sample w - for interest
> 4x1D for each parameter (fix all other params to their current value and update one) time 4x proposal scale & 4x acceptance ratios
> (visually doesnt look like the best sampling on slides (not mixed with))

2D. plot to notice banded structure in samples: signature of the algorithm. (lack of mixing)
can use estimated cov. matrix (from initial MCMC or Laplace approx) for correlated proposal dist. in 4D metropolis?
> helps it know what direction to sample in

use this as proposal for 4D metropolis -> see much better movement/mixing, chains look more similar -> improvement!

[A scatter plot for w and $\Omega_M$ showing a banana-shaped posterior distribution.]
> use initial run as a proposal dist for a new 4D metropolis run.

> better define param dist to better match data so it guides us to sample the right direction (effectively explore sample space)
> Estimate cov matrix from sample points. This initial run is used as a proposal dist for a new 4D metropolis run.

---
(Page 105)

(slides)
> see pg. __ for many effective samples?

### How Many Iterations to get an independen t sample? Autocorrelation Function

For each scalar parameter $\theta$

Chain: $(\theta_1,...,\theta_N)$

Consider sample mean $\bar{\theta} = \frac{1}{N} \sum_{i=1}^N \theta_i$

If chain were serially uncorrelated: $\text{Cov}(\theta_i, \theta_j)=0$, $(i\neq j)$
then $\text{Var}[\bar{\theta}] = \text{Var}(\theta)/N$.

However, chain is typically serially correlated:
> autocorrelation coefficient
$\text{Cov}[\theta_i, \theta_j] = C_{|i-j|} = \text{Var}[\theta] \rho_{|i-j|}$.

Then for large $N$: $\text{Var}[\bar{\theta}] = \text{Var}(\theta) \times \frac{\tau}{N} = \text{Var}(\theta)/N_{eff}$

Where $\tau = 1+2\sum_{t=1}^\infty \rho_t$ is the AUTOCORRELATION TIMESCALE
> basically $\tau = 5$. how many samples are required to time 'before' your next 'indep.' sample?

and $N_{eff} = N/\tau$ is the EFFECTIVE SAMPLE SIZE
(ESS) (equivalent number of independent samples).
> after small n, $\rho$ depends on underlying dist.

PROOF: $\rightarrow$ Find $\text{Var}[\bar{\theta}]$
> (remainder section): $\text{Var}(\sum a_i x_i) = \sum a_i^2 \text{Var} x_i$
> serially correlated: errors at one point in chain are related to errors at another point in chain. each value in chain depends on values of other values in chain. as a fn. of distance between these points in chain.

$\text{Var}[\bar{\theta}] = \text{Var}[\frac{1}{N}\sum_i \theta_i] = \frac{1}{N^2} \text{Var}[\sum_i \theta_i] = \frac{1}{N^2} (\sum_{i,j} \text{Cov}(\theta_i, \theta_j))$
> reminder: variance of sum is sum of covariances!
> small correlation

> correction term: if just a few df between for infinitely large chain imagine going in chain direction & adding $j+1, j+2,...$. If it diverges you get factors of 2 going left & right of factors $N$. And if it doesn't then the fact that it converges in. for large n sums approach integral of $\rho(t)$
> for large N, $\text{Var}(\bar{\theta}) \approx \frac{\text{Var}(\theta)}{N} [1+2\sum_t \rho_t]$

---
(Page 106)

> do this in practice:

### Estimating the Autocorrelation / ESS
For each scalar parameter $\theta$:

Sample covariance of lag $t$:
$\hat{C}_t = \frac{1}{N-t} \sum_{i=1}^{N-t} (\theta_i-\bar{\theta})(\theta_{i+t}-\bar{\theta})$
> I can use this to get autocorr for distance t apart, then normalize variance
> $\hat{C}_0 = $ sample variance of $\theta$

$\hat{\rho}_t = \frac{\hat{C}_t}{\hat{C}_0} \quad$ Sample autocorrelation of lag $t$

$\hat{\tau} = 1+2 \sum_{t=1}^{\infty} \hat{\rho}_t$ Estimated autocorrelation time.
> Truncate at T lags s.t. $\hat{\rho}_T \approx 0.1$

$N_{eff} = N/\hat{\tau} \quad$ Effective number of independent samples

> want to estimate $\hat{\rho}_t, t=1... N-1$ until $\rho_t \approx 0$
> need to know no of samples that are correlated with current 'element'.
> only useful as long as sample dist is N-t apart (or N-t points).
> G-H gives sample distances far apart (t is large).

Slowest parameter is the limiting one!
> autocorr. time limited by slow-mixing from poor proposal

In practice, what matters is: time / $N_{eff}$.

### Compute time / Effectively Independent Sample

> weighs up how fast algo is vs how many effective indep samples it is required to get a final sample. Good to compare algos.

[A diagram with "Autocorrelation" on the y-axis and "lag" on the x-axis. A curve is shown starting high and decaying towards zero. It has a high peak at the beginning, indicating high correlation for small lags. Text next to it:
> slide: Gelman happily with gibbs
> autocorr is bad
> banded.
> better sampling -> don't see banded structure in sampling.

> log-log or lin-log? example
> N=10,000,
> $N_{eff} \approx 13$
> still lots of data loss. not good.
> $\approx 3$ to get one independent sample.

> two diff methods of sampling
> 1) 4D metropolis
> 2) metro-within-gibbs
> better leads to quicker mixing (larger $N_{eff}$)
]

---
(Page 107)

> can mix & match these algos: e.g.

(slides)
### Mixed Samplers
> with things that we are 'less certain' about

Target Posterior (intractable -cannot sample): $\pi(x, \beta, \theta)$

Suppose these $\pi(\alpha|\beta, \theta)$
$\pi(\beta|\alpha, \theta)$ are tractable
$\pi(\alpha, \beta|\theta)$

but $\pi(\theta|\alpha, \beta)$ is not tractable

> so can't do complete gibbs sample

Mixed sampler:
> so gibbs sampler for not intractable $\theta$ step with metropolis proposal ratio.

1) Sample $\alpha^t \sim \pi(\alpha|\beta^{t-1}, \theta^{t-1})$
2) Sample $\beta^t \sim \pi(\beta|\alpha^t, \theta^{t-1})$
3) Metropolis (proposal/acc./rej.) update $\theta^t$

### Parameter Blocking
Target posterior (intractable - cannot sample): $\pi(\alpha, \beta, \theta)$.
As before, suppose $\pi(\theta|\alpha, \beta)$ not tractable.
> other conditionals tractable viz $\pi(\alpha|\beta, \theta)$

Mixed Sampler with Blocking: (more efficient)
1) Jointly sample $\alpha^t, \beta^t \sim \pi(\alpha, \beta|\theta^{t-1})$
2) Metropolis (proposal/acc./rej.) update.
> usually more efficient

---
(Page 114)

(slides)
> postiror mode, $\theta_{MAP}$

### MCMC in Practice

1. Find the mode(s) using optimisation, run estimate to replace use Laplace approx. to obtain a proposal cov. matrix
> usually want to compute whole posterior, not just mode
> maybe not sufficient for multimodal distributions
> can explore region of interest

2. Begin multiple (4-8 parallel) chains at starting positions dispersed around the mode(s) (can draw from approx cov. matrix)
> maybe to avoid getting stuck at one mode.

3. Scale Metropolis proposals to tune 25-50% acceptance rate (depending on dimensionality of jump).

4. Use proposal cov. matrix that reflects the shape of the posterior (positive posterior shape, diagonal matrix?)

5. After run, look at chains (if possible) to check for obvious problems
> should look like noise

6. Compute Gelman-Rubin ratio comparing within-chain-variance to between-chain-variance to check that chains are well-mixed (should be very close to 1), and assess burn-in
> if R-hat not 1, might need longer runs, not proper sampling.

7. Compute autocorrelation timescale and effective sample size to make sure you have enough independent samples for inference

8. If all checks out, remove burn-in, thin, and combine chains to compute posterior inferences.

---
(Page 115)

### Overview of MCMC
* MCMC Algorithms
  - Metropolis/ M-H algorithms
  - Gibbs sampling
  - Metropolis-within-Gibbs
  - dont cover modern MCMC use gradient-assisted info -> Hybrid / HMC / NUTS
> most other algos are just specific versions of M-H

* Assessing/Comparing performance of MCMC algo.s.
  - Gelman-Rubin statistics for comparing mixing of multiple chains
  - Autocorrelation time - how long to get an independent sample?
  - Effective sample size - how many indep. samples do I have?

* Detailed balance & theoretical considerations

> end of MCMC!
> is next time gaussian processes.

---
(Page 116)

## Lecture 21
_12.3.25_

### GAUSSIAN PROCESSES (slides)

#### What is a Gaussian Process?

* A GP is a (possibly infinite) collection of R.V.s $\{f_t\}$ (typically indexed by some ordering in time, space or wavelength), such that any finite subset of R.V.s have a jointly multivariate Gaussian dist.

* Any vector $f = \{f_t: t=1,...,N\}$ of a finite subset is multivariate Gaussian, therefore it is completely described by a mean $E[f]$ and cov. matrix $\text{Var}[f] = \text{Cov}[f_t, f_{t'}]$.

* Elements of the cov. matrix are determined by a function of the coordinates e.g. $\text{Cov}[f_t, f_{t'}] = k(t,t')$ called the covariance function or kernel.

* A G.P. w/ mean function $m(t)$ and kernel $k(t,t')$ is denoted
$$ f(t) \sim GP(m(t), k(t,t')) $$

* A G.P. provides a distribution over functions
> say... e.g. $f_t = f(t_1), ..., f(t_N)$
> $f(t) \sim N \left( \begin{bmatrix} m(t_1) \\ m(t_2) \end{bmatrix}, \begin{bmatrix} k(t_1, t_1) & k(t_1, t_2) \\ k(t_2, t_1) & k(t_2, t_2) \end{bmatrix} \right)$
> this expression? is completely defined by its mean $m(t)$ and covariance $k(t, t')$

> the dist of a GP. is the joint of all these (compatibly many) variables i.e. it is a dist. over fns with a continuous domain (e.g. time or space)

---
(Page 117)

see notes on moodle
(slides)

### Review: Properties of multivariate Gaussian

Full prob. density: ($\Sigma$ positive definite)
$N(f|\mu, \Sigma) = [\det(2\pi\Sigma)]^{-1/2} \exp(-1/2 (f-\mu)^T \Sigma^{-1} (f-\mu))$

Joint dist. of components:
$f = \begin{pmatrix} U \\ V \end{pmatrix} \sim N \left( \begin{bmatrix} U_0 \\ V_0 \end{bmatrix}, \begin{bmatrix} \Sigma_U & \Sigma_{UV} \\ \Sigma_V & \Sigma_{VU} \end{bmatrix} \right)$

If you observe/know/condition on $V$:
Conditional dist. $U|V \sim N(E[U|V], \text{Var}[U|V])$
Conditional mean $E[U|V] = U_0 + \Sigma_{UV} \Sigma_V^{-1} (V-V_0)$
Conditional variance $\text{Var}[U|V] = \Sigma_U - \Sigma_{UV} \Sigma_V^{-1} \Sigma_{VU}$

If $V=$ observed data, $U=$ unobserved parameters,
then $P(U|V)$ is a posterior pdf!
> reminder, marginals: $P(V) = \int P(U,V)dU = N(V|\mu_V, \Sigma_V)$
> $P(U) = N(U|\mu_U, \Sigma_U)$
> (simply just drop irrelevant variable from mean vector & for matrix: e.g. $\mu_V \rightarrow \mu_U$
> $\Sigma = \begin{bmatrix} \Sigma_{UU} & \Sigma_{UV} \\ \Sigma_{VU} & \Sigma_{VV} \end{bmatrix} \rightarrow \Sigma_U$)
> then conditional's come from $P(U|V) = P(U,V)/P(V)$.
> (dont know if I need to know prior for this?)

---
(Page 118)

(slides)

### What are GPs used for in Astrophysics?
- Some physical models are very nearly gaussian e.g. CMB is G.P. on the sphere, or w/ gaussian
> (e.g damped random walks) or (curves, overprogression)

- "Nonparametric" models: flexible gns to use when an accurate parametric astrophysically-motivated fn. is not available or is imperfect.

- Nonparametric $\rightarrow$ no. of parameters/latent variables grows with the dataset.

- Interpolation/Emulation: to generate a smooth curve going through some observation or simulation points

- Correlated noise/error model: When you marginalise out the latent function, you are effectively accounting for noise/fluctuations correlated over time/space/wavelength $\rightarrow$ "nuisance function"
> (goals similar to individual patches from time series, combined spectrum. down side can't plot (?).)
> example: planetary phase, in-between times

ex1)
[A diagram shows a circle representing a stellar spectrum with absorption lines. Another plot shows a function with dips. The text says "e.g. disentangle signal from binary". A wiggly line is drawn below, labeled "noise". Another line is drawn, labeled "continuum".]
> target decided want to fit GP to data points, draw from GP to get smooth interpolation to fill in gaps.

ex2)
[A diagram with a grid representing an image, with text "lensed quasar: brightness randomly fluctuates in time but... behind galaxy: it will be lensed & light takes two copies, fluctuates on an image will appear fixed time later than on the other image... use GP to describe underlying fcn so we can... said time delay... use GP to model underlying (latent) light curve (a damped random walk).]
> lensing galaxy

ex3)
> Supernovae can also... weight back... cancel... get a copy of the same SN?
> but copies are magnified by some value is diff 'time shift' between copies...
> model light curves as realisations of GP, & introduce params to estimate $\theta_1, \theta_2$?

---
(Page 119)

e.g. fit GP to SN light curve time series

### G.P. as a prior on functions (slides)
But we only ever evaluate the fn. on a finite set of points!

A grid of times: $t = (t^1, ..., t^i, ..., t^N)^T$
A vector of fn. values evaluated on the time grid:
$f = (f(t^1), ..., f(t^i), ..., f(t^N))^T$

Assume a squared exponential kernel / cov. fn.:
$\text{Cov}[f(t), f(t')] = k(t,t') = A^2 \exp(-(t-t')^2/\tau^2)$

Assume a const. prior mean fn.:
$E[f(t)] = m(t) = C$.
> often assume zero-mean C=0.

Prior on function: $P(f|A, \tau) = N(f|\mathbb{1}c, K)$
Drawing from prior: $f|A, \tau \sim N(\mathbb{1}c, K)$

Cov. matrix $K$ is populated by evaluating the kernel:
$\text{Cov}[f(t), f(t')] = K(t,t') = A^2 \exp(-(t-t')^2/\tau^2)$
for all pairs of points in t:
$K_{ij} = k(t_i, t_j) = A^2 \exp(-(t_i-t_j)^2/\tau^2)$

> how to choose this kernel? set A, C, $\tau$ to positive values & generate say 20 random fns from GP prior
> (notice how fns change if we change A, $\tau$)
> hyper-parameters, k values since they affect covariance structure.

---
(Page 120)

### Fitting a GP to data (slides)

1) PREDICTION
If we knew the characteristic scales of the kernel (hyperparameters) ($A, \tau^2$) then how do we fit the data at observed times to find the curve for unobserved times?

2) MODEL SELECTION
Given the observed data, how do we fit for the characteristic scales of the kernel (hyperparameters)?

#### Posterior inference w/ G.P.s, Estimating the underlying curve:
$f_0 = $ fn. at observed times $t_0$ (training set)
$f_* = $ fn. at unobserved times $t_*$ (prediction set)

Joint:
$\begin{pmatrix} f_0 \\ f_* \end{pmatrix} \sim N \left( \begin{bmatrix} \mathbb{1}c \\ \mathbb{1}c \end{bmatrix}, \begin{bmatrix} K(t_0, t_0) & K(t_0, t_*) \\ K(t_*, t_0) & K(t_*, t_*) \end{bmatrix} \right)$

Populating the cov. matrix:
$K(t,t')$ has i,j-th entry $=k(t_i, t_j)$
Using the assumed kernel fn.
$\text{Cov}[f(t), f(t')] = k(t,t') = A^2 \exp(-(t-t')^2/\tau^2)$

> using properties of MVG vectors
> partition

---
(Page 121)

Posterior (conditional on observations, is also Gaussian):
$f_* | f_0 \sim N(E[f_*|f_0], \text{Var}[f_*|f_0])$

Posterior predictive mean:
$E[f_*|f_0] = \mathbb{1}c + K(t_*, t_0)K(t_0, t_0)^{-1}(f_0 - \mathbb{1}c)$

Posterior predictive variance/covariance:
$\text{Var}[f_*|f_0] = K(t_*, t_*) - K(t_*, t_0)K(t_0, t_0)^{-1}K(t_0, t_*)$

> ignore measurement error -> bad
> GP tries to get by going through centre of every single data point
> include measurement error: good

> how do we account for this?

### Accounting for measurement error

$y_0 | f_0 \sim N(f_0, W)$
> "model of measurement error as another gaussian process"

$y_0$ are measured values of $f_0$ at time $t_0$.
at each observation $y_0^i | f_0^i \sim N(f_0^i, \sigma_i^2)$

$W$ is measurement cov. matrix.
Most common case: heteroskedastic uncorrelated measurement error:
$\text{Cov}(\epsilon_i, \epsilon_j) = W_{ij} = \delta_{ij} \sigma_i^2$

Measurement error model: (
data = latent value + meas. error
$y_0 = f_0 + \epsilon$
$\epsilon \sim N(0, W)$
(mean-zero Gaussian noise)
)
> cov matrix W

---
(Page 122)

[Derivation as the sum of two GPs at the observed times: Probabilistic Generative Mode]

GP of intrinsic curve
$f(t) \sim GP(m(t)=c, k(t,t'))$

$f_0=$ latent fn. at observed times to
GP Prior: $f_0 \sim N(\mathbb{1}c, K(t_0, t_0))$

GP of measurement error:
$y_0 = $ measurements (with error) of $f_0$ at times $t_0$
$y_0 | f_0 \sim N(f_0, W)$

Same as: (mean-zero error)
$y_0 = f_0 + \epsilon \qquad \epsilon \sim N(0, W)$

GOAL: Need Joint Dist. of data and latent fn.
> data we observe
> data at unobserved times
$ \begin{pmatrix} y_0 \\ f_* \end{pmatrix} \sim N([\;?\;], [\;?\;]) $

Then can calc fn. (posterior) prediction at unobserved points
$f_*|y_0 \sim N(E[f_*|y_0], \text{Var}[f_*|y_0])$

Using conditional properties of MV Gaussian as before.
$ \begin{pmatrix} y_0 \\ f_* \end{pmatrix} \sim N \left( \begin{bmatrix} E[y_0] \\ E[f_*] \end{bmatrix}, \begin{bmatrix} \text{Cov}[y_0, y_0] & \text{Cov}[y_0, f_*] \\ \text{Cov}[f_*, y_0] & \text{Cov}[f_*, f_*] \end{bmatrix} \right) $
> is this possible/ 'admissable' to use this?

---
(Page 123)

Intrinsic/Latent Process: $f_0 \sim N(\mathbb{1}c, K(t_0, t_0))$
Measurement Process: $y_0 | f_0 \sim N(f_0, W)$
$y_0 = f_0 + \epsilon \qquad \epsilon \sim N(0, W)$ (meas. error)

To find cov. submatrices, apply bilinearity of covariance
$\text{Cov}(y_0, y_0) = \text{Cov}(f_0, f_0) + \text{Cov}(\epsilon, \epsilon) + 2\text{Cov}(f_0, \epsilon)$
$\text{Cov}(f_0, f_0) = K(t_0, t_0)$ (GP of intrinsic curve)
$\text{Cov}(\epsilon, \epsilon) = W$ (measurement noise)
(the two processes are indep. -> uncorrelated)
$2\text{Cov}(f_0, \epsilon)=0$
$\therefore \quad \text{Cov}[y_0, y_0] = K(t_0, t_0) + W$

Use joint of latent $f$ at observed and unobserved times
$ \begin{pmatrix} f_0 \\ f_* \end{pmatrix} \sim N \left( \begin{bmatrix} \mathbb{1}c \\ \mathbb{1}c \end{bmatrix}, \begin{bmatrix} K(t_0, t_0) & K(t_0, t_*) \\ K(t_*, t_0) & K(t_*, t_*) \end{bmatrix} \right) $

to derive similar arguments for
$\text{Cov}[y_0, f_*] = \text{Cov}\{f_0, f_*\} + \text{Cov}\{\epsilon, f_*\} = K(t_0, t_*) + 0$
$\text{Cov}[f_*, f_*] = K(t_*, t_*)$
> remember $f_0, f_*$ correlated

Fill out cov. matrix:
$ \begin{pmatrix} y_0 \\ f_* \end{pmatrix} \sim N \left( \begin{bmatrix} \mathbb{1}c \\ \mathbb{1}c \end{bmatrix}, \begin{bmatrix} K(t_0, t_0)+W & K(t_0, t_*) \\ K(t_*, t_0) & K(t_*, t_*) \end{bmatrix} \right) $

---
(Page 124)

now can compute fn. prediction at unobserved points:

Posterior Predictive (conditional on data $y_0$)
$f_* | y_0 \sim N(E[f_*|y_0], \text{Var}[f_*|y_0])$

Using MV Gaussian conditional properties:
Posterior predictive mean
$E[f_*|y_0] = \mathbb{1}c + K(t_*, t_0) [K(t_0, t_0)+W]^{-1} (y_0 - \mathbb{1}c)$
Posterior predictive variance/covariance
$\text{Var}[f_*|y_0] = K(t_*, t_*) - K(t_*, t_0) [K(t_0, t_0)+W]^{-1} K(t_0, t_*)$

[Two plots are shown. The first plot shows a wiggly curve fitting through several data points exactly. This is labeled "ignore meas error overfitting". The second plot shows a similar curve but it passes near the data points, not necessarily through them, with error bars shown on the points. This is labeled "include meas. error". An arrow points to this second plot with the text "looks incl. like an actual supernova".]

(see slides)

---
(Page 125)

### Now we know how to do (1) PREDICTION, how do we do (2) MODEL SELECTION?

#### Tuning hyperparameters ($A, \tau$)

Recall our probabilistic generative model:
Intrinsic/latent GP process
(make explicit dependence on hyperparameters)
$f(t) \sim GP(m(t)=C, K_{A,\tau^2}(t,t'))$
kernel: $k_{A,\tau^2}(t,t')=A^2\exp(-(t-t')^2/\tau^2)$
$f_0 \sim N(\mathbb{1}c, K_{A,\tau^2}(t_0,t_0))$

Gaussian measurement process $y_0 = f_0 + \epsilon \quad \sim N(0,W)$

Integrating out the latent fn. $f(t)$
(or use addition of MVN R.V.s) gives us the marginal likelihood:
> $\int P(y_0|f_0, A, \tau^2) P(f_0|A, \tau^2) df_0 = \int P(y_0|f_0, W) P(f_0|c, K_{A,\tau^2}) df_0$
> $N[y_0|f_0, W] \times N[f_0|\mathbb{1}c, K(t_0,t_0)]$

$P(y_0|A, \tau^2) = \int P(y_0|f_0) \times P(f_0|A, \tau^2) df_0$
> meas-error, latent GP
> joint Normal -> Marginal is $N(y_0|\mathbb{1}c, K_{ij} + C_{ij})$

$L(A, \tau^2) = P(y_0|A, \tau^2) = N[y_0|\mathbb{1}c, K_{A,\tau^2}(t_0,t_0)+W]$

* W = meas. error covariance
* $K_{A,\tau^2}$ = GP covariance

Which we can optimise (max likelihood) or specify a prior on ($A, \tau$) and sample from posterior.
> results for this example on slides.
> find for this case its not obvious that this has 10 significant parameters