---
(Page 76)

> satellite galaxies
> large magellanic cloud
> milky way (south galaxy)
> measurements

## Astrostatistics case study
**"Bayesian estimates of the milky way and andromeda masses using high-precision astrometry & cosmological simulations"**

* want to measure mass of milky way (constrains mass of our galaxy)
* satellite galaxies of our milky way (LMC) -> use their orbits/motions to estimate mass of our own galaxy.
* measure distance of satellite galaxy to center galaxy, angular momentum etc.
* use simulation to plot kinematic properties against mass of host galaxy. (proportional to mass)
* use bayesian inference to infer actual mass of our galaxy from measured kinematic properties & results of simulations
* illustris simulation: simulate population of galaxies, galaxies attract mass, galaxies form, interact, get some dark matter, stars, galaxies... (web)
* can measure properties of satellite galaxy (proxies in most of simulations) get catalogue of properties.
* simulation generates samples (prior from) p(angular mom...) -> intractable in sense that we can't write it down but we can draw from it.

velocities(v), positions(r), moments(j) of a satellite correlated w/ central galaxy mass via galaxy formation simulators (physical prior) $v,r,j$ = latent (true) values. $M_{vir}$ = mass of galaxy, $\theta$=params I, $M_{vir}$
(joint prior p(v,r,j, M_vir | $\theta$ from simulations))
from HST we have
measure LMC data, v, r, j, we have ...
astrometric measurement! likelihood $p(x|d)$, now use importance sampling to estimate posterior.
$L(d) = L(x|d) = N(V_{obs} | V, \sigma_v^2) \times N(V_{obs_t} | V_t, \sigma_v^2)$
$ \times N(V_{obs} | V_t, \sigma_v^2)$
-> bayesian inference
-> importance sampling
-> kernel density estimation

[Diagram of two elliptical orbits representing trajectories around a central point labeled "LMC". The orbits are contained within a larger circle labeled "my disk plane". The diagram is annotated "trajectories depend on mass of milky way".]

---
(Page 77)

> (board)

### IMPORTANCE SAMPLING THE MILKY WAY GALAXY (Mald+17)

* $\vec{d}$ = (noisy) measurements of kinematic properties of our Milky way satellite's (e.g. large Magellanic Cloud) properties (e.g. distance from central galaxy, velocity, angular momentum...) $\vec{d} = v, r, j$.
* $\vec{z}$ = latent (true) values of satellite properties
* $m = log_{10}$ [mass of central galaxy]

**GOAL:** Estimate $E[g(m) | \vec{d}] = \int g(m) P(m|\vec{d}) dm$

(e.g. let $g(m)=m$ for posterior mean or $g(m)=(m-E[m|\vec{d}])^2$ for posterior variance)

Posterior mean: $E[m|\vec{d}] = \int m P(m|\vec{d}) dm$
> marginalise posterior

> full posterior
> = $\int m \left[ \int P(\vec{z},m|\vec{d}) d\vec{z} \right] dm$

$P(\vec{z},m|\vec{d}) = \frac{P(\vec{d}|\vec{z}) P(\vec{z}, m)}{\iint P(\vec{d}|\vec{z}) P(\vec{z},m) d\vec{z}dm}$
> (Gaussian) measurement likelihood
> physical prior (encoded in cosmo. sims)

> why not $p(z|m)$
> pragmatic assumption that prior knowledge from cosmo sims can provide any additional information

$E[m|\vec{d}] = \frac{\iint m P(\vec{d}|\vec{z}) P(\vec{z},m) d\vec{z}dm}{\iint P(\vec{d}|\vec{z}) P(\vec{z},m) d\vec{z}dm}$
> (bottom is the evidence, just need to integrate again)

Cosmo. sims generate galactic systems that can be thought of as giving us samples $(\vec{z}_j, m_j)$
$j=1, ..., m$ from the physical prior ($intractable$) $P(\vec{z},m)$.

> we want to use the samples of the prior
> -> we want $E[m|\vec{d}]$, so need $P(m|\vec{d})$ which we get from marginalising $P(\vec{z}, m|\vec{d})$. We know $p(\vec{d}|\vec{z})$ from some noise model, $P(\vec{z}, m|\vec{d}) = P(\vec{d}|\vec{z})P(\vec{z},m) / P(\vec{d})$. Then integrate to get $P(m|\vec{d})$ and integrate again to get $E[m|\vec{d}]$.

---
(Page 78)

-> use the prior itself as the importance function!
> with IS can get away with intractable & label sampling since they cancel!

Apply Monte Carlo approx.:
$E[m|\vec{d}] \approx \frac{\frac{1}{m}\sum_{j=1}^{m} m_j P(\vec{d}|\vec{z}_j)}{\frac{1}{m}\sum_{j=1}^{m} P(\vec{d}|\vec{z}_j)} = \sum_{j=1}^{m} m_j w_j$
> treatable. think.

> E.g. from sims we have samples from prior but not interested in it directly.
> now we only care about samples from post but can't sample (or eval) likelihood (itself)

(Repeat for posterior variance)
> self-norm importance weight
$w_j = \frac{P(\vec{d}|\vec{z}_j)}{\sum_{k=1}^{n} P(\vec{d}|\vec{z}_k)}$

$\sum_{j=1}^{n} w_j = 1$

REMARKS
$P(d|z,m) = P(d|z)$
$d$ conditionally indep. of $m$ given $z$.
> conditional independence assumption. Roughly means that if I knew the $z$ value measurement error would just add noise around that and all the information about the mass is contained in the model.
> meaning: distance or velocity of a satellite galaxy does not provide info directly on the prior on mass on which comes from ranging in the prior on.

> reminder by taking unconditional posterior to $q$ i.e. prior so what happens to that because we used prior as q so it effectively cancels out because we used prior as q.
$w_j \propto \frac{P(d|z)P(z|m)}{Q(z_j,m_j)}$
> include prior as importance function itself

> because we get away with not being able to evaluate $q(z_j,m_j)$ itself (because we can only get samples from intractable distribution) since it cancels with the prior!

---
(Page 79)

## Lecture 16
_28.2.25_

{
Today: continuing Bayesian computation, importance sampling, MCMC
recommend: paper Speagle J "A conceptual Introduction to Markov chain Monte Carlo Methods".
> don't need to do gibbs sampling/MCMC for everything. important sampling still this useful
}

### WEIGHTED KERNEL DENSITY ESTIMATE
> (slides)

$Wkde(\theta) = \sum_{s=1}^{M} w_s \times N(\theta | \theta_s, bw^2)$
> ($w_s := \frac{P(\theta_s | D)}{Q(\theta_s)}$)

$w_s$ = normalised importance weights

bandwidth: Silverman's rule of thumb $bw = \left(\frac{4\hat{\sigma}^5}{3n}\right)^{1/5}$

Estimate $\sigma^2$ from posterior $Var[\theta]$ estimate.
Use importance sampling effective sample size (ESS) for $n$.

### Ideal case:
if equal weights: $w_i=1/m$ reduces to $M = \frac{1}{1+0} = m$

$kde(\theta) = \sum_{s=1}^{m} \frac{1}{m} \times N(\theta | \theta_s, bw^2)$

---
(Page 80)

continue from example before:
### Posterior of Milky Way w/ weighted KDE
> see also Bayesto sheet 2 q3

> (slides)

[A graph with P(Mvir|d) on the y-axis and Mvir on the x-axis, which is on a log scale from 10^11 to 10^13. There are two curves. The blue curve, labeled "prior P(Mvir | V, r, j)", is a broad distribution centered around 10^12. The black curve, labeled "posterior P(Mvir | V, r, j)", is a sharper distribution, shifted slightly to the right of 10^12. The black curve is annotated "MW Mvir = $1.7^{+1.33}_{-0.52}$ x 10^12".]

> posterior given LMC data
> what does it mean? mean weight?

### Posterior HPD
X% HPD (X% credible interval(s) with highest density containing X% of posterior)

[Two similar graphs of a unimodal posterior distribution. The first shows a 40% HPD interval as a shaded region in the center, with the label "40% HPD = [11.83, 12.65]". The second shows a 68% HPD interval, a wider shaded region, labeled "68% HPD = [12.13, 12.42]".]

[A graph of a posterior distribution with mean and mode marked. mean=12.25, mode=12.28.]

> can make combining posterior from data from multiple satellite galaxies & do same calculation w/ weighted KDE.
> combine inference from different sources to get more accurate estimate (this is an ideal case for Bayesian inference)

---
(Page 81)

> Transcribe: MCMC provides samples from the posterior. The purpose of MCMC is to actually do the integral to evaluate expectation.

### MARKOV CHAIN MONTE CARLO (MCMC)
> (slides)

> classic use for MCMC for sample from some probability dist that you disk that are too complex or high dimensional to study with analytic techniques alone (not necessarily Bayesian).

Goal is to evaluate the posterior $P(\theta|D)$:

* Simple likelihoods/conjugate priors admit analytic solutions to the posterior
> e.g. exponential family distribution given that data follow a particular distribution...
> but these don't apply for all problems

* Simple models may allow direct draws $\theta_i \sim P(\theta|D)$ i.e. "direct simulation"

* Small numbers of parameters p (small dimensionality) -> Evaluate posterior on a p-dimensional grid.
Inefficient for $p>3$ (end up wasting compute time evaluating $P(\theta|D)$ where it is close to zero.

* Realistic models with many parameters
(high dimensional parameter space)
> and we have no analytic solution
-> Markov Chain Monte Carlo is the gold standard!

Whole point of MCMC is to generate samples from the posterior when we can't directly sample from it
$E[g(\theta)|D] = \int g(\theta)P(\theta|D) d\theta \approx \frac{1}{m} \sum_{i=1}^{m} g(\theta_i)$

* Posterior simulation - MCMC

* How? - Generate a correlated sequence (chain) of random variates (monte carlo) that (in a limit) are draws from the posterior. The next value in the sequence only depends on the current values (Markov)
> (explain how later)

* Algorithm cleverly constructed to ensure distribution of chain values -> posterior dist. = stationary dist. in the long run.

---
(Page 82)

> special case of Metropolis-Hastings

### Simplest MCMC: METROPOLIS ALGORITHM
> (slides)

1. Choose random starting point $\mu_0$
> any old way

2. At each step $i=1, ..., N_{MC}$ propose a new parameter value $\mu_{prop}$ via $J(\mu)$. The proposal distribution $J(\mu)$ is usually $N(\mu_i, \tau^2)$
> draws from proposal dist. say jump.
> jumping density noise

The proposal scale $\tau$ is chosen for efficiency!
> cleverly!

3. Evaluate ratio of posteriors at proposed vs. current values
Metropolis Ratio $r = \frac{P(\mu_{prop}|y)}{P(\mu_i|y)}$

4. If $\mu_{prop}$ is a better solution (higher posterior), $r>1$, accept the new value
$\mu_{i+1} = \mu_{prop}$

Else accept with probability $r$ i.e. accept with probability $\min(r,1)$. Stay at same value
$\mu_{i+1} = \mu_i$
(and include in chain - keep repeated values)

5. Repeat 2-4 until reach some measure of convergence and gather enough samples to compute your inference.

> Metropolis: next value depends on current value
> Monte Carlo: randomness

---
(Page 83)

> returns to example of analytic solution

### Example: simple gaussian unknown $\mu$
> (slides)

Likelihood: $y_i \sim N(\mu, \sigma^2=1) \quad i=1,...,N$
Prior: $p(\mu) \propto 1$
=> Posterior: $P(\mu|y) = N(\mu|\bar{y}, \sigma^2/N)$
> (analytic solution)
> (L12)

Do some MCMC steps (proposal scale 0.6)
> code on moodle to try yourself, check if you get summary statistics agreeing w/ the posterior
1000 iterations metropolis algo.:

[Two plots side by side.
Left plot has two subplots. Top subplot is a trace plot of `μ` versus chain step (0 to 1000). The values fluctuate around a mean. It is labeled "Metropolis trace plot" and "acceptance ratio = 0.50". Bottom subplot is a trace plot of `log p(μ|y)` vs chain step, showing similar fluctuations.
Right plot shows a blue histogram labeled "posterior histogram" and a solid black curve labeled "analytic posterior". They are very closely matched. The x-axis is `μ` and the y-axis is "density". It is labeled `Nmc = 1000`.]

> During burn-in e.g. past 50 iterations.
> acceptance ratio 0.47
> accept rate goes up if we make small steps. We want to be exploring
> r ratio goes up by itself

Can also cut off 1st half of chain to cut burn in (aggressive solution, more empirical cut-offs exist).

[Two plots mirroring the ones above, but with the x-axis only going to 500, representing the chain after burn-in is discarded. The trace plot for `μ` is shown, and the resulting posterior histogram is shown.]

> (usually if we have ID dist. probably wouldn't need to do MCMC, we'd just use CDF, invert & sample. Can just get away from that. but shown here is example.)

---
(Page 84)

> (print of sheet 2 on moodle)

## Lecture 17
_3.3.25_

{Today: continuing Bayesian computation: MCMC}

### Recap MCMC:
> ideally want iid but too long run. want to get enough indep samples that this works
> not iid (correlated) but looks like. Produce Marcov Name.

Generate a correlated sequence (chain) of random variates (Monte Carlo) that (in a limit) are draws from the posterior. The next value in the sequence only depends on current values (Markov).

### d-DIM METROPOLIS ALGORITHM:
> (slides)

Posterior $P(\theta|D)$ where $dim(\theta) = d$

Symmetric proposal/jump dist. $J(\theta^*|\theta) = J(\theta|\theta^*)$

1. Choose random starting point $\theta_0$

2. At step $i=1,...,N$ propose new parameter value
$\theta^* \sim N(\theta_{i-1}, \Sigma_p)$
> need to set $\Sigma_p$
> what to set $\Sigma_p$ to? could set it up to be symmetric, maybe positive definite matrix usually normal to ensure metropolis is symmetric proposal?

The proposal distribution is
$J(\theta^*|\theta_{i-1}) = N(\theta^*|\theta_{i-1}, \Sigma_p)$

3. Evaluate ratio of posteriors at proposed vs. current values
$r = \frac{P(\theta^*|y)}{P(\theta_{i-1}|y)}$

---
(Page 85)

4. Accept $\theta^*$ with probability
$\min(r,1): \theta_i = \theta^*$

If not accept, stay at same value $\theta_i = \theta_{i-1}$ for the next step and include in chain.

5. Repeat 2-4 until reach some measure of convergence and gather enough independent samples to compute your inference. (reduce Monte Carlo error)

---
(Page 86)

> widely used way to sample from MVG? (re: step 2 d-dim metropolis)

### Multivariate Gaussian Draws
> (board)

$\vec{\theta} \in \mathbb{R}^d$
Metropolis proposal $\theta_i^* \sim N(\vec{\theta}_{i-1}, \Sigma_p)$

$\Sigma_p$ = real, symmetric, positive definite $d \times d$ matrix
$det(\Sigma_p) > 0$ and $\Sigma_p^{-1}$ exists.

1. Cholesky decomposition. Find $L$ s.t. $\Sigma_p = L L^T$
> see in homework 2
$L$ is lower triangular Cholesky factor
"matrix square root"
[Diagram of a lower triangular matrix, with the lower triangle shaded blue.]

> Cholesky decomp. expensive so don't want to do Cholesky decomp at every point in your loop, so instead... do this:

2. Draw $\theta^* \sim N(\vec{\theta}_{i-1}, \Sigma_p)$
by $\theta^* = \vec{\theta}_{i-1} + L \vec{z}$
> (do this computationally w/ code)
where $\vec{z}$ is a d-vector of iid uniform unit normal r.v.s
$z_i \stackrel{iid}{\sim} N(0,1) \quad i=1,...,d$

> Check: why not sampling $\theta^*$ like this is indeed the same as sampling from MVG?
Check: $E[\vec{\theta}^*] = \vec{\theta}_{i-1}$
> wiki for $\theta_i+L\vec{z}$ has the desired dist. due to the affine transformation.
> any affine transformation $Y=A\vec{x}+b$ of a gaussian is also a gaussian
$Cov[\theta^*, \theta^{*T}] = Cov[L\vec{z}, (L\vec{z})^T]$ (bilinear)
> use to very summary
> 1. generate iid dist
> 2. find square root by transforming
> 3. find derived covariance matrix

$= Cov[L\vec{z}, \vec{z}^T L^T] = L Cov[\vec{z}, \vec{z}^T] L^T$
$= L \mathbb{I} L^T = L L^T = \Sigma_p$

---
(Page 87)

### Example:
> (slides)
Recall: analytic posterior density for Gaussian $(\mu, \sigma^2)$ model. With non-informative priors ($P(\mu)\propto1, P(\log \sigma^2)\propto1 \implies P(\sigma^2)\propto \sigma^{-2}, \sigma^2>0$).
(multi-param. inference)
> in the mean $\mu\in\mathbb{R}$
> variance $\sigma^2>0$

* Joint posterior: $(\sigma^2 > 0)$
$P(\mu, \sigma^2 | y) \propto (\sigma^2)^{-n/2-1} \exp\left(-\frac{(n-1)S^2}{2\sigma^2}\right) \exp\left(-\frac{n}{2\sigma^2}(\bar{y}-\mu)^2\right)$

* Marginal of $\mu$:
$P(\mu|y) = \int P(\mu, \sigma^2|y) d\sigma^2 \propto \left[1+\frac{n(\mu-\bar{y})^2}{(n-1)S^2}\right]^{-n/2}$
$= t_{n-1}(\mu | \bar{y}, S^2/n)$
[Graph showing a t-distribution for P(μ|y) vs μ.]

* Marginal of $\sigma^2$:
$P(\sigma^2|y) = \int P(\mu, \sigma^2|y) d\mu = \text{Inv-}\chi^2(\sigma^2|n-1, S^2)$
[Graph showing an inverse-chi-squared distribution for P(σ²|y) vs σ².]

* Joint posterior:
$P(\mu, \sigma^2|y) = P(\mu|\sigma^2,y) P(\sigma^2|y) = N(\mu|\bar{y}, \sigma^2/n) \times \text{Inv-}\chi^2(\sigma^2|n-1,S^2)$
> factorisation from Bayes' rule

[Diagram of the 2D joint posterior P(μ,σ²|y) showing concentric, elongated contours in the μ-σ² plane.]

Metropolis 2D example
[Two marginal posterior plots, one for P(μ|y) and one for P(σ²|y), each showing a blue histogram from the MCMC samples overlaid with a black analytic curve. A 2D scatter plot shows the MCMC chain samples in the μ-σ² plane, concentrated in the high-probability region. Arrow points from 2D plot to marginal plots.]
> MC chain Nmc=2000, acc. ratio = 0.39

> Reminder: $\vec{x} \sim N(\vec{\mu}, \Sigma)$, then $Cov[x_i,x_j] = \Sigma_{ij}$. Some props to denote this covariance.
> e.g. $\Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{21} & \sigma_{22} \end{pmatrix} = \begin{pmatrix} Var(x_1) & Cov(x_1,x_2) \\ Cov(x_2,x_1) & Var(x_2) \end{pmatrix}$, here $\Sigma_1 = \text{Cov}(x_1,x_1)$
Think in components!
$Cov[\theta_i^*, \theta_j^*] = Cov[\theta_{i-1, i} + L_{ik}z_k, \theta_{i-1,j} + L_{jl}z_l] = L_{ik}L_{jl}Cov[z_k,z_l] = L_{ik}L_{jl}\delta_{kl}$
> $Cov[x,y] = Cov[x,y]$
> $Cov[x,y] = \delta Cov(N,N)$
$= L_{ik}L_{jk}$

---
(Page 88)

> (could also do this case for Metropolis - 'random walk' MCMC) + side note
> combine this w/ proposal dist to updating new positions

### METROPOLIS-HASTINGS ALGORITHM
> (slides)

More General Jumping rule $J(\theta_a | \theta_i)$
(Need not be symmetric $J(\theta_a | \theta_b) \neq J(\theta_b | \theta_a)$)

1. Choose random starting point $\theta_0$

2. At step $i=1,...,N$ propose a new parameter value
$\theta^* \sim J(\theta^* | \theta_{i-1})$

3. Evaluate M-H ratio of posteriors at proposed vs current values
$r = \frac{P(\theta^*|y)/J(\theta^*|\theta_{i-1})}{P(\theta_{i-1}|y)/J(\theta_{i-1}|\theta^*)}$

4. Accept $\theta^*$ with probability
$\min(r,1): \theta_i = \theta^*$

If not accept, stay at same value $\theta_i = \theta_{i-1}$ and include in chain.

5. Repeat 2-4 until reach some measure of convergence and gather enough samples to compute your inference.

> METROPOLIS-HASTINGS THEORY: See Wikipedia
> A Markov process (chain) to reach stationary distribution (i.e. the posterior)
> The chain must be ergodic for its stationary distribution to exist and be unique. (detailed balance is a sufficient condition, but not necessary)
> Ergodicity is given by being aperiodic & positive recurrent. Process of going from sample between states as chain progresses is s.t. that regardless of starting state, it is guaranteed that the probability of any particular state being visited again other than the first is guaranteed and that the irreducibility of the markov process (probability of moving between any state to any state is >0). These are formalisms that any sample to the same dist. w/any starting point (if repeats convergence guaranteed due to this).
> Design a markov process (by constructing transition probability) that fulfills these 2 conditions and so converges. this is how metropolis hastings algo. designed.

---
(Page 89)

* d-dim Metropolis is just a special case, where
$J(\theta^*|\theta_i) = N(\theta^*|\theta_i, \Sigma_p) = N(\theta_i|\theta^*, \Sigma_p) = J(\theta_i|\theta^*)$
is a symmetric proposal dist.

* More general asymmetric proposals, allow "biased" proposals -> more probable to propose towards a certain direction.

* With some knowledge of structure of the posterior, can sometimes engineer a clever proposal $J(\theta_i|\theta_j)$.
> e.g.?

### GIBBS SAMPLING
> when you can't sample from $P(\theta|D)$ directly but can sample from $P(\theta_i|D)$, $P(\theta_j|D)$ directly. (conditionals)

* Special case of Metropolis-Hastings (Gelman BDA ch 11)

* Multi-dim sampling when you can utilize the set of conditional posterior distributions as proposal distns. for each parameter - Metropolis Hastings ratio = 1 (always accept).

* If joint posterior is $P(\theta, \phi|D)$
And you can solve for tractable (you can draw) conditionals:
$P(\theta|\phi, D)$
$P(\phi|\theta, D)$

* Jump along each parameter-dimension one at a time.
> derive $p(\theta_i|\theta_{-i})$ ...
> (1) draw $\theta_i$ from $p(\theta_i|\theta_{-i})$ Then can starting dist satisfy... To construct the chain, need probability $\pi(\theta'|\theta)$. Suppose $p(\theta'|\theta) = \int p(\theta'|\theta,\theta_1) p(\theta_1|\theta) d\theta_1, p(\theta'|\theta_1) = p(\theta'|\theta_1)$.
> (2) so, $A(\theta_1, \theta') = \frac{\pi(\theta')J(\theta_1|\theta')}{\pi(\theta_1)J(\theta'|\theta_1)}$ prob. accept.
> (3) choose accept/reject such that this condition is met. M-H gives us $A(\theta_1,\theta')=\min(1, \frac{P(\theta'|D)J(\theta_1|\theta')}{P(\theta_1|D)J(\theta'|\theta_1)})$

---
(Page 90)

### 2-dim GIBBS SAMPLER
> (slides)

> conditional requires -> always accept

1. Choose random starting point $(\theta_1^0, \theta_2^0)$

2. At cycle $t$, update
$\theta_1^t \sim P(\theta_1 | \theta_2^{t-1}, D)$
> first entry depends on this

3. Then update
$\theta_2^t \sim P(\theta_2 | \theta_1^t, D)$
(each complete set (pair) of updates is called a Gibbs cycle)

4. Record current values of chain $(\theta_1^t, \theta_2^t)$

5. Repeat 2-4 until reach some measure of convergence (G-R) and gather enough indep. samples to compute your inference! reduce Monte Carlo error!

---
(Page 91)

### d-DIM GIBBS SAMPLER

Parameter vector $\theta = (\theta_1, ..., \theta_d)$

Current state at j-th update within cycle t:
$(\theta_1^{t}, ..., \theta_j^{t-1}, ..., \theta_d^{t-1})$
> value of component 1 at iteration t
> value of all components except j at previous time cycle t (or earlier j updates)
$\theta_{-j}^t \equiv (\theta_1^t, ..., \theta_{j-1}^t, \theta_{j+1}^{t-1}, ..., \theta_d^{t-1})$
> d-1 dim vector

1. iteration t=1 value... components already updated(?). Position j (previous) using updated(?) components.
Choose random starting point $\theta_0$
> no? only uses $\theta_0$

2. At cycle t, update through the d-parameters:
For each $j=1,...,d$ move the jth parameter to
$\theta_j^t \sim P(\theta_j | \theta_{-j}^t, D)$
(update $\theta_j$ conditional on current values of all other parameters)

3. After updating all d parameters, record current state $\theta^{t+1}$

4. Repeat 2-3 until reach convergence and enough samples

> (e.g. d=3) start $\theta^0 = (\theta_1^0, \theta_2^0, \theta_3^0)$
> $\theta_1^1 \sim P(\theta_1|\theta_2^0, \theta_3^0)$ } cycle
> $\theta_2^1 \sim P(\theta_2|\theta_1^1, \theta_3^0)$ } t=1
> $\theta_3^1 \sim P(\theta_3|\theta_1^1, \theta_2^1)$ }
> etc.

Gelman slightly diff notation:
$\theta_{-i}^t = (\theta_1^t, ..., \theta_{i-1}^t, \theta_{i+1}^{t-1}, ..., \theta_k^{t-1})$
propose from: $\theta_i^* \sim P(\theta_i | \theta_{-i}^{t-1})$

---
(Page 92)

### Gibbs Sampling: Example
> (Gelman BDA section 11.1)
> (slides)
> Consider single observation y=(y1, y2)

Likelihood: $y = \begin{pmatrix} y_1 \\ y_2 \end{pmatrix} \sim N \left( \begin{pmatrix} \theta_1 \\ \theta_2 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix} \right)$
> ($\rho$ known)

Priors: $P(\theta_1) = P(\theta_2) \propto 1$

Posterior: $\begin{pmatrix} \theta_1 \\ \theta_2 \end{pmatrix} | y \sim N \left( \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix} \right)$
$P(\theta|y) = P(\theta_1, \theta_2|y) P(\theta_2|y) = P(\theta_1|\theta_2, y) P(\theta|y)$
> simple to derive another formulation by completing the square or demonstrating Gibbs sampling from bivariate.
> Bivariate gaussian joint can be decomposed into conditionals.

Conditional Posteriors (e.g. properties of MVG's): see moodle
> 1D gaussians as proposals
$\theta_1 | \theta_2, y \sim N(y_1 + \rho(\theta_2-y_2), 1-\rho^2)$
$\theta_2 | \theta_1, y \sim N(y_2 + \rho(\theta_1-y_1), 1-\rho^2)$

[Diagrams illustrating Gibbs sampling for a 2D correlated Gaussian posterior.
On the left, a contour plot of a 2D posterior `P(θ₁,θ₂|y)` with elliptical contours, indicating correlation.
An arrow points to a trace plot on the right labeled "Gibbs sampling trace path: P(θ₁,θ₂|y)". This plot shows the sampler moving in axis-aligned steps (right angles), exploring the posterior distribution.
Another arrow points down from the trace plot to two marginal posterior density plots, one for `P(θ₁|y)` and one for `P(θ₂|y)`. Each shows a histogram created from the samples, approximating the true marginal distribution.]
> Remark: FLAW
> if $\rho \approx 1$ highly correlated, the gibbs sampler projects perpendicular to this, so only makes small jumps. conditional variance is but dist is diagonal so needs small steps to explore the space.
> gibbs will also be a problem if metropolis hastings is slow, but especially a problem here!!

---
(Page 93)

> ex sheet 3
> read w/sampling tut on moodle

## Lecture 18
_5.3.25_

{
Today: Continuing Bayesian Computation:
Metropolis-Hastings, Gibbs Sampling, applications to SN cosmology, Assessing Convergence and Mixing.
}

### GIBBS SAMPLING STEP AS A SPECIAL CASE OF METROPOLIS-HASTINGS
> (Gelman 11.3) (board)

> Gibbs is special case of metropolis hastings that always accepts.
> show this?

At the jth parameter update within iteration t
(target $\pi(\vec{\theta}) = P(\vec{\theta} | \vec{D})$)
> p(θ) on my notes elsewhere

M-H ratio, $r = \frac{\pi(\vec{\theta}^*) / J(\vec{\theta}^* | \vec{\theta}^{t-1})}{\pi(\vec{\theta}^{t-1}) / J(\vec{\theta}^{t-1} | \vec{\theta}^*)}$

Recall $\theta_{-j}^{t-1} = (\theta_1^t, ..., \theta_{j-1}^t, \theta_{j+1}^{t-1}, ..., \theta_d^{t-1})$
> current components of $\theta$ except $\theta_j$
have updated up to component j-1.

In Gibbs, we choose
> can only move to new proposals where only one coord is updating, rest stay the same
> only move in jth dim. All other coords are fixed.
$J(\vec{\theta}^* | \vec{\theta}^{t-1}) = \begin{cases} \pi(\theta_j^* | \vec{\theta}_{-j}^{t-1}) & \text{if } \vec{\theta}_{-j}^* = \vec{\theta}_{-j}^{t-1} \\ 0 & \text{otherwise} \end{cases}$

$r = \frac{\pi(\vec{\theta}^*) / \pi(\theta_j^* | \vec{\theta}_{-j}^{t-1})}{\pi(\vec{\theta}^{t-1}) / \pi(\theta_j^{t-1} | \theta_{-j}^{t-1})}$
> $\vec{\theta}_{-j}^* = \vec{\theta}_{-j}^{t-1}$
> use this defn. to expand terms into components.

$= \frac{\pi(\theta_j^* | \theta_{-j}^{t-1})\pi(\theta_{-j}^{t-1}) / \pi(\theta_j^* | \theta_{-j}^{t-1})}{\pi(\theta_j^{t-1} | \theta_{-j}^{t-1})\pi(\theta_{-j}^{t-1}) / \pi(\theta_j^{t-1} | \theta_{-j}^{t-1})} = \frac{\pi(\theta_{-j}^{t-1})}{\pi(\theta_{-j}^{t-1})} = 1$
> ALWAYS ACCEPT

> saves on computation -> always accept, don't need to calc ratio

---
(Page 94)

### Metropolis within Gibbs
> (slides)
> Gibbs is golden. Forums suggest multiple problems or params in problem, but sometimes you can't solve for conditionals for some parameters -> use that substep w/ metropolis rule.

$\theta = (\theta_1, ..., \theta_d)$
$\theta_{-j} = (\theta_1, ..., \theta_{j-1}, \theta_{j+1}, ..., \theta_d)$

* When you can't solve for tractable conditional distributions for all $\theta_j$:
$P(\theta_j | \theta_{-j}, D)$

* Replace each substep for updating each jth parameter $\theta_j$ with a separate Metropolis rule, compute Metropolis ratio and accept/reject.

* Cycle through all parameters and repeat for all N MCMC steps
> multiple proposal/rejects within full set of parameters

PREVIOUS PAGE
Gibbs as special case of M-H:
MH-R: $\frac{P(\theta^*)J(\theta^{t-1}|\theta^*)}{P(\theta^{t-1})J(\theta^*|\theta^{t-1})}$
> do I need to re-write this?
> $\theta_{-j}^*$ is current component of $\theta_{-j}^*$
> (for Gibbs)
Propose proposal distribution $J(\theta^*|\theta^{t-1}) = \begin{cases} P(\theta_j^*|\theta_{-j}^{t-1}) & \text{if } \theta_{-j}^* = \theta_{-j}^{t-1} \\ 0 & \text{otherwise} \end{cases}$
> all components stay the same as previous.

> i.e. only propose new points where all components are the same except $\theta_j^*$ where $\theta_j^*$ is picked according to $P(\theta_j^*|\theta_{-j}^{t-1})$
$r = \frac{P(\theta_j^*, \theta_{-j}^{t-1})}{P(\theta_j^{t-1}, \theta_{-j}^{t-1})} \frac{P(\theta_j^{t-1}|\theta_{-j}^{t-1})}{P(\theta_j^*|\theta_{-j}^{t-1})}$
> expands conditionals
$r = \frac{P(\theta_j^*|\theta_{-j}^{t-1})P(\theta_{-j}^{t-1})}{P(\theta_j^{t-1}|\theta_{-j}^{t-1})P(\theta_{-j}^{t-1})} \frac{P(\theta_j^{t-1}|\theta_{-j}^{t-1})}{P(\theta_j^*|\theta_{-j}^{t-1})} = 1$
> (all cancel out, so $r=1$)
> ALWAYS ACCEPT
> exactly we propose from the distribution we need to be sampling from, through gibbs.
> so could have just defined that way as well instead of integrate that step

---
(Page 95)

### d-dim Metropolis-within-Gibbs Sampler
> (slides)

$\theta = (\theta_1, ..., \theta_d)$
$\theta_j^t = (\theta_1^{t+1}, ..., \theta_{j-1}^{t+1}, \theta_j^t, ..., \theta_d^t)$

1. Choose a random starting point $\theta_0$

2. At cycle $t=1,...,N$, cycle through the d-parameters:
A. For each $j=1,...,d$, propose a new jth parameter value from a 1-dimensional Gaussian
$\theta_j^* \sim N(\theta_j^t, \tau_j^2)$
> proposal scale $\tau^2$

B. Evaluate ratio of posteriors at proposed vs current values:
$r = P(\theta_j^*, \theta_{-j}^t | D) / P(\theta_j^t, \theta_{-j}^t | D)$
$= P(\theta_j^* | \theta_{-j}^t, D) / P(\theta_j^t | \theta_{-j}^t, D)$
> (new point)
> (old point)
> note: only one component is changing. other are fixed.

C. Accept $\theta_j^{t+1} = \theta_j^*$ with prob $\min(r,1)$, otherwise $\theta_j^{t+1} = \theta_j^t$.

3. After full cycle, record current values $\theta^{t+1}$

4. Repeat steps 2 for all parameters until convergence and enough samples

> Note:
> * adjust $\tau_j^2$:
> * empirically aim for 30%-40% acceptance rate
> * if small step, will accept very fast but not explore param space, too large, never accept, never move anywhere

[Diagrams showing a trace plot and two marginal posterior histograms. The trace plot shows samples in a 2D parameter space. The histograms show the marginal distributions `p(θ₁|y)` and `p(θ₂|y)`, approximating the true posteriors.]

---
(Page 96)

> advantages: always accept
> gibbs: > must use `my` conditional derived from posterior give algorithm more info about which direction is likely to have more probability
> disadvantages: -> requires analytic properties; but if we can't derive them, can still use metropolis within Gibbs.

* ## Mixed Gibbs Sampler:

-> Can replace sampling from conditionals with accept/reject for just the parameters with intractable conditionals.

> in cycle t
> for each j:
> if we do have analytic conditional,
> just propose, normal from that & always accept
> if not,
> propose new point from proposal dist. accept w/ prob min(r,1). (i.e. do metropolis)
> $r = \frac{p(\theta_j^*, \theta_{-j}^t)}{p(\theta_j^t, \theta_{-j}^t)}$

---
(Page 97)

### Tuning d-dim Metropolis
> this is proposal scale

* $\theta^* \sim N(\theta; \Sigma_p)$: If proposal scale $\Sigma_p$ is too large, will get too many rejections and not go anywhere. If proposal scale too small, you will accept very many small moves: inefficient random walk.

* ## Laplace Approximation
$P(\theta|D) \approx N(\theta | \theta_{MAP}, A^{-1})$
$\theta_{MAP}$ = posterior mode
$A_{ij} = - \left. \frac{\partial^2}{\partial\theta_i \partial\theta_j} \ln P(\theta|D) \right|_{\theta = \theta_{MAP}}$
> approx posterior cov matrix $A^{-1}$

* Choose $\Sigma_p = c^2 A^{-1}$, $c \approx 2.4/\sqrt{\dim(\theta)}$

* Scale proposal to aim for acceptance ratio of 44% in 1D, 23% in dim > 5.

> note: will capture correlation if p. high
> Laplace Approx. gives proposal some correlation detection
> target highly correlated

---
(Page 98)

### LAPLACE APPROXIMATION
> new thing to begin w/
> (slides)

Unnormalised Posterior: $\hat{P}(\theta|D) = P(D|\theta)P(\theta)$
Find MAP estimate:
> 'maximum a posteriori'
> 'max. a posterior'
$\theta_{MAP} = \operatorname{argmax}_{\theta} \ln \hat{P}(\theta|D)$
> same as max likelihood but for posterior
> $P(\theta|D)$ with non-informative priors

Taylor expansion (first deriv. is zero):
$\ln \hat{P}(\theta|D) \approx \ln \hat{P}(\theta_{MAP}|D) - \frac{1}{2} (\theta - \theta_{MAP})^T A (\theta - \theta_{MAP}) + ...$
$A_{ij} = - \left. \frac{\partial^2}{\partial\theta_i \partial\theta_j} \ln \hat{P}(\theta|D) \right|_{\theta = \theta_{MAP}}$

$\hat{P}(\theta|D) \approx \hat{P}(\theta_{MAP}) \times \exp\left(-\frac{1}{2} (\theta - \theta_{MAP})^T A (\theta - \theta_{MAP})\right)$

Gaussian Approximation:
> approx (sampling posterior)
$P(\theta|D) \approx N(\theta | \theta_{MAP}, A^{-1})$

$A^{-1}$ is an approximate posterior covariance matrix.

> reminder: multi-dim taylor exp:
> let $f(\theta) = \ln \hat{P}(\theta|D)$
> $f(\theta) = f(\theta_{MAP}) + \nabla f(\theta_{MAP})(\theta - \theta_{MAP}) + \frac{1}{2} (\theta - \theta_{MAP})^T H(\theta_{MAP})(\theta - \theta_{MAP}) + ...$
> $\nabla f(\theta_{MAP})=0$
> $H(\theta) = \nabla^2 f(\theta_{MAP})$. - $\frac{1}{2} (\theta - \theta_{MAP}) \tilde{A} (\theta - \theta_{MAP})$
> Hessian of $f$ (can write it as matrix of 2nd order derivatives) evaluated at $\theta$.
> A is -H eval at $\theta_{MAP}$

---
(Page 99)

> ex sheet 3

## Astrostatistics Case study: Supernova Cosmology

> measure brightness of SN

Cosmology: params control relationship between distance & redshift
dark energy EOS parametrized by w ($P=w\rho$)
modern cosmology -> $w=-1$?

type Ia SN almost standard candles
Measure their magnitude & plot vs redshift.
Curve change depending on how we treat.
params. $\Omega_m, \Omega_{\Lambda}$ -> so observed data constrains the parameters -> some assumptions hold vs. $z \gg 0$.

comoving distance is integral from zero to redshift.
distance probed by SN is luminosity distance.
get d_L by multiplying that by dimensions (of univ. MA?)
distance redshift also influenced by curvature of universe etc.

choose diff. params. plot theoretical dist. modulus vs. redshift.
diff. curve for diff. flatness? etc.

data: ...
[A small sketch of data points on a distance-redshift plot.]

want to fit data to curve to infer params??

[A sketch of a single line representing the distance-redshift relation.]

[A sketch of multiple curves for different parameter values on a distance modulus vs. redshift (z) plot.]

---
(Page 100)

> (slides) (lots of inconsistent notation here -> ignore it)

For us, simplify, let's say SNae are standard candles.

$M_s \sim N(M_0, \sigma^2_{int})$
> if distance modulus is...
## Population Distribution
$m_s = M_s + \mu(z_s; \vec{\Omega})$ (log) inv. sq. law
Assume data (apparent magnitudes and redshifts) $\{m_s, z_s\}$ are measured perfectly.
$\theta = (H_0, \Omega_m, \Omega_c, w)$
> (for consistency in notation change $\vec{\Omega}$)
## Cosmological Parameters

First assume w=-1 (cosmological constant)
Derive model, likelihood, posterior
$\vec{\Omega} = (\Omega_m, \Omega_c, w) \quad h = H_0 / (100 \text{km s}^{-1} \text{Mpc}^{-1})$
> use h to have distance modulus independent of H_0 value.
$\mu(z_s; H_0, \vec{\Omega}) = 25 + 5\log_{10}\left( \frac{c}{H_0} d_L(z_s; \vec{\Omega})\right)$
> Mpc
> luminosity distance of object defined relative to its observed redshift. The expression here was used to fit SN data (type Ia) to get accel. univ (1998) results, say "standard candle"
$M_s = m_s + \mu(z_s; H_0, \vec{\Omega})$
$P(m_s|z_s) = ...$
> $d_L = \frac{c}{H_0} d_c$
> (dimensionless)

$M_0$ and $log(h)$ cannot be separately constrained! (not separately identifiable)

> @paramehric?
> multimodality often a good sign, not getting stuck in param space.
[A small sketch showing a multimodal posterior, with the note "overlap: 'mixing'"]

> ex sheet 3
> have to use metropolis
> use laplace approx, find MAP to estimate to sample posterior