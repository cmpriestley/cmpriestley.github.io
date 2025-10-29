This is a complete transcription of the provided PDF document containing lecture notes on Astrostatistics.

***

# Astrostatistics
*modern statistical methods for astronomy*

## Lecture 2
_27.1.25_

[Image: A data point with error bars] data $(x_i, y_i)$

### ① Where did it come from?

Invert question - if you knew "true" values $(\eta_i, \xi_i)$, how do you generate the data?
$(\eta_i, \xi_i)$ are *latent variables*.
$x_i = \xi_i + \epsilon_{xi}$, $E_{xi} \sim N(0, \sigma_x^2)$

**Posit:**
$x_i | \xi_i \sim N(\xi_i, \sigma_{x_i}^2)$  
$y_i | \eta_i \sim N(\eta_i, \sigma_{y_i}^2)$
> **MEASUREMENT ERROR MODEL**

### ② Where do $\eta_i, \xi_i$ come from?

**@ $\eta_i$?**
*intrinsic scatter* $\epsilon_i \sim N(0, \sigma^2)$

**Posit:** $\eta_i = \alpha + \beta \xi_i + \epsilon_i$
> **linear model**

### ③ Where does $\xi_i$ come from?

**Posit:** $\xi_i \sim N(\mu, \tau^2)$
> "hyperparameters"
> **POPULATION DISTRIBUTION MODEL**

---
(Page 1)

[Image of a graph plotting y vs x. It shows a dashed line representing the linear model $y = \alpha + \beta \xi$. Around this line, there is a region showing the intrinsic scatter with width $\sigma$. A point $(\xi_i, \eta_i)$ is on this line of intrinsic scatter, not on the main linear model line. From this point $(\xi_i, \eta_i)$, a vertical line extends to show a data point $(x_i, y_i)$, illustrating the measurement error. Below the x-axis, a Gaussian curve is drawn, centered at $\mu$ with a width $\tau$, representing the population distribution of $\xi$.]

### GENERATIVE MODEL

$\xi \sim N(\mu, \tau^2)$ 
> Population Distribution

$\eta_i | \xi_i \sim N(\alpha + \beta \xi_i, \sigma^2)$
> Regression Model

$x_i | \xi_i \sim N(\xi_i, \sigma_{x_i}^2)$  
$y_i | \eta_i \sim N(\eta_i, \sigma_{y_i}^2)$
> Measurement Error

**Knowns and Unknowns:**

- $\Psi = (\mu, \tau)$
  > Population Distribution Independent Variable "Hyperparameters"
- $\theta = (\alpha, \beta, \sigma^2)$
  > Regression Parameters
- $(\xi_i, \eta_i)$
  > Latent (true) Variables ({(x,y) without measurement error})
- $(x_i, y_i)$
  > Observed Data
- $(\sigma_{x_i}, \sigma_{y_i})$
  > with measurement uncertainties

---
(Page 2)

## Lecture 3
_29.1.25_

### FUNDAMENTALS AND NOTATION

(F&B ch. 2-3, Ivezic ch. 3-5)
(measurement we take, take values)

Data are realisations of **random variables** (outcomes of probabilistic experiments/observations)

A **random variable** $X$ can take on values in some domain (measurable space)
e.g. (discrete) $\mathbb{Z}$, (continuous) $\mathbb{R}$, (multivariate/vector combinations) $\mathbb{R}^N$

**Discrete case:**

Define the probability distribution over possible outcomes with a **probability mass function** (PMF)
$$ \mathrm{Pr}(X=k) = P_X(k) $$
> prob r.v. takes some value k
e.g. $\mathrm{Pr}(0 \le X \le 2) = \sum_{k=0}^2 P_X(k)$
Normalisation: $\sum_{k=0}^\infty P_X(k) = 1$

**Continuous case:**
(normally don't work with discrete e.g. 10^6 photons -> continuous)
Define the **probability density function** (PDF)
$$ \mathrm{Pr}(x \le X \le x+dx) = P_X(x)dx $$
e.g. $\mathrm{Pr}(0 \le X \le 2) = \int_0^2 P_X(x)dx$
Normalisation: $\int_{-\infty}^\infty P_X(x)dx = 1$

---
(Page 3)

and the **cumulative distribution function** (CDF)
$$ \mathrm{Pr}(X \le x_0) = \int_{-\infty}^{x_0} P_X(x)dx $$

**Multivariate continuous case:**
$\vec{x} \in \mathbb{R}^N$, $S \subset \mathbb{R}^N$
> member of, subset
$$ \mathrm{Pr}(\vec{x} \in S) = \int_S P_{\vec{x}}(\vec{x}) d^N\vec{x} $$
> volume element in $\mathbb{R}^N$ (small set values RV X takes values in)

Normalisation: $\int_{\mathbb{R}^N} P_{\vec{x}}(\vec{x}) d\vec{x} = 1$

When there is no ambiguity, we may simplify notation,
i.e. $P_X(x) \rightarrow P(x)$.
> random variable

Given a distribution $P(x)$ for RV $x$,
$X \sim P(x)$ means "distributed as", "is drawn from", "draw $X$ from $P(x)$" depending on context
> e.g. in algorithm means this not "approximately" sign

Astronomers' measurements are produced by physical processes that are inherently probabilistic (truly random processes thanks to **quantum mechanics**).

---
(Page 4)

### Example 1: Photometry / Photon counting

[Image of a star emitting photons towards a detector]
> Chandra X-ray observatory

rate = photons arriving / time = $r$

$ \mathrm{Pr}(k \text{ photons in time } T) = \frac{e^{-\lambda}\lambda^k}{k!} $ where $\lambda = rT$
$ P(k) = \text{Poisson}(k|\lambda) $
> (also written) $k \sim \text{Poisson}(\lambda)$

for $k \rightarrow \text{large}$, $P(k) \approx N(k|\lambda, \lambda)$
> Gaussian/Normal
> a limit theorem

Define **Gaussian Random Variable**
$X \sim N(\mu, \sigma^2)$, $x \in \mathbb{R}$
PDF: $P(x) = N(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}(x-\mu)^2/\sigma^2}$

Generalise to **Multivariate Gaussian Vector**
$\vec{v} \in \mathbb{R}^N$ (e.g. $N=2, \vec{v} = \begin{pmatrix} x \\ y \end{pmatrix}$)
PDF: $P(\vec{v}) = N(\vec{v}|\vec{\mu}, \Sigma) = |2\pi\Sigma|^{-\frac{1}{2}} e^{-\frac{1}{2}(\vec{v}-\vec{\mu})^T \Sigma^{-1} (\vec{v}-\vec{\mu})}$
$\vec{\mu} \in \mathbb{R}^N$, $\Sigma$ is NxN symmetric pos. def. matrix.

---
(Page 5)

### Example: Astrometry

[Image illustrating astrometry: a star's light passing through the atmosphere (with turbulence) to a detector. This creates a Point Spread Function (PSF) on the detector, which is a 2D distribution. This 2D distribution is shown as a circular contour plot and its marginal distributions p(y) and p(x) are shown as 1D Gaussians along the axes.]
POINT SPREAD FN. (PSF)
> (JOINT)
> marginals

Joint distribution $P(x,y) = N\left( \begin{pmatrix} x \\ y \end{pmatrix} \middle| \begin{pmatrix} \mu_x \\ \mu_y \end{pmatrix}, \begin{pmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2 \end{pmatrix} \right)$
(simplifying $\sigma_x^2 = \sigma_y^2$)
$= N(x|\mu_x, \sigma_x^2) N(y|\mu_y, \sigma_y^2)$

Law of total probability (LTP):
$P(x) = \int_{-\infty}^\infty P(x,y)dy = N(x|\mu_x, \sigma_x^2)$
$P(y) = \int_{-\infty}^\infty P(x,y)dx = N(y|\mu_y, \sigma_y^2)$

In this case, $P(x,y) = P(x)P(y)$, x,y indep. RVs.

In general, this is not true:
$P(x,y) = N\left( \begin{pmatrix} x \\ y \end{pmatrix} \middle| \begin{pmatrix} \mu_x \\ \mu_y \end{pmatrix}, \begin{pmatrix} \sigma_x^2 & \rho\sigma_x\sigma_y \\ \rho\sigma_x\sigma_y & \sigma_y^2 \end{pmatrix} \right)$ $\rho > 0$
> (positive slant)

[Image of a correlated 2D Gaussian distribution, shown as an elliptical contour, with its marginal distributions P(x) and P(y) along the axes.]
$P(x,y) \ne P(x)P(y)$
> marginals same as above

---
(Page 6)

### Conditional Probability
> lower dim distributions obtained by integrating out other variables
> marginals

$P(x,y) = P(x|y)P(y) = P(y|x)P(x)$
> joint -> conditionals

$$ P(y|x) = \frac{P(x,y)}{P(x)} = \frac{P(x|y)P(y)}{P(x)} $$
> **BAYES' THEOREM**

Note that LTP: $P(x) = \int P(x,y)dy = \int P(x|y)P(y)dy$
$P(y|x) = \frac{P(x|y)P(y)}{\int P(x|y)P(y)dy}$

In the bivariate case: Gaussian
$P(y|x) = N\left(y | \mu_y + \rho\frac{\sigma_y}{\sigma_x}(x-\mu_x), \sigma_y^2(1-\rho^2)\right)$
> if I know x, I can predict y
> If it is bivariate gaussian
> original variance in y decreases by factor (1-p^2)
> variance of prediction error is smaller

[Image showing a bivariate Gaussian distribution with its marginals. A vertical slice at $x=x_0$ is taken, and the resulting conditional distribution $P(y|x=x_0)$ is shown. This conditional distribution is a narrower Gaussian than the marginal $P(y)$. The means of these conditional distributions form the regression line.]

Observations $x_0$ gives you information to predict y
$p(y|x_0) \ne p(y)$, $\rho>0$.

---
(Page 7)

## Lecture 4
_31.1.25_

### MOMENTS & SUMMARIES OF PROB. DISTRIBUTIONS
$X \sim P(x)$ -> X is RV drawn from P(x)

**Expect values**
$$ E[X] = \int x P(x) dx $$
$$ E[g(x)] = \int g(x) P(x) dx $$

**Variance**
$$ \mathrm{Var}(X) = E[(X - E(X))^2] $$
$$ = E[X^2] - (E[X])^2 $$

**Multivariate (x,y):**

**Covariance**
$$ \mathrm{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] $$
$$ (=E[XY] - E[X]E[Y]?) $$

**Multivariate Normal:**
$$ \begin{pmatrix} X \\ Y \end{pmatrix} \sim N\left( \begin{pmatrix} \mu_x \\ \mu_y \end{pmatrix}, \begin{pmatrix} \sigma_x^2 & \sigma_x\sigma_y\rho \\ \sigma_x\sigma_y\rho & \sigma_y^2 \end{pmatrix} \right) \quad -1 < \rho < 1 $$
> PROPERTIES: `Cov(X,X)=Var(X)`, `Cov(aX,bY)=abCov(X,Y)`, etc.

\* $X \perp Y \rightarrow \mathrm{Cov}(X,Y) = 0$
\* general case $\mathrm{Cov}(X,Y) = \sigma_x\sigma_y\rho$
> `Cov(x,y)=p` is for standard normals (σ=1).

**Useful Properties:**
Collection of random variables
$\{X_1, ..., X_N\}, \{Y_1, ..., Y_N\}$
> set of x & y measurements labelled 1 to N

---
(Page 8)

Def. $S = \sum_{i=1}^N a_i X_i$, $T = \sum_{j=1}^M b_j Y_j$, $a_i, b_j \in \mathbb{R}$

**Expectations are linear** $\rightarrow E[S] = \sum a_i E[X_i]$
> expectation of sum = sum of expectation (linearity of integral)

**Bilinearity of covariance:**
$ \mathrm{Cov}(S,T) = \mathrm{Cov}(\sum_{i=1}^N a_i X_i, \sum_{j=1}^M b_j Y_j) = \sum_i \sum_j a_i b_j \mathrm{Cov}(X_i, Y_j) $
$ \mathrm{Var}(S) = \mathrm{Cov}(S,S) = \sum_{i=1}^N a_i^2 \mathrm{Var}(X_i) + \sum_{i \ne j} a_i a_j \mathrm{Cov}(X_i, X_j) $
... useful for summing multiple sources of error.
> variance of a sum of RV is covariance of sum with itself.
> `just two terms` `some errors correlated` `not correlated`
> IMPORTANT! Variance of sum is sum of covariances!
> `Var(Σ X_i) = ΣΣ Cov(X_i, X_j) = Σ Cov(X_i, X_j) + Σ Var(X_i)`

### TRANSFORMATIONS OF RVs
$X \sim P(x)$
$Y = \phi(x)$ (invertible & differentiable function)
what is $P(y)$?

$x = \phi^{-1}(y)$
$p(y) = P(x) \left| \frac{dx}{dy} \right| = P(\phi^{-1}(y)) \left| \frac{d\phi^{-1}(y)}{dy} \right|$
> Jacobian

**Example:**
$X \sim U(0,1)$
$Y = - \ln X \rightarrow P(y) = \begin{cases} e^{-y}, & 0 \le y < \infty \\ 0, & \text{o/w} \end{cases}$
$\frac{dx}{dy} = -e^{-y}$
> $y = \text{Expon}(1)$

[Image of the PDF for a Uniform(0,1) distribution and the resulting Exponential(1) distribution for Y.]

---
(Page 9)

### Propagation of error
> can think of this as transformation of RV but with variances

Measurement $x_0$ w/ error variance $\sigma_x^2$, error bar $\sigma_x$
[Image of a non-linear function $\phi(x)$. A narrow Gaussian $P(x)$ centered at $x_0$ with width $\sigma_x$ is shown on the x-axis. This is mapped by $\phi(x)$ to a skewed distribution $P(y)$ on the y-axis with width $\sigma_y$.]

$y = \phi(x) = \phi(x_0) + \left.\frac{d\phi}{dx}\right|_{x=x_0}(x-x_0) + ...$
$\mathrm{Var}(y) \approx \left| \frac{d\phi}{dx} \right|_{x=x_0}^2 E[(x-x_0)^2]$
$$ \sigma_y^2 \approx \left| \frac{d\phi}{dx} \right|_{x=x_0}^2 \sigma_x^2 $$

**Example:**
Astronomers' flux $f, \sigma_f$
magnitude $m = -2.5 \log_{10}(f) + \text{const.}$
$$ \sigma_m \approx \left| \frac{2.5}{\ln(10)} \right| \frac{\sigma_f}{f} $$
> `log_10(f) = ln(f)/ln(10)`
> `2.5/ln(10)` approx 1. Good enough.

... goodness of approximation depends on how linear transformation is 'in the neighborhood' where we have uncertainty.
It will break down if uncertainty large enough or trans. non-linear enough.

**MULTIVARIATE:** $Y = g(X_1, ..., X_d) \approx g(x^0) + \sum_i \frac{\partial g}{\partial x_i} (X_i - x_i^0)$ (linear taylor exp.)
$\rightarrow$ `use rule Var(aX+bY) = a^2Var(X) + b^2Var(Y) + 2abCov(X,Y)`
$ \mathrm{Var}(Y) \approx \sum_i \left(\frac{\partial g}{\partial x_i}\right)^2 \mathrm{Var}(X_i) + \sum_{i \ne j} \left(\frac{\partial g}{\partial x_i}\right)\left(\frac{\partial g}{\partial x_j}\right) \mathrm{Cov}(X_i, X_j) $
> intrinsic correlation

---
(Page 10)

**Multivariate case:**
$X_{obs} = (X_1, ..., X_N)$, $Y = g(X_1, ..., X_N)$
$$ \sigma_Y^2 \approx \sum_{i=1}^N \left.\left(\frac{\partial g}{\partial x_i}\right)^2\right|_{X_{obs}} \mathrm{Var}(X_i) + \sum_{i \ne j} \left.\left(\frac{\partial g}{\partial x_i}\right)\left(\frac{\partial g}{\partial x_j}\right)\right|_{X_{obs}} \mathrm{Cov}(X_i, X_j) $$
> important term, allows you to pick out correlated errors

### iid SEQUENCES OF RVs

**iid = independent & identically distributed**
(e.g. repeated measurement)
> why care? e.g. astronomers do multiple observations, often model them as iid.

Suppose we have a sequence of indexed RVs.
$X_1, X_2, X_3, ..., X_N$ iid $P(X)$
This means RVs are independent! (`cov(X_i, X_j) = 0`)
identically distrib. $X_i \sim P(X)$, $i \ne j \quad X_i \perp X_j$

**Joint distribution** $\sim P(X_1, X_2, ..., X_N) = \prod_{i=1}^N P(X_i)$

**Not example:**
$\begin{pmatrix} X_1 \\ X_2 \end{pmatrix} \sim N\left( \begin{pmatrix} \mu \\ \mu \end{pmatrix}, \begin{pmatrix} \sigma^2 & \sigma^2\rho \\ \sigma^2\rho & \sigma^2 \end{pmatrix} \right) \quad \rho \ne 0$
May $P(X_1) = \int P(X_1, X_2) dX_2 = N(X_1|\mu, \sigma^2)$, $P(X_2) = N(X_2|\mu, \sigma^2)$
$X_1, X_2$ identically distributed but not independent (or as last lecture. $P(X_2|X_1) \ne P(X_2)$).
$$ \mathrm{Cov}(X_1, X_2) = \rho \ne 0 $$
$$ \sqrt{\mathrm{Var}(X_1)}\sqrt{\mathrm{Var}(X_2)}} = \rho \ne 0 $$
> show not independent: `p(x1,x2) != p(x1)p(x2)` or showing `ρ != 0` (equivalently cov != 0)

---
(Page 11)

### Limit theorems for iid RVs
(see book for technicalities)

**① LAW OF LARGE NUMBERS (LLN)**
$X_1, ..., X_N$ iid $P(X)$
$\mu = E[X_i] = \int x P(x)dx$
> {theoretical model, population mean}
Def: Sample mean $\bar{X}_N = \frac{1}{N} \sum_{i=1}^N X_i$
as $N \rightarrow \text{large}$, $\bar{X}_N \rightarrow \mu$
> i.e. sample mean gives good approx. to population mean when N large.

**② CENTRAL LIMIT THEOREM (CLT)**
[Image: a non-Gaussian population distribution `P` evolving into a Gaussian sampling distribution for `X̄_N` as `N -> ∞`.]
$X_1, ..., X_N$ iid $P(X)$, $\sigma^2 = \mathrm{Var}(X_i)$ finite.
As $N \rightarrow \text{large}$, $\frac{\bar{X}_N - \mu}{\sigma/\sqrt{N}} \rightarrow N(0,1)$
> Unit gaussian
i.e. $\bar{X}_N \rightarrow N(\mu, \sigma^2/N)$, error in the mean = $\sigma/\sqrt{N}$
> note: this is a general result, we haven't assumed `P(x)` is gaussian, just that it has finite σ^2.

### STATISTICAL ESTIMATORS
data, RV
Probability model $P_{\theta}(X)$ or $P(X|\theta)$
> parameters
Suppose data are $D=(X_1, ..., X_N)$, $X_1, ..., X_N$ iid $P_{\theta}(X)$
> data are iid drawn from probability distribution.
Estimate $\theta$ from $D$ by constructing an **estimator** $\hat{\theta}(D)$ that yields a "good" value for the **estimand** $\theta$.
> thing you are estimating

---
(Page 12)

## Lecture 5
_3.2.25_

[Example context: CMB radiation from ~300,000 years after Big Bang. Fluctuations in plasma temp are map of fluctuations. A key property is the mean temperature. How might we find the mean of this distribution?]
> "point estimation"

### STATISTICAL ESTIMATORS CONT.

Construct a "good" estimator $\hat{\theta}(D) = \hat{\theta}(\vec{X})$ for estimand $\theta$.
> `θ̂`: estimate, `θ`: estimand, true value

Prob model $P_{\theta}(X)$ or $P(X|\theta)$
$D=(X_1, ..., X_N)$ iid $P(X|\theta)$.
$X_i \sim P_{\mu}(x)$, $\mu = E_P(x)$

[Image of a generic probability distribution with mean `μ` and standard deviation `σ`.]

e.g. (CMB Example) estimate $\mu$ from temp. measurement in each pixel $X_i, i=1, ..., N$
Estimand $\mu=E(X_i)$
Estimator $\hat{\mu}=f(\vec{X})$

**Possible Estimators:** (how do we choose?)
1. Sample mean $\frac{1}{N}\sum_i X_i$ > unbiased
2. Take $K < N$, $\frac{1}{K}\sum_{i=1}^K X_i$ > truncated sample mean, unbiased, inconsistent
3. $\frac{1}{N-1}\sum_i X_i$ > not unbiased but asymptotically unbiased
4. Midrange $\frac{1}{2}[\max(\vec{X}) + \min(\vec{X})]$
5. Median
6. Bin $\rightarrow$ Histogram $\rightarrow$ mode

---
(Page 13)

7. Just report 3 Kelvin

How do you evaluate different possible **estimators** for a particular **estimand**?
($\mu \rightarrow \hat{\mu}(\vec{X})$)

### Criteria for Estimators $\hat{\theta}$ for $\theta$

**UNBIASEDNESS**: $\hat{\theta}$ is **unbiased estimate** for $\theta$ if $E_P[\hat{\theta}] = \theta$
> definition e.g. `E_p`

bias: $b(\hat{\theta}) = E[\hat{\theta}] - \theta$
> expectation is over repeated experiments sampling the full data distribution $P(\vec{X}|\theta)$

Imagine you did J experiments with fixed sample size N, $j=1, ..., J$
$\rightarrow \vec{X}_j = (X_{1,j}, ..., X_{N,j})$ -> J datasets $\vec{X}_1, ..., \vec{X}_J$
$\hat{\theta}_1 = \hat{\theta}(\vec{X}_1), \hat{\theta}_2=\hat{\theta}(\vec{X}_2), ... \hat{\theta}_J=\hat{\theta}(\vec{X}_J)$
Unbiased as $J \rightarrow \infty$, $\frac{1}{J}\sum_{j=1}^J \hat{\theta}_j = E[\hat{\theta}] = \theta$

[Image of a number line with the `true θ` marked, and several estimates `θ̂_i` scattered around it, suggesting their average will land on `true θ`.]

---
(Page 14)

N.B. You only really did one experiment!

**ASYMPTOTICALLY UNBIASED**: $E[\hat{\theta}(\vec{X})] \rightarrow \theta$ as $N \rightarrow \infty$
(e.g. ③ on previous page)

**CONSISTENCY**: As you gather more data (sample size N $\rightarrow$ large), $\hat{\theta}$ converges to $\theta$.
$$ \forall \epsilon > 0, \mathrm{Pr}(|\hat{\theta}-\theta| > \epsilon) \rightarrow 0 \text{ as } N \rightarrow \infty $$
> e.g. ② not consistent

**EFFICIENCY**: Smallest **mean squared error**
$ \mathrm{MSE}(\hat{\theta}) = E[(\hat{\theta}(\vec{X}) - \theta)^2] $
(for unbiased $\rightarrow$ smallest variance)

$$ \mathrm{MSE}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}] + E[\hat{\theta}] - \theta)^2] = \mathrm{Var}(\hat{\theta}) + \mathrm{Bias}^2(\hat{\theta}) $$

**BIAS-VARIANCE TRADEOFF**: Sometimes the most **efficient** estimator is biased and the unbiased estimator is not the most **efficient**.

**MINIMUM VARIANCE UNBIASED ESTIMATORS (MVUE)**

**Caveat**: Unbiased estimators can be wrong (obviously wrong).

---
(Page 15)

**Ex.**
$X_i \stackrel{iid}{\sim} U(0, \theta)$, $i=1, ..., N$
Estimate $\theta$? Unbiased Estimator?

[Image of a uniform distribution P(x) from 0 to θ.]

Suppose we take sample mean:
$\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i$
$E[\bar{X}] = \theta/2$
$\hat{\theta}_u = 2\bar{X}$ unbiased estimator
since $E[\hat{\theta}_u] = 2(\theta/2) = \theta$
> `unbiased`

Suppose instead
$\hat{\theta}_m = \max(X_1, ..., X_N) < \theta$
> `always biased!`

generate samples $\rightarrow$ RNG $\vec{X} = (0.32, 0.46, 0.97, 2.77, 7.06)$
unbiased estimate $\hat{\theta}_u = 4.63$ (unbiased but clearly wrong)
biased estimate $\hat{\theta}_m = 7.06$ (biased but much more sensible)

**Unbiasedness** is not a property of any single experiment, only a property of averaging over all experiments that **did not happen**.

---
(Page 16)

**Ex.** Estimator properties depend on data process $p(X|\theta)$, $N=10$
(choose a, b s.t. $M=(a+b)/2$, $\mathrm{Var}(X) = (b-a)^2/12 = \sigma^2$)

| Estimator         | Gaussian $X_1, ..., X_N \sim N(\mu, \sigma^2=1)$ | Uniform $X_1, ..., X_N \sim U(a,b)$ |
| ----------------- | ------------------------------------- | ------------------------------------ |
| **Estimator Variance** | | |
| Sample Mean $\bar{X}$ | $\sigma^2/N = 0.10$                   | $\sigma^2/N = 0.10$                  |
| Midrange $\frac{\min(X)+\max(X)}{2}$ | $\frac{\pi^2\sigma^2}{24\ln N} = 0.18$ | $\frac{6\sigma^2}{(N+2)(N+1)} \approx \frac{6\sigma^2}{N^2} = 0.06$ |
|                   | midrange less efficient               | midrange more efficient              |

> both sample mean & midrange are unbiased, but midrange `V` goes down as `1/N^2` whereas `V_mean` goes down as `1/N`.

### LIKELIHOOD-BASED INFERENCE

Probability model $P(\vec{D}|\theta)$ we call "**sampling distribution**"
Probability distribution for possible/potential datasets $\vec{D}$, for a given parameter value $\theta$.

We observe $\vec{D}_{obs}: P(\vec{D}=\vec{D}_{obs}|\theta) = L(\theta)$
> **Likelihood Function**
> Sampling dist: probability dist. over all possible outcomes of D for a given θ. (before it is observed). N.B. $\int P(D|\theta)dD = 1$

---
(Page 17)

## Lecture 6
_5.2.25_

### LIKELIHOOD-BASED INFERENCE CONT.

Probability model $D \sim P(D|\theta)$, $\int P(D|\theta)dD = 1$
> **Sampling distribution**

[Image showing three different sampling distributions $P(D|\theta_1)$, $P(D|\theta_2)$, $P(D|\theta_3)$ as functions of D. A vertical line at $D_{obs}$ intersects these curves, providing the likelihood values for each $\theta$.]

**Likelihood**: $P(D=D_{obs}|\theta) = L(\theta)$
N.B. $\int L(\theta)d\theta$ not necessarily = 1

[Image showing the likelihood function $L(\theta)$ as a function of $\theta$. The values at $\theta_1, \theta_2, \theta_3$ correspond to the heights of the intersections in the previous graph. The maximum likelihood is at $\theta_2$, so $\hat{\theta}_{MLE} = \theta_2$.]

Suppose we now observe data $D_{obs}$ (outcome or realisation of D), the **likelihood** is the sampling dist. evaluated at $D=D_{obs}$, viewed as a fn. of the parameters. i.e. $L(\theta) = P(D=D_{obs}|\theta)$.

---
(Page 18)

**Notation**:
Often we will elide the distinction between $\vec{D}$ and $\vec{D}_{obs}$
$P(\vec{D}|\theta)$ understood to mean $P(\vec{D}=\vec{D}_{obs}|\theta)$.

**log likelihood** $l(\theta) = \ln L(\theta)$.

**iid case:**
$X_i \stackrel{iid}{\sim} f(X|\theta) \quad i=1, ..., N, \quad P(\vec{X}|\theta) = \prod_{i=1}^N f(x_i|\theta)$
$L(\theta) = P(\vec{X}|\theta) = \prod_{i=1}^N f(x_i|\theta)$ -> individual likelihood
$l(\theta) = \sum_{i=1}^N \ln f(x_i|\theta)$ -> log likelihood

### Fisher Information $\leftrightarrow$ Uncertainty

DEFINE:
$L(\theta) = P(x|\theta)$
**Score** $S = \frac{\partial}{\partial\theta}\ln L(\theta) = \frac{\partial}{\partial\theta}\ln P(x|\theta)$
> `p(x|θ)`
$E[S] = E[\frac{\partial}{\partial\theta}\ln P(x|\theta)] = \int [\frac{\partial}{\partial\theta}\ln P(x|\theta)] P(x|\theta) dx$
$= \int [\frac{1}{P(x|\theta)} \frac{\partial}{\partial\theta} P(x|\theta)] P(x|\theta) dx$
$* = \int \frac{\partial}{\partial\theta} P(x|\theta) dx = \frac{\partial}{\partial\theta} \int P(x|\theta)dx = \frac{\partial}{\partial\theta}[1] = 0$
\* Under regularity conditions $\frac{\partial}{\partial\theta} \leftrightarrow \int dx$

---
(Page 19)

**Fisher Information** = Variance of the **Score**
$I(\theta) = \mathrm{Var}(S) = E[S^2] - (E[S])^2$
$I(\theta) = E[(\frac{\partial}{\partial\theta}\ln P(x|\theta))^2] - (E[\frac{\partial}{\partial\theta}\ln P(x|\theta)])^2$
?$= -E[\frac{\partial^2}{\partial\theta^2}\ln P(x|\theta)]$

$E[\frac{\partial^2}{\partial\theta^2}\ln P(x|\theta)] = E[\frac{\partial}{\partial\theta}[\frac{1}{P(x|\theta)}\frac{\partial}{\partial\theta}P(x|\theta)]]$
$= E[-\frac{1}{P(x|\theta)^2}(\frac{\partial}{\partial\theta}P(x|\theta))^2 + \frac{1}{P(x|\theta)}\frac{\partial^2}{\partial\theta^2}P(x|\theta)]$
$= E[-(\frac{\partial}{\partial\theta}\ln P(x|\theta))^2 + \frac{1}{P(x|\theta)}\frac{\partial^2}{\partial\theta^2}P(x|\theta)]$
$\int [\frac{1}{P(x|\theta)}\frac{\partial^2}{\partial\theta^2}P(x|\theta)] P(x|\theta) dx = \frac{\partial^2}{\partial\theta^2} \int P(x|\theta) dx = \frac{\partial^2}{\partial\theta^2}(1) = 0$
$= -E[(\frac{\partial}{\partial\theta}\ln P(x|\theta))^2]$
$= -I(\theta)$

$$ I(\theta) = E[-\frac{\partial^2}{\partial\theta^2}(\ln P(x|\theta))] $$

> from info theory: amount of information r.v. X carries about an unknown parameter $\theta$ (upon which the probability of X depends)
> Formally: variance of score or repeated value of observed information.
> F.I. used to calculate covariance matrices associated with MLE.

---
(Page 20)

### Cramer-Rao Lower Bound
> "inverse of fisher info is a lower bound on the variance of any unbiased estimate"

Suppose we have estimator $T(x)$ for $\theta$,
$E[T] = \theta + b(\theta)$.

Consider $\mathrm{Cov}(S, T(X)) = E[S(X)T(X)] - E[S]E[T]$
> score
> $E[S]=0$ assuming unbiased.
$= E[T(X) \frac{1}{P(X|\theta)} \frac{\partial}{\partial\theta} P(X|\theta)]$
$= \int [T(X) \frac{1}{P(X|\theta)} \frac{\partial}{\partial\theta} P(X|\theta)] P(X|\theta) dx$
$* = \int T(X) \frac{\partial}{\partial\theta} P(X|\theta) dx = \frac{\partial}{\partial\theta} \int T(X) P(X|\theta) dx$
> * under regularity conditions
$= \frac{\partial}{\partial\theta} E[T(X)] = 1 + b'(\theta)$

$\mathrm{Cov}(S,T) = \sqrt{\mathrm{Var}(S)} \sqrt{\mathrm{Var}(T)} \mathrm{Corr}(S,T) \leftarrow -1 \le \mathrm{Corr} \le 1, |\mathrm{Corr}| \le 1$
$|\mathrm{Cov}(S,T)| \le \sqrt{\mathrm{Var}(S)}\sqrt{\mathrm{Var}(T)}$
$(1+b'(\theta))^2 \le \mathrm{Var}(S)\mathrm{Var}(T)$
$\le I(\theta) \mathrm{Var}(T)$

$$ \mathrm{Var}(T) \ge \frac{(1+b'(\theta))^2}{I(\theta)} $$
> **CRLB** gives low bound on minimum poss variance for an estimator. There are so many estimators to choose. The one with the lowest variance is often preferred.

$T$ unbiased $\rightarrow b=0$
$$ \mathrm{Var}(T) \ge I(\theta)^{-1} $$
> useful property

---
(Page 21)

### CRLB Multivariate Case

$\vec{X} \sim P(\vec{X}|\vec{\theta})$
and $\vec{T}(\vec{x})$ is unbiased estimator for $\vec{\theta}$,
$E[\vec{T}(\vec{x})] = \vec{\theta}$, $E[T_j(\vec{x})] = \theta_j$
Then $\left[ \mathrm{Cov}(\vec{T}(\vec{x})) \ge I^{-1}(\theta) \right]$
has elements $\mathrm{Cov}(T_j, T_k)$

Fisher Matrix $I(\theta)$ has elements
$I_{jk} = E[-\frac{\partial^2}{\partial\theta_j \partial\theta_k} \ln P(\vec{x}|\vec{\theta})]$
> **Hessian**

In particular:
$\left[ \mathrm{Var}(T_i) \ge [I^{-1}(\theta)]_{ii} \right]$
(diagonal terms)

---
(Page 22)

### Max Likelihood

$X_i \stackrel{iid}{\sim} f(x|\theta) \quad i=1, ..., N$
$L(\theta) = P(\vec{x}|\theta) = \prod_{i=1}^N f(x_i|\theta)$

$\hat{\theta}_{MLE} = \underset{\theta}{\mathrm{argmax}} \, L(\theta)$
$= \underset{\theta}{\mathrm{argmax}} \, \ln L(\theta)$
> (i think?) to find `θ_MLE`: `∂l/∂θ=0`. multiparameter `∂l/∂θ_1=0`, `∂l/∂θ_2=0` etc.

**MLE Properties** (Model is true!)
\* **Consistent**: $\hat{\theta}_{MLE} \rightarrow \theta_{true}$ as $N \rightarrow \infty$
\* **Asymptotically unbiased**: $E[\hat{\theta}_{MLE}] \rightarrow \theta_{true}$ as $N \rightarrow \infty$
\* **Asymptotically Normal**: $(\hat{\theta}_{MLE} - \theta_{true}) \rightarrow N(0, I^{-1})$
> Asymptotic normality: consistent estimators `θ̂` have dist 'around' true `θ` that approaches a normal dist (with a decreasing `σ` on `1/√N`)

[Image showing that as N increases, the distribution of `θ̂_MLE` around the true `θ` becomes a narrower Gaussian, achieving the CRLB.]

> why fisher info is useful: find covariance matrices associated with MLE
> achieves CRLB as N->inf (efficient!). No other estimator is more efficient.

---
(Page 23)

## Lecture 7
_7.2.25_

### Max Likelihood Properties (cont.)

$X_i \stackrel{iid}{\sim} f(X|\theta) \quad i=1, ..., N$
$L(\theta) = P(\vec{X}|\theta) = \prod_{i=1}^N f(X_i|\theta)$

MLE $\rightarrow \hat{\theta}_{MLE} = \underset{\theta}{\mathrm{argmax}} \, L(\theta) = \underset{\theta}{\mathrm{argmax}} \, \ln L(\theta)$

\* **Consistent**: $\hat{\theta}_{MLE} \rightarrow \theta_{true}$ as $N \rightarrow \infty$
> $(\forall \epsilon > 0, \mathrm{Pr}(|\hat{\theta}_{MLE} - \theta_{true}| > \epsilon) \rightarrow 0 \text{ as } N \rightarrow \infty)$

\* **Asymptotically Unbiased**: $E[\hat{\theta}_{MLE}] \rightarrow \theta_{true}$ as $N \rightarrow \infty$
> (but not necessarily unbiased)

\* **Asymptotically Normal**: $(\hat{\theta}_{MLE} - \theta_{true}) \stackrel{d}{\rightarrow} N(0, I^{-1})$

$I(\theta) = -E[\frac{\partial^2 l(\theta)}{\partial\theta^2}] = -NE[\frac{\partial^2 \ln f}{\partial\theta^2}]$
> **Expected Fisher Information**
$\hat{I} = \text{"observed Fisher Info."} = -\frac{\partial^2 l}{\partial\theta^2}|_{\theta=\hat{\theta}_{MLE}}$
$\approx I(\theta)$ as $N \rightarrow \infty$

\* **Efficient**: Asymptotically achieves CRLB
$\mathrm{Var}(\hat{\theta}_{MLE}) \rightarrow I^{-1}$ as $N \rightarrow \infty$
> "saturates"

---
(Page 24)

\* **Functionally invariant**: $\alpha = g(\theta)$
> "equivariance property", parameter trans.
$\hat{\alpha}_{MLE} = g(\hat{\theta}_{MLE})$

### Multiparameter Case MLE

$L(\vec{\theta}) = p(\vec{x}|\vec{\theta}) = \prod_{i=1}^N p(x_i|\vec{\theta})$
$\hat{\theta}_{MLE} = \underset{\vec{\theta}}{\mathrm{argmax}} \, L(\vec{\theta})$

\* **Asymptotically Normal**:
$\vec{\theta}_{MLE} - \vec{\theta}_{true} \stackrel{d}{\rightarrow} N(\vec{0}, \vec{I}^{-1})$
> **Inverse Fisher Matrix**

$I_{jk} = \text{Exp. Fisher Info Matrix}$
$= E[-\frac{\partial^2 l}{\partial\theta_j \partial\theta_k}] = -NE[\frac{\partial^2 \ln f}{\partial\theta_j \partial\theta_k}]$
> Hessian
Observed F.I. = $\hat{I}_{jk} = -\frac{\partial^2 l}{\partial\theta_j \partial\theta_k}|_{\vec{\theta}=\hat{\vec{\theta}}_{MLE}}$
$\approx I_{jk}$ as $N \rightarrow \infty$

\* **Asymptotically Efficient**:
$\mathrm{Cov}(\vec{\theta}_{MLE}) \rightarrow \vec{I}^{-1}$
In particular, $\mathrm{Var}(\theta_{MLE, i}) \rightarrow (\vec{I}^{-1})_{ii}$ (diagonal)
> usually we want to know variance of particular components

---
(Page 25)

**Quick Example:** (evaluate Fisher Matrix)
$X_i \sim N(\mu, \sigma^2)$, $i=1, ..., N$
$\vec{\theta} = (\mu, \sigma^2)$
$L(\vec{\theta}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2}(x_i-\mu)^2/\sigma^2}$
$\rightarrow l(\vec{\theta}) = \sum_{i=1}^N -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{1}{2}\frac{(x_i-\mu)^2}{\sigma^2}$

1st deriv:
$\frac{\partial l}{\partial\mu} = \sum_{i=1}^N \frac{(x_i-\mu)}{\sigma^2} = 0$
$\rightarrow \hat{\mu}_{MLE} = \frac{1}{N}\sum x_i = \bar{X}