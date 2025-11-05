
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
$$ \rightarrow \hat{\mu}_{MLE} = \frac{1}{N}\sum x_i = \bar{X}

> 2nd deriv:
$$ \frac{\partial^2 l}{\partial\mu^2} = \sum_{i=1}^N -1/\sigma^2 = -N/\sigma^2 $$
> deriv w.r.t. $\sigma^2$
$$ \frac{\partial l}{\partial \sigma^2} = \sum_{i=1}^N - \frac{1}{2}\frac{1}{\sigma^2} + \frac{1}{2}\frac{(x_i - \mu)^2}{(\sigma^2)^2} = 0 $$
$$ \rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2 $$
> symmetrical, unbiased but biased variance
> see e.g. wiki "Variance"
(N.B. this is **biased**! $E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2$)
But **sample variance** $S^2 = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2$ is **unbiased** for $\sigma^2$.

> 2nd deriv w.r.t $\sigma^2$
> don't actually need
$$ \frac{\partial^2 l}{\partial (\sigma^2)^2} = \sum_{i=1}^N \frac{1}{2\sigma^4} - \frac{(x_i - \mu)^2}{\sigma^6} $$
$$ E\left[\frac{\partial^2 l}{\partial (\sigma^2)^2}\right] = - \frac{N}{2\sigma^4} \quad (\text{using } E[(x_i-\mu)^2] = \sigma^2) $$
> Bessel's correction, biases/expectations etc, some $(x_i-\mu)^2$ start as random sum, $(x_i-\bar{x})^2$ depends on data $(x_i, \bar{x})$ are correlated, number of d.f. split

---
(Page 26)

> convince yourself

Cross term $E\left[\frac{\partial^2 l}{\partial \sigma^2 \partial \mu}\right] = 0$ (using $E[x_i-\mu]=0$)
> (actually $\frac{\partial^2 l}{\partial \sigma^2 \partial \mu}$ etc?)

$$ I = \begin{pmatrix} -\frac{\partial^2 l}{\partial \mu^2} & -\frac{\partial^2 l}{\partial \mu \partial \sigma^2} \\ -\frac{\partial^2 l}{\partial \sigma^2 \partial \mu} & -\frac{\partial^2 l}{\partial (\sigma^2)^2} \end{pmatrix} = \begin{pmatrix} N/\sigma^2 & 0 \\ 0 & N/2\sigma^4 \end{pmatrix} $$
$$ \rightarrow I^{-1} = \begin{pmatrix} \sigma^2/N & 0 \\ 0 & 2\sigma^4/N \end{pmatrix} $$
$$ \left. \begin{aligned} \mathrm{Var}(\hat{\mu}_{MLE}) &= \sigma^2/N \geq (I^{-1})_{11} \\ \mathrm{Var}(\hat{\sigma}^2_{MLE}) &\geq (I^{-1})_{22} \end{aligned} \right\} $$
> check this yourself. is variance greater than CRLB? should follow from MLE theory?

$$ \mathrm{Var}(\hat{\mu}_{MLE}) = (I^{-1})_{11} \text{ as } N \rightarrow \infty \quad \checkmark \text{ as expected} $$

---
(Page 27)

### Laplace Approximation
> remember how's "likelihood" your "map"

recall $l(\theta) = \ln L(\theta)$
> want to be near the parameters for MLE
Taylor expansion around MLE:
$$ l(\theta) \approx l(\hat{\theta}_{MLE}) + \underbrace{\left.\frac{\partial l}{\partial \theta}\right|_{\hat{\theta}_{MLE}}}_{0}(\theta - \hat{\theta}_{MLE}) + \frac{1}{2}\underbrace{\left.\frac{\partial^2 l}{\partial \theta^2}\right|_{\hat{\theta}_{MLE}}}_{-\hat{I}} (\theta - \hat{\theta}_{MLE})^2 + \dots $$

Asymptotically, $N \rightarrow$ large
$$ l(\theta) \approx l(\hat{\theta}_{MLE}) - \frac{1}{2}\hat{I}(\theta - \hat{\theta}_{MLE})^2 $$
$$ L(\theta) \approx L(\hat{\theta}_{MLE}) e^{-\frac{1}{2}(\theta - \hat{\theta}_{MLE})^2 / \hat{I}^{-1}} \quad \text{(Gaussian shape)} $$
> useful bit for homework, in terms of MLE result and Hess

Multiparameter case
$$ L(\vec{\theta}) \approx L(\hat{\vec{\theta}}_{MLE}) \times e^{-\frac{1}{2}(\vec{\theta} - \hat{\vec{\theta}}_{MLE})^T \hat{I} (\vec{\theta} - \hat{\vec{\theta}}_{MLE})} $$

Reminder: multidim Taylor expansion
$$ f(x) = f(a) + \underbrace{\nabla f(a)}_{\text{grad at a}}(x-a) + \frac{1}{2}(x-a)^T \underbrace{H(a)}_{\text{Hessian of f at a}}(x-a) + \dots $$
e.g. $f(x,y) = f(a,b) + \frac{\partial f}{\partial x}(x-a) + \frac{\partial f}{\partial y}(y-b) + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(x-a)^2 + \frac{\partial^2 f}{\partial y^2}(y-b)^2 + \frac{\partial^2 f}{\partial x \partial y}(x-a)(y-b) + \dots$

---
(Page 28)


### Example (slides): Calibrating type Ia supernova absolute magnitudes with intrinsic dispersion and heteroskedastic measurement error.
> Heteroskedastic

### Determining Astronomical Distances using Standard Candles
> key step in determining hubble constant.

1. **Estimate** or **model luminosity L** of a class of astronomical objects
2. **Measure apparent brightness** or **flux F**
3. **Derive distance D** to object using inverse square law
   $$ F = L/4\pi D^2 $$
4. Optical astronomers units $\mu = m-M$
   - $m$ = **apparent magnitude** (log apparent brightness, flux)
   - $M$ = **absolute magnitude** (log luminosity)
   - $\mu$ = **distance modulus** (log distance)

> can use hubble to get distances
Hubble: $V = H_0 \times D$ distance $\propto$ velocity (redshift)
> Hubble's law $z \approx H_0 d$

Now use **type Ia Supernovae** as **standard candles**

We want to **measure the hubble constant**
$\rightarrow$ can use **local distance ladder**
> ladder because one method is bootstrapped into the method of another one.

> 2nd tier of distance ladder
Can use **cepheid stars**: period of rotation linked to luminosity
> (pulsating)
> 1st tier: geometry, 2nd tier: cepheids, 3rd tier: supernovae. use links between tiers to calibrate. e.g. same distance, cepheids & supernovae in the same galaxy
> i.e. 3 rungs of the distance ladder.
> hubble flow things that are further away tend to move away from us faster

---
(Page 29)

## Lecture 8
_10.2.25_

### Recap example (slides)
- Measure **apparent brightness** or **flux F**
- Derive **distance** to object using **inverse square law**
  $F = L/4\pi D^2$
- Key step in determining **hubble constant**. > (redshift)
- 3 rungs to distance ladder: geometry, cepheids, supernovae (can calibrate between rungs).

> we will focus on cepheids
> both on board!

### Calibrating SNIa Abs Mags (board)

Calibrator sample we observe: N SNe, $s=1,...,N$
- $M_s$ = true apparent magnitude
- $\hat{M}_s$ = measured apparent magnitude
- $\hat{M}_s \sim N(M_s, \sigma_{ms}^2) \leftarrow$ given, public

equivalently, $\hat{M}_s = M_s + \mathcal{E}_{ms} \quad \mathcal{E}_{ms} \sim N(0, \sigma_{ms}^2) \leftarrow$ data

**Cepheid distance estimates** to SN galaxy:
- $\mu_s$ = true distance modulus
  $= 25 + 5\log_{10}(\frac{\text{distance}}{\text{Mpc}})$
> again with measurement error on distance
- $\hat{\mu}_s \sim N(\mu_s, \sigma_{\mu s}^2) \leftarrow$ given in data table

equiv. $\hat{\mu}_s = \mu_s + \mathcal{E}_{\mu s} \quad \mathcal{E}_{\mu s} \sim N(0, \sigma_{\mu s}^2) \leftarrow$ "data"

---
(Page 30)

### Abs Mag Distribution (Population)
> e.g. want to... intrinsic scatter looks like random variations in absolute mag due to dust, properties etc.
$M_s \sim N(M_0, \sigma_{int}^2)$ $\leftarrow$ unknown
equiv. $M_s = M_0 + \mathcal{E}_{int,s} \rightarrow N(0, \sigma_{int}^2)$

> We want to estimate $\theta$, so form likelihood eq:
> latent variable without noise $\rightarrow$ relate to observed variables
$M_s = m_s - \mu_s$ (**latent variable equation**)

Define: $\hat{M}_s = \hat{m}_s - \hat{\mu}_s$ (Estimated Abs Mag)
$= M_s + \mathcal{E}_{ms} - \mu_s - \mathcal{E}_{\mu s}$
$= M_s + \mathcal{E}_{ms} - \mathcal{E}_{\mu s} \rightarrow \sigma^2_{err,s}$
> use: variance of sum of gaussian RVs is sum of variance
$= M_s + \mathcal{E}_{err,s} \quad N(0, \sigma^2_{ms} + \sigma^2_{\mu s})$
> combine errors
> abs mag define
$\hat{M}_s = M_0 + \mathcal{E}_{int,s} + \mathcal{E}_{err,s}$
> w/o measurement errors
$\hat{M}_s \sim N(M_0, \sigma^2_{int} + \sigma^2_{err,s})$
> combine variances (as independent)

### Likelihood:
> note: $\hat{M}_s$ not i.i.d! How? (different $\sigma^2_{err,s}$)
> aka heteroskedastic
$P(\hat{M}_s|M_0, \sigma^2_{int}) = N(\hat{M}_s|M_0, \sigma^2_{int} + \sigma^2_{err,s})$
> (independence)
$L(\theta) = P(\hat{\mathbf{M}}|M_0, \sigma^2_{int}) = \prod_{s=1}^N N(\hat{M}_s|M_0, \sigma^2_{int} + \sigma^2_{err,s})$

**heteroskedastic** measurement error
> typical in astronomy
> example of heteroskedastic data is multiple sources of instruments, this one cannot be solved analytically. e.g. need to solve numerically.
**cannot solve MLE analytically!**
> (see slides for code example)

---
(Page 31)

# Astrostatistics Lecture Notes - Pages 26-166
*Transcribed from handwritten notes using Gemini 2.5 Pro*

---

> (evaluate Fisher Matrix)

### Quick Example:

$X_i \stackrel{\text{iid}}{\sim} N(\mu, \sigma^2)$ , $i=1, ..., N$
$\hat{\theta} = (\mu, \sigma^2)$
$$ L(\vec{\theta}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2}(x_i - \mu)^2 / \sigma^2} $$
$$ \rightarrow \ell(\vec{\theta}) = \sum_{i=1}^N -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{1}{2}\frac{(x_i - \mu)^2}{\sigma^2} $$
> 1st deriv.
$$ \frac{\partial \ell}{\partial \mu} = \sum_{i=1}^N \frac{(x_i - \mu)}{\sigma^2} = 0 $$
$$ \rightarrow \hat{\mu}_{MLE} = \frac{1}{N}\sum_{i=1}^N x_i = \bar{X} \rightarrow \mathrm{Var}(\hat{\mu}_{MLE}) = \sigma^2/N $$
> 2nd deriv.
$$ \frac{\partial^2 \ell}{\partial \mu^2} = \sum_{i=1}^N -1/\sigma^2 = -N/\sigma^2 $$
> deriv w.r.t. $\sigma^2$
$$ \frac{\partial \ell}{\partial \sigma^2} = \sum_{i=1}^N - \frac{1}{2}\frac{1}{\sigma^2} + \frac{1}{2}\frac{(x_i - \mu)^2}{(\sigma^2)^2} = 0 $$
$$ \rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2 $$
> symmetrical, unbiased but biased variance
> see e.g. wiki "Variance"
(N.B. this is **biased**! $E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2$)
But **sample variance** $S^2 = \frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2$ is **unbiased** for $\sigma^2$.

> 2nd deriv w.r.t $\sigma^2$
> don't actually need
$$ \frac{\partial^2 \ell}{\partial (\sigma^2)^2} = \sum_{i=1}^N \frac{1}{2\sigma^4} - \frac{(x_i - \mu)^2}{\sigma^6} $$
$$ E\left[\frac{\partial^2 \ell}{\partial (\sigma^2)^2}\right] = - \frac{N}{2\sigma^4} \quad (\text{using } E[(x_i-\mu)^2] = \sigma^2) $$
> Bessel's correction, biases/expectations etc, some $(x_i-\mu)^2$ start as random sum, $(x_i-\bar{x})^2$ depends on data $(x_i, \bar{x})$ are correlated, number of d.f. split

---
(Page 26)

> convince yourself

Cross term $E\left[\frac{\partial^2 \ell}{\partial \sigma^2 \partial \mu}\right] = 0$ (using $E[x_i-\mu]=0$)
> (actually $\frac{\partial^2\ell}{\partial \sigma^2 \partial \mu}$ etc?)

$$ \mathcal{I} = \begin{pmatrix} -\frac{\partial^2 \ell}{\partial \mu^2} & -\frac{\partial^2 \ell}{\partial \mu \partial \sigma^2} \\ -\frac{\partial^2 \ell}{\partial \sigma^2 \partial \mu} & -\frac{\partial^2 \ell}{\partial (\sigma^2)^2} \end{pmatrix} = \begin{pmatrix} N/\sigma^2 & 0 \\ 0 & N/2\sigma^4 \end{pmatrix} $$
$$ \rightarrow \mathcal{I}^{-1} = \begin{pmatrix} \sigma^2/N & 0 \\ 0 & 2\sigma^4/N \end{pmatrix} $$
$$ \left. \begin{aligned} \mathrm{Var}(\hat{\mu}_{MLE}) &= \sigma^2/N \geq (\mathcal{I}^{-1})_{11} \\ \mathrm{Var}(\hat{\sigma}^2_{MLE}) &\geq (\mathcal{I}^{-1})_{22} \end{aligned} \right\} $$
> check this yourself. is variance greater than CRLB? should follow from MLE theory?

$$ \mathrm{Var}(\hat{\mu}_{MLE}) = (\mathcal{I}^{-1})_{11} \text{ as } N \rightarrow \infty \quad \checkmark \text{ as expected} $$

---
(Page 27)

### Laplace Approximation
> remember how's "likelihood" your "map"

recall $\ell(\theta) = \ln L(\theta)$
> want to be near the parameters for MLE
Taylor expansion around MLE:
$$ \ell(\theta) \approx \ell(\hat{\theta}_{MLE}) + \underbrace{\left.\frac{\partial \ell}{\partial \theta}\right|_{\hat{\theta}_{MLE}}}_{0}(\theta - \hat{\theta}_{MLE}) + \frac{1}{2}\underbrace{\left.\frac{\partial^2 \ell}{\partial \theta^2}\right|_{\hat{\theta}_{MLE}}}_{-\hat{\mathcal{I}}} (\theta - \hat{\theta}_{MLE})^2 + \dots $$

Asymptotically, $N \rightarrow$ large
$$ \ell(\theta) \approx \ell(\hat{\theta}_{MLE}) - \frac{1}{2}\hat{\mathcal{I}}(\theta - \hat{\theta}_{MLE})^2 $$
$$ L(\theta) \approx L(\hat{\theta}_{MLE}) e^{-\frac{1}{2}(\theta - \hat{\theta}_{MLE})^2 / \hat{\mathcal{I}}^{-1}} \quad \text{(Gaussian shape)} $$
> useful bit for homework, in terms of MLE result and Hess

Multiparameter case
$$ L(\vec{\theta}) \approx L(\hat{\vec{\theta}}_{MLE}) \times e^{-\frac{1}{2}(\vec{\theta} - \hat{\vec{\theta}}_{MLE})^T \hat{\mathcal{I}} (\vec{\theta} - \hat{\vec{\theta}}_{MLE})} $$

Reminder: multidim Taylor expansion
$$ f(x) = f(a) + \underbrace{\nabla f(a)}_{\text{grad at a}}(x-a) + \frac{1}{2}(x-a)^T \underbrace{H(a)}_{\text{Hessian of f at a}}(x-a) + \dots $$
e.g. $f(x,y) = f(a,b) + \frac{\partial f}{\partial x}(x-a) + \frac{\partial f}{\partial y}(y-b) + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(x-a)^2 + \frac{\partial^2 f}{\partial y^2}(y-b)^2 + \frac{\partial^2 f}{\partial x \partial y}(x-a)(y-b) + \dots$

---
(Page 28)

### Example (slides): Calibrating type Ia supernova absolute magnitudes with intrinsic dispersion and heteroskedastic measurement error.
> Heteroskedastic

### Determining Astronomical Distances using Standard Candles
> key step in determining hubble constant.

1. **Estimate** or **model luminosity L** of a class of astronomical objects
2. **Measure apparent brightness** or **flux F**
3. **Derive distance D** to object using inverse square law
   $$ F = L/4\pi D^2 $$
4. Optical astronomers units $\mu = m-M$
   - $m$ = **apparent magnitude** (log apparent brightness, flux)
   - $M$ = **absolute magnitude** (log luminosity)
   - $\mu$ = **distance modulus** (log distance)

> can use hubble to get distances
Hubble: $V = H_0 \times D$ distance $\propto$ velocity (redshift)
> Hubble's law $z \approx H_0 d$

Now use **type Ia Supernovae** as **standard candles**

We want to **measure the hubble constant**
$\rightarrow$ can use **local distance ladder**
> ladder because one method is bootstrapped into the method of another one.

> 2nd tier of distance ladder
Can use **cepheid stars**: period of rotation linked to luminosity
> (pulsating)
> 1st tier: geometry, 2nd tier: cepheids, 3rd tier: supernovae. use links between tiers to calibrate. e.g. same distance, cepheids & supernovae in the same galaxy
> i.e. 3 rungs of the distance ladder.
> hubble flow things that are further away tend to move away from us faster

---
(Page 29)

## Lecture 8
_10.2.25_

### Recap example (slides)
- Measure **apparent brightness** or **flux F**
- Derive **distance** to object using **inverse square law**
  $F = L/4\pi D^2$
- Key step in determining **hubble constant**. > (redshift)
- 3 rungs to distance ladder: geometry, cepheids, supernovae (can calibrate between rungs).

> we will focus on cepheids
> both on board!

### Calibrating SNIa Abs Mags (board)

Calibrator sample we observe: N SNe, $s=1,...,N$
- $M_s$ = true apparent magnitude
- $\hat{M}_s$ = measured apparent magnitude
- $\hat{M}_s \sim N(M_s, \sigma_{ms}^2) \leftarrow$ given, public

equivalently, $\hat{M}_s = M_s + \mathcal{E}_{ms} \quad \mathcal{E}_{ms} \sim N(0, \sigma_{ms}^2) \leftarrow$ data

**Cepheid distance estimates** to SN galaxy:
- $\mu_s$ = true distance modulus
  $= 25 + 5\log_{10}(\frac{\text{distance}}{\text{Mpc}})$
> again with measurement error on distance
- $\hat{\mu}_s \sim N(\mu_s, \sigma_{\mu s}^2) \leftarrow$ given in data table

equiv. $\hat{\mu}_s = \mu_s + \mathcal{E}_{\mu s} \quad \mathcal{E}_{\mu s} \sim N(0, \sigma_{\mu s}^2) \leftarrow$ "data"

---
(Page 30)


### Abs Mag Distribution (Population)
> e.g. want to... intrinsic scatter looks like random variations in absolute mag due to dust, properties etc.
$M_s \sim N(M_0, \sigma_{int}^2)$ $\leftarrow$ unknown
equiv. $M_s = M_0 + \mathcal{E}_{int,s} \rightarrow N(0, \sigma_{int}^2)$

> We want to estimate $\theta$, so form likelihood eq:
> latent variable without noise $\rightarrow$ relate to observed variables
$M_s = m_s - \mu_s$ (**latent variable equation**)

Define: $\hat{M}_s = \hat{m}_s - \hat{\mu}_s$ (Estimated Abs Mag)
$= M_s + \mathcal{E}_{ms} - \mu_s - \mathcal{E}_{\mu s}$
$= M_s + \mathcal{E}_{ms} - \mathcal{E}_{\mu s} \rightarrow \sigma^2_{err,s}$
> use: variance of sum of gaussian RVs is sum of variance
$= M_s + \mathcal{E}_{err,s} \quad N(0, \sigma^2_{ms} + \sigma^2_{\mu s})$
> combine errors
> abs mag define
$\hat{M}_s = M_0 + \mathcal{E}_{int,s} + \mathcal{E}_{err,s}$
> w/o measurement errors
$\hat{M}_s \sim N(M_0, \sigma^2_{int} + \sigma^2_{err,s})$
> combine variances (as independent)

### Likelihood:
> note: $\hat{M}_s$ not i.i.d! How? (different $\sigma^2_{err,s}$)
> aka heteroskedastic
$P(\hat{M}_s|M_0, \sigma^2_{int}) = N(\hat{M}_s|M_0, \sigma^2_{int} + \sigma^2_{err,s})$
> (independence)
$L(\theta) = P(\hat{\mathbf{M}}|M_0, \sigma^2_{int}) = \prod_{s=1}^N N(\hat{M}_s|M_0, \sigma^2_{int} + \sigma^2_{err,s})$

**heteroskedastic** measurement error
> typical in astronomy
> example of heteroskedastic data is multiple sources of instruments, this one cannot be solved analytically. e.g. need to solve numerically.
**cannot solve MLE analytically!**
> (see slides for code example)

---
(Page 31)

### Supernova Absolute Magnitude Distribution: Selection Effects (board)
> begin thought experiment to see what might go wrong, what might we need to account for?

Same ex. but now
Assume measurement error = 0 = $\sigma_{m,s} = \sigma_{\mu s}$
> for simplicity of argument
Population variability $\sigma_{int} \rightarrow \sigma$.
Only intrinsic population is source of variability ($\sigma^2 = \sigma_{int}^2$).

Suppose distance is the same for an entire sample of SN
$\mu_s = \mu$

population dist
iid
$M_s \sim N(M_o, \sigma^2)$
> latent variable eq
> $(M_s = M_o + \mu_s)$

combine
$\rightarrow m_s \sim N(M_o + \mu, \sigma^2)$ $s=1,...,N$

[Diagram showing the effect of a survey limit on observing supernovae. The x-axis is apparent magnitude 'm', labelled 'dimmer' to the right. The y-axis is distance modulus 'μ', labelled 'fainter' upwards. A diagonal dashed line represents 'm = M_o + μ'. Two distances, μ1 and μ2, are shown. At each distance, there is a Gaussian distribution of observed magnitudes 'm_s' centered on 'm = M_o + μ_i'. A vertical dashed line at 'm_lim' represents the survey limit. For the closer distance μ1, the entire Gaussian is to the left of m_lim, so all SN are observed. For the farther distance μ2, the Gaussian is centered closer to m_lim, and the right tail of the distribution is cut off by the limit, meaning some fainter SN at that distance go unobserved. The observed SN population is shown as blue filled Gaussians, while unobserved ones are indicated by 'x's under the truncated part of the curve. The equation for the diagonal line is μ = m - M_o.]

Labels on diagram:
- Above μ2 Gaussian: `observed SN`
- Right of μ2 Gaussian, past m_lim: `unobserved SN`
- To the right of m_lim: `to faint to be observable`
- Below x-axis: `m1 = M_o + μ1`, `m2 = M_o + μ2`
- Vertical line: `M_lim` `survey limit`

> (We stop)
> Telescopes can only detect up to certain magnitude limit, can't see anything dimmer than this.
> So for a set of SNs, say Ms are acting "majoritively" look similar but in the end we don't see the one above some limit, what do we do?

---
(Page 32)

Data: $\{m_s\} = \vec{m}$
fixed $\mu$ for sample

Naive Likelihood: $P(m_s|M_o, \sigma^2) = N(m_s|\mu+M_o, \sigma^2)$
$L(M_o, \sigma^2) = \prod_{s=1}^N P(m_s|M_o, \sigma^2)$
> naively apply naive likelihood

If $M_o + \mu$ close to $M_{lim}$, MLE will be biased:
$$ \left\{ \begin{array}{ll} \hat{M}_o & \text{too bright} \\ \hat{\sigma} & \text{too small} \end{array} \right. - \left( \begin{array}{l} \text{we didn't see} \\ \text{m,dim everywhere} \end{array} \right) $$
> b/c tail of dist. is cut off.
> different width, smaller than true width

> how do we resolve this?
### Accounting For selection Effects (sheet 1 & 2)

Let $I_s = \begin{cases} 1, & \text{if SN observed} \\ 0, & \text{if SN NOT observed} \end{cases}$
> indicator variable

Define Selection Function:
> prob indicator given data
$$ P(I_s|m_s) = \begin{cases} 1, & m_s < M_{lim} \\ 0, & m_s \ge M_{lim} \end{cases} $$
$$ = 1 - H(m_s - M_{lim}) $$
> heaviside step fn

[Graph of the selection function. The x-axis is m_s, the y-axis is P(I_s|m_s). The function is 1 for m_s < M_lim and drops to 0 at m_s = M_lim. A point is marked at M_lim on the x-axis.]

---
(Page 33)

## Lecture 9
_12.2.25_

> (brief recap SN problem setup on board - useful for sheet 1)
> mainly

### SELECTION EFFECTS cont.

recap:
$M_s = N(M_o, \sigma^2)$
$s=1,...,N$ (known)
$\mu_s = \mu$ (known)
> known distance
> no measurement error

[Diagram showing a Gaussian distribution P(M) centered at M. A vertical line at M_lim cuts off the right tail of the distribution. The observed samples are marked with 'x's under the curve to the left of M_lim.]

Formulate likelihood to account for selection effects:
Let $I_s = \begin{cases} 1, & \text{if SN observed} \\ 0, & \text{if SN NOT observed} \end{cases}$

Selection function:
$$ S(m_s) = P(I_s=1|m_s) = \begin{cases} 1, & m_s < M_{lim} \\ 0, & m_s \ge M_{lim} \end{cases} $$

[Graph of the selection function S(m_s) vs m. S(m_s) is 1 for m < M_lim and 0 for m >= M_lim. The x-axis is labeled m, y-axis S(m_s). A point is marked at M_lim on the x-axis.]
> $1-H(m_s-M_{lim})$

---
(Page 34)

### Observed Data Likelihood:
$$ P(m_s|I_s=1, \theta) = \frac{P(I_s=1, m_s | \theta)}{P(I_s=1|\theta)} $$
$$ = \frac{P(I_s=1|m_s, \theta)P(m_s|\theta)}{\int P(I_s=1|m_s, \theta)P(m_s|\theta)dm_s} $$
$$ = \frac{S(m_s)N(m_s|M_o+\mu, \sigma^2)}{\int S(m_s)N(m_s|M_o+\mu, \sigma^2)dm_s} $$
> cut off above M_lim
$$ = \frac{[1-H(m_s-M_{lim})]N(m_s|M_o+\mu, \sigma^2)}{\int_{-\infty}^{M_{lim}} N(m_s|M_o+\mu, \sigma^2)dm_s} $$
> normalise
Gaussian CDF $\rightarrow \Phi$

> "look up on wikipedia"
(TRUNCATED) NORMAL
$ = TN(m_s|M_o+\mu, \sigma^2, -\infty, M_{lim}) $
> $P(m_s|I_s=1, \theta)$
untruncated mean & variance $\uparrow$ lower truncation limit $\uparrow$ upper truncation limit $\uparrow$

[Diagram of a truncated normal distribution. A Gaussian curve is shown, with the area to the right of M_lim cut off. The original mean M_o+μ is marked, which is different from the mean of the truncated distribution.]

Challenge: what if $S(m_s) = P(I_s=1|m_s) = \Phi\left(\frac{M_{lim}-m_s}{\sigma_{lim}}\right)$ ?
> Not a sharp selection limit, but a soft boundary. e.g. detection of objects in sea of other 'clutter' objects.

[Diagram showing a smooth, sigmoid-like selection function S(m_s) decreasing from 1 to 0 around M_lim. The curve is annotated with σ_lim, indicating the softness of the cutoff. The x-axis is m_s and y-axis is S(m_s).]
> $\sigma_{lim}$ is a parameter of the model

---
(Page 35)

(slides)
> Ex sheet 1 q 2: star formation in Perseus
> 1) clustering dist of stars differ from field
> 2) stars in dense cloud regions are hidden
> 3) selection effect
> 3) in densest regions, will be where largest stars form
> -> pareto or power law distribution. $P(m) \propto m^{-\alpha}$ (for $m>m_o$)

[Diagram showing a power-law distribution. A peaky 'naive likelihood' is drawn, contrasted with a broader 'true likelihood'. The x-axis is the power law exponent α.]
> naive MLE -> 'biased', does not account for selection effects.

### QUANTIFYING UNCERTAINTY USING BOOTSTRAP

Frequentist interpretation:
Consider variability of your estimator $g(\vec{x})$ for $\theta$ under (imaginary) repetitions of your experiment. (Random realisations of the potential data).

How does $g(\vec{x})$ behave under the potential datasets you did not observe?
under $p(x|\theta)$
e.g. $Var[g(\vec{x})] = E[g(\vec{x}) - E[g(\vec{x})]]$

If $g(\vec{x})$ is approximately Gaussian distributed
$\implies 68\%$ confidence interval $g(\vec{z}) \pm \sqrt{Var(g(\vec{z}))}$
> standard confidence interval
> $[g(\vec{z})-\sqrt{Var(g(\vec{z}))}, g(\vec{z})+\sqrt{Var(g(\vec{z}))}]$

$(1-\alpha)\%$ confidence interval $[L(\vec{x}), U(\vec{x})]$ contains the true value $\theta_{true}$ in at least $(1-\alpha)\%$ of the realisations.
> assuming repeated experiments, resulting intervals that contain the parameter will approach 68%
> NOT that -> parameter interval contains the true parameter (common misconception)

---
(Page 36)

> get different interval for each experiment -> 68% of intervals contain θ. These realisations contain θ.

$[L(\vec{x}), U(\vec{x})]$ is a random interval vs. $[L(\vec{x}_{obs}), U(\vec{x}_{obs})]$ evaluated on observed numerical values, only one dataset $\vec{x}_{obs}$! (either contains $\theta_{true}$ or doesn't)

### Bootstrap:

Use the observed dataset to simulate the variability of the unobserved (imaginary) data sets.
BOOTSTRAP SAMPLE = sample with replacement from the observed dataset to the sample size

Eg. $X_1, ..., X_5 \stackrel{iid}{\sim} \text{Poisson}(\lambda)$
$\rightarrow P(X_i) = \frac{\lambda^{X_i} e^{-\lambda}}{X_i!}$
> could do w/ max. likelihood but suppose don't know dist.

Real data (observed) $\vec{X}_{obs}=(3,8,2,4,5)$.
Suppose you want to estimate the skewness of $P(x)$ (asymmetry)
i.e. skewness $= \frac{E[(x-\mu)^3]}{(\sigma^2)^{3/2}} \rightarrow \substack{\text{true mean} \\ \mu=\lambda \\ \text{true variance} \\ \sigma^2=\lambda}$

Sample skewness $g(\vec{x}) = \frac{\frac{1}{N}\sum_{i=1}^N (x_i-\bar{x})^3}{\left(\sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i-\bar{x})^2}\right)^3}$

---
(Page 37)

Bootstrap B "replicate" datasets from observed dataset:
> "sample w/ replacement" from original eg sample 10 times from 3,8,2,4,5 gives

$\vec{X}^{obs} = (3,8,2,4,5) \quad \hat{g}_{obs} = g(\vec{x}^{obs}) = 0.6927$
$\vec{X}^{b=1} = (2,5,4,4,4) \quad \hat{g}_1 = g(\vec{x}^{b=1}) = -0.8625$
$\vec{X}^{b=2} = (2,4,2,8,8) \quad \hat{g}_2 = g(\vec{x}^{b=2}) = 0.2115$
$\vec{X}^{b=3} = (5,2,8,2,5) \quad \hat{g}_3 = g(\vec{x}^{b=3}) = 0.3436$
...
$\vec{X}^{b=B} = \quad\quad\quad\quad\quad\quad \hat{g}_B =$

Can now compute sample variance
$\hat{Var}(\{\hat{g}_1, ..., \hat{g}_B\}) \rightarrow \frac{1}{N-1}\sum_i (x_i-\bar{x})^2 (?)$
Standard error = $\sqrt{\hat{Var}}$

$\hat{g} = 0.6927 \pm 0.635 \approx 68\% \text{ C.I.}$

> (slides)
> back to star formation paper example:
> how do I get standard error on exponent?
> draw bootstrap replicates and compute this against bootstrap
> get model to each bootstrapped realisation
> [Image of a histogram labeled 'true'] [Image of a histogram labeled 'naive'] plot max likelihood estimates for each as histograms -> naive (little bit of a different peak)

---
(Page 38)

## Lecture 10
_14.2.25_

### REGRESSION (slides)

* Fit a function $E[y|x] = f(x;\theta)$ for the mean relation between $y$ and $x$.

* Basic approaches
    $\rightarrow$ ordinary least squares (homoskedastic scatter)
    $\rightarrow$ generalized least squares (heteroskedastic, correlated scatter)
    $\rightarrow$ weighted least squares (minimum $\chi^2$, known variance)
    $\rightarrow$ maximum likelihood

* Real data problems require more complex modelling
    $\rightarrow$ regression dilution from covariate measurement errors

[Diagram showing scattered data points with error bars, with a regression line fit through them. The x-axis is luminosity. The y-axis is not labeled.]
> eg. each point has diff measurement error. regression dilution; namely apply OLS -> get biased Pluto-slope? (measurement error in x) (sheet 2)

### Ordinary least squares (OLS):

Linear model $y_i = \beta_0 + \sum_{j=1}^{k-1} \beta_j x_{ij} + \epsilon_i$
$\downarrow$
$Y = X\beta + \epsilon$
$i=1,...,N$ objects, $E[\epsilon_i]=0$, homoskedastic
$Var[\epsilon_i]=\sigma^2$ (known).

---
(Page 39)

$y_i = \beta_0 + \sum_{j=1}^{k-1} \beta_j x_{ij} + \epsilon_i$
> don't think this is too important to memorise? (slides)
> error bars in y only, i.e. error in x is negligible
> minimise sum squared of these distances

> individual sum of squares
Minimise w.r.t. $\beta$ (solve for gradient=0):
$$ RSS = \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^{k-1} \beta_j x_{ij})^2 = (Y-X\beta)^T(Y-X\beta) $$
> (*simple linear regression via wiki) eg. for $k=2$

$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T Y$
> got except for?
$Var(\hat{\beta}_{OLS}) = \sigma^2(X^T X)^{-1}$
> unbiased & has minimum variance under these assumptions
$E(\hat{\beta}_{OLS}) = \beta$ (unbiased) $\rightarrow$ BLUE
> best linear unbiased estimate

Estimate unknown variance $\sigma^2$
> OLS estimate of variance $\sigma^2$ is $\hat{\sigma}^2 = \frac{RSS}{n-k}$
> (for 2 parameter example)
$\hat{\sigma}^2 = \frac{1}{N-k}(Y-X\hat{\beta})^T(Y-X\hat{\beta})$
> k=2 eg. simple linear eg. 2 parameter (intercept, slope)

(WEIGHTED LEAST SQUARES) - aka $\chi^2$ minimisation:
$X^2 = \sum_{i=1}^N \frac{(y_i - \beta_0 - \sum_j \beta_j x_{ij})^2}{\sigma_i^2 \quad \leftarrow \text{known!}}$
> $\chi^2$ r.v. = sum of squared Gaussian r.v.
> $\downarrow$
> $X^2 \sim \chi^2_{N-k}$ (if constraints)

Minimise w.r.t. $\beta$:
If Gaussian errors, at $\beta=\beta_{min}$, $X^2 \sim \chi^2_{N-k}$.
Model check: $E(\chi^2_{N-k}) = N-k \rightarrow (\text{if assumption is true (contrast)})$
$\frac{X^2}{N-k} \approx 1$ (for large $N-k$)

> A sketch: should have a $\chi^2$ distribution with N-k degrees of freedom (if gaussian errors?)
> reduced $\chi^2$ statistic: a way to test model fit.

---
(Page 40)

These are special cases of generalized least squares.
Linear model $y_i = \beta_0 + \sum_{j=1}^{k-1} \beta_j x_{ij} + \epsilon_i \quad i=1,..,N$ objects
$\downarrow$
$Y = X\beta + \epsilon$
$E[\epsilon_i]=0$, correlated errors
$Var[\epsilon] = Cov[\epsilon, \epsilon^T] = W$ (known)

Minimize w.r.t. $\beta$
$\hat{RSS} = (Y-X\beta)^T W^{-1} (Y-X\beta)$
$\hat{\beta}_{GLS} = (X^T W^{-1} X)^{-1} X^T W^{-1} Y$
$E[\hat{\beta}_{GLS}] = \beta$ (unbiased)
$Var[\hat{\beta}_{GLS}] = (X^T W X)^{-1}$

They can also be thought of as maximum likelihood (assuming Gaussian errors)
$Y=X\beta + \epsilon$, $Y \sim N(X\beta, W)$
Maximize w.r.t. $\beta$
$L(\beta) = P(Y|\beta, X) = N(Y|X\beta, W)$
> this week's "linear regression" talk on moodle covers this in more detail
> (read slides)

---
(Page 41)

> minimum $\chi^2$ estimation gives values of $\theta$ that make $\chi^2(\theta)$ as small as possible
### $\chi^2$ MINIMISATION (board)
(vs max likelihood)
> "weighted least squares" errors not iid (but still indep) -> weight according to true $\sigma_i^2$ so that true dist doesn't get distorted by points w/ large $\sigma_i$

$(x_i, y_i)$ $i=1,...,N$
Case: Variance unknown
$y_i = f(x_{ij}, \theta) + \epsilon_i$, $Var(\epsilon_i) = \sigma_i^2$ known
$= a+bx_i + \epsilon_i$ (linear)

$\chi^2(\theta) = \sum_{i=1}^N \frac{(y_i - f(x_i; \theta))^2}{\sigma_i^2}$
$\hat{\theta} = \underset{\theta}{\operatorname{argmin}} \chi^2$

Relation to maximum likelihood:
$y_i = f(x_i;\theta) + \epsilon_i$, $\epsilon_i \stackrel{iid}{\sim} N(0, \sigma_i^2)$
$P(y_i|x_i) = N(y_i|f(x_i;\theta), \sigma_i^2)$
$L(\theta) = \prod_{i=1}^N P(y_i|x_i;\theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_i^2}} e^{-\frac{1}{2}(y_i - f(x_i;\theta))^2/\sigma_i^2}$
$-2 \ln L(\theta) = \sum_{i=1}^N \ln(2\pi\sigma_i^2) + \sum_{i=1}^N \frac{(y_i-f(x_i;\theta))^2}{\sigma_i^2}$
$= \sum_{i=1}^N \ln(2\pi\sigma_i^2) + \chi^2(\theta)$
> $\chi^2(\theta)$ is constant
> $(min \chi^2(\theta))$ is same as max likelihood

$\implies$ If $\sigma_i$ known: $\hat{\theta}_{min \chi^2} = \hat{\theta}_{MLE}$
$\implies$ If $\sigma_i^2$ (or a component of it) unknown then NOT true!

> (slides again, eg. ML is used in GLS).

---
(Page 42)

Case: Variance component unknown: $\leftarrow$ unknown
$y_i = f(x_{ij}, \theta) + \epsilon_{int}^i + \epsilon_m^i$ $\quad \epsilon_{int}^i \sim N(0, \sigma_{int}^2)$
$\quad\quad\quad\quad\quad\quad\quad\quad\quad \epsilon_m^i \sim N(0, \sigma_{m,i}^2) \leftarrow$ known

$\chi^2(\theta, \sigma_{int}^2) = \sum_{i=1}^N \frac{(y_i - f(x_{ij}, \theta))^2}{\sigma_{int}^2 + \sigma_{m,i}^2}$
Problem: $\chi^2$ is minimised when $\sigma_{int} \rightarrow \infty$ !!

Max. likelihood:
$P(y_i|x_i, \theta, \sigma_{int}^2) = N(y_i|f(x_{ij}\theta), \sigma_{int,i}^2 + \sigma_{m,i}^2)$
$L(\theta, \sigma_{int}^2) = \prod_{i=1}^N [2\pi(\sigma_{int,i}^2+\sigma_{m,i}^2)]^{-1/2} e^{-\frac{1}{2}(y_i-f(x_{ij},\theta))^2 / (\sigma_{int}^2+\sigma_{m,i}^2)}$
$-2\ln L(\theta, \sigma_{int}^2) = \sum_{i=1}^N \frac{(y_i - f(x_{ij}\theta))^2}{\sigma_{int}^2 + \sigma_{m,i}^2} + \ln[2\pi(\sigma_{int}^2+\sigma_{m,i}^2)]$
$\quad\quad\quad\quad\quad\quad\quad\quad\quad$ (1) $\quad\quad\quad\quad$ (2) $\uparrow$
When $\sigma_{int} \rightarrow \text{large}$, $\downarrow$ but $\uparrow$

Max likelihood $\rightarrow$ finite estimate of $\sigma_{int}$!
> (e.g. OLS assumes gaussian, independent errors with zero mean, constant variance, but derived from explicit modelling)

Lesson: $\chi^2$ is an ad-hoc prescription. Likelihood is derived from explicit modelling assumptions.

> slides: OLS leads to regression dilution when there are errors in x (really measurement errors) which causes slope to be shallower than true value aka regression dilution.
> (OLS gives shallower slope if you have error on x data, regression dilution)

---
(Page 43)

### Probabilistic Generative Modelling (slides)

> ($P(D|\alpha,\theta)$)

* Forward model comprises series of probabilistic steps describing conceptually how the observed data was generated from the parameters of interest.

* Can introduce intermediate parameters / unobserved latent variables $\alpha$ (e.g. true values corresponding to the observed data)
> ... what data would be had if there had been no measurement error.

* From forward model, derive the sampling distribution
e.g. $P(D|\theta) = \int P(D|\alpha)P(\alpha|\theta) d\alpha$
> integrate out to have returns of interest

* Using observed data D, draw inference from likelihood function
$L(\theta) = P(D|\theta)$

* Or if Bayesian with prior $P(\theta)$: sample posterior
$P(\theta|D) \propto P(D|\theta)P(\theta)$

> params: $\theta$ want, $\alpha$ nuisance
> $\int P(D|\theta) = \int [ \int P(D, \alpha|\theta)d\alpha ] = \int P(D|\alpha, \theta) P(\alpha|\theta) d\alpha$
> two ways:
> $p(\theta|D) = \int p(\theta, \alpha|D) d\alpha \quad \checkmark$
> show as before from joint $P(D|\theta) = \int P(D|\alpha,\theta)P(\alpha|\theta)d\alpha$

---
(Page 44)

> (how do I come up with model for this complex situation where I have measurement error in my x and y data?)

### Generative Model (slides) (same example from L2)

- Population distribution $\quad \xi \sim N(\mu, \tau^2)$
- Regression Model $\quad \eta_i|\xi_i \sim N(\alpha+\beta\xi_i, \sigma^2)$
- Measurement Error $\quad [x_i, y_i]|\xi_i, \eta_i \sim N([\xi_i, \eta_i], \Sigma)$

for this example let $\Sigma = \begin{pmatrix} \sigma_{x_i}^2 & 0 \\ 0 & \sigma_{y_i}^2 \end{pmatrix}$ (for simplicity)
$x_i = \xi_i + \epsilon_{x,i} \leftarrow \sigma_{x,i}^2$
$y_i = \eta_i + \epsilon_{y,i} \leftarrow \sigma_{y,i}^2$
> probabilistic generative model for linear regression

- Population dist. indep. variable $\quad \psi=(\mu,\tau)$
- Regression Parameters $\quad \theta=(\alpha,\beta,\sigma^2)$
- Latent (true) variables (no meas. error) $\quad (\xi_i, \eta_i)$
- Observed data with measurement uncertainties $(\sigma_{xi}, \sigma_{yi})$ $\quad (x_i, y_i) \rightarrow \Sigma = \begin{pmatrix} \sigma_{xi}^2 & 0 \\ 0 & \sigma_{yi}^2 \end{pmatrix}$

> Now can
Generate observed data from latent variables. (see slides -> same as L3)
> (now I've introduced all these new parameters how do I get rid of them?)
### Formulating likelihood Function: Marginalising latent variables
> (derive this sheet 2) (and on slides) "marginalisation"

$P(x_i, y_i|\theta, \psi) = \iint P(x_i, y_i, \xi_i, \eta_i | \theta, \psi) d\xi_i d\eta_i$
> "observed data likelihood" $\quad$ "complete data likelihood"
$P(x_i, y_i|\theta, \psi) = \iint \underbrace{P(x_i,y_i|\xi_i,\eta_i)}_{\text{measurement error}} \underbrace{P(\eta_i|\xi_i,\theta)}_{\text{regression}} \underbrace{P(\xi_i|\psi)}_{\substack{\text{population} \\ \text{distribution of covariate}}} d\xi_i d\eta_i$

---
(Page 45)

## Lecture 11
_17.2.25_

(slides)
> obtain from previous, multiply marginal likelihoods to get likelihood for all data
Observed data likelihood:
$$ P(x,y|\theta,\psi) = \prod_{i=1}^N P(x_i, y_i|\theta,\psi) \quad \text{(indep.)} $$

In frequentist statistics: distinction between data and parameters: parameters are fixed and unknown, but not "random". Only "data" are random realisations of random variables.

What is the nature of the latent variables $\xi_i, \eta_i$?
- They have distribution $(\xi_i, \eta_i) \sim P(\xi_i, \eta_i|\theta, \psi) = P(\eta_i|\xi_i, \theta) P(\xi_i|\psi)$
- Often called "nuisance parameters" $\rightarrow$ needed to complete the model but not the parameters of interest ($\theta, \psi$).
- Are the latent variables "data" or "parameters"?
> "missing data" $\quad$ "nuisance parameters"

### BAYESIAN VIEWPOINT

* There is a symmetry between data D and parameters $\theta$ - both are r.v.s described by probability distributions

* Actually they are described by a joint probability
$P(D, \theta)$

---
(Page 46)

(slides)
* Data are r.v.s whose realisations are observed, parameters are r.v.s not observed

* Goal is to infer the unobserved parameters from the observed data using the rules of probability
$$ P(\theta|D) = \frac{P(D, \theta)}{P(D)} \quad \text{CONDITIONAL PROBABILITY} $$
$$ P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} \quad \text{BAYES' THEOREM} $$

* Probability interpreted as degree of belief / uncertainty in hypothesis
> how sure are you in the values of that parameter?

### Bayes' Theorem
* Joint probability of data & parameters: $P(D, \theta) = P(D|\theta)P(\theta) = P(\theta|D)P(D)$
* Probability of parameters given data:
$$ \underbrace{P(\theta|D)}_{\substack{\text{posterior prob.}: \\ \text{degree of belief}}} = \frac{\overbrace{P(D|\theta)}^{\substack{\text{likelihood} \\ \text{sampling distribution}}} \overbrace{P(\theta)}^{\substack{\text{prior prob.}: \\ \text{degree of belief}}}}{\underbrace{P(D)}_{\substack{\text{normalisation} \\ \text{constant}}}} $$

> example: bayesian inference of the dimensionality of space time. (see lecture)
> observe light & detect w/ interferometer, signal from black hole merger (?)
> idea: OK, use your tools to infer dimensionality of spacetime
> dimensionality of spacetime is not a parameter that varies or changes. either one thing or another, but we don't know it. what is our degree of belief in the hypothesis.

---
(Page 47)

### Simple Gaussian Example (slides)
Frequentist Confidence vs. Bayesian Credible intervals

1) Frequentist Confidence Interval
$Y_1, ..., Y_N \stackrel{iid}{\sim} N(\mu, 1)$ (sample mean)
Sampling dist. of statistic $\bar{Y} \sim N(\mu, \sigma^2/N)$
$[\bar{Y} - \sigma_{\bar{Y}}, \bar{Y}+\sigma_{\bar{Y}}]$ is a 68% confidence interval
* Under repeated experiments, 68% of the (random) confidence intervals constructed this way will contain (cover) $\mu$
This does not mean that the probability is 68% that $\mu$ lies within the interval ($y_{obs}$)
> frequentist view: $\mu$ is fixed number, it does not have prob dist. interval will either cover that number or it doesn't

2) Bayesian credible Interval (board)
assume flat prior: $P(\mu) \propto 1$
> every value of $\mu$ is equally likely
> (not normalizable if interval is $-\infty$ to $\infty$, so say $\propto 1$ (?)) "improper prior"
can derive posterior $p(\mu|Y=y_{obs})$:
$Y_i \stackrel{iid}{\sim} N(\mu, \sigma^2)$
> unknown $\uparrow$ $\uparrow$ known
Likelihood: $P(\vec{y}|\mu,\sigma^2) = \prod_{i=1}^N N(Y_i;\mu,\sigma^2)$
$= \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2}(y_i-\mu)^2/\sigma^2}$
$= (2\pi\sigma^2)^{-N/2} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i-\mu)^2}$

---
(Page 48)

Likelihood $P(\vec{y}|\mu, \sigma^2)$ only depends on data $\vec{y}$ through the sufficient statistics $\bar{y}$ and $S^2$. Any two datasets $\vec{y}$ and $\vec{y}'$ with same sufficient statistics will give the same inference model.
Likelihood principle: all info about parameters from data is located in likelihood function.
$ \sum (y_i-\mu)^2 = \sum (y_i - \bar{y} + \bar{y} - \mu)^2 = \sum [(y_i-\bar{y})^2 + 2(y_i-\bar{y})(\bar{y}-\mu) + (\bar{y}-\mu)^2] = \sum(y_i-\bar{y})^2 + N(\bar{y}-\mu)^2 $
$P(\vec{y}|\mu, \sigma^2) = (2\pi\sigma^2)^{-N/2} e^{\frac{-(N-1)S^2}{2\sigma^2}} e^{-\frac{N}{2\sigma^2}(\bar{y}-\mu)^2}$
def: where $\left\{ \begin{array}{l} \bar{y} = \frac{1}{N}\sum Y_i \rightarrow \text{simple mean} \\ S^2 = \frac{1}{N-1}\sum(y_i-\bar{y})^2 \end{array} \right.$ sufficient statistics
> "sample variance" (unbiased) estimator for $\sigma^2$ is $S^2$.

Case 1: $\sigma^2=1$ is known, $P(\mu)\propto 1$
> improper prior
Bayes' theorem $P(\mu|\vec{y}) \propto P(\vec{y}|\mu) P(\mu)$
(ignore constant terms)
$P(\mu|\vec{y}) \propto e^{-\frac{N}{2\sigma^2}(\bar{y}-\mu)^2}$ (unnormalised posterior)
> notice looks like part of a gaussian, only part gives variance is so this must define probability dist over $\mu$.
> integrate over $\mu$
could:
$P(\mu|\vec{y}) = A e^{-\frac{N}{2\sigma^2}(\bar{y}-\mu)^2} \rightarrow \int d\mu \rightarrow \text{Find } A$
> require $\int P d\mu = 1$
or:
$P(\mu|\vec{y}) = \frac{1}{\sqrt{2\pi(\sigma^2/N)}} e^{-\frac{1}{2\sigma^2/N}(\mu-\bar{y})^2} = N(\mu|\bar{y}, \sigma^2/N)$
> normalised posterior
> gaussian on $\mu$ tells us what degree of belief we have in various values of $\mu$
> 68% prob $\mu$ lies between $\bar{y}\pm\sigma/\sqrt{N}$

[Diagram of a Gaussian posterior distribution for μ, centered at $\bar{y}$. The area corresponding to 68% posterior probability is shaded, spanning $\bar{y} \pm \sigma/\sqrt{N}$.]
> integrates within these limits gives 0.68
> "68% credible interval"

> see slide: frequentist vs. bayes for more detailed comparison
> written overlays?

---
(Page 49)

Reading: Sivia Ch 1-3 Gelman "Bayesian data analysis"
F & B 3.8
Ivezic 5

## Lecture 12
_19.2.25_

(slides)

### Frequentist vs. Bayes

* Frequentists make statements about the data (or statistics or estimators = functions of the data) conditional on the parameter
$P(D|\theta)$ or $P(f(D)|\theta)$

* Often goal is to get a "point estimate" or confidence intervals with good properties under repeated experiments $\rightarrow$ ("long run")

* Arguments are based on datasets that could've happened but didn't e.g. null hypothesis testing

* Bayesians make statements about the probability of parameters conditional on the dataset $D=D_{obs}$ that you actually observed:
$P(\theta|D=D_{obs})$
This requires an interpretation of the probability as quantifying a "degree of belief" in a hypothesis

* Bayesian answer is the full posterior density $P(\theta|D=D_{obs})$ quantifying the "state of knowledge" after seeing the data. Any numerical estimates are attempts to (imperfectly) summarise the posterior.

---
(Page 50)
---
(Page 51)
## Bayes Advantages (slides)
* Ability to include prior information P(θ)
    - Incorporate info. from external datasets. P(θ) is the posterior from some other data P(θ|D_ext). E.g. cosmology. diff datasets give diff ways to probe cosmological parameters -> encode external posterior from one analysis as prior in another. E.g. CMB analysis -> constraints -> to incorporate information since one parameter often has multiple observables in cosmology.
* Regularisation: penalises overfitting with complex model. e.g. gaussian process prior
    > punishment term for complex model
    - e.g. linear regression, ridge regression is linear regression with a penalty term in likelihood. Can be interpreted with priors. e.g. can rule out. Think large slopes will occur and can incorporate that into prior. Can also penalise complex models -> Occam's razor.
    - It's not flat. > e.g. weights using L_1 norm, L_2 norm etc.
* "Noninformative"/weakly informative/default priors when you don't have much prior information.
    - e.g. might only be able to guess param is between 0 and 1 but still information! -> cannot have prior without some information, but reminder: definition of likelihood is the probability of data given param. P(D|θ)
* Likelihood (you're plugged in observed dataset and are viewing that as a fn. of the parameters) is not a probability density in the parameters. But multiply by a prior (even flat), and the (normalised) posterior is a probability density, conditional/marginal probabilities can be computed.
    (e.g., in multi-parameter cases, useful to derive conditional densities 'example later').
* Ability to deal with high dimensional parameter space e.g. latent variables or nuisance parameters and marginalise them out (analytically/numerically) e.g. posterior mode or MLE is called maximum a posteriori -> can find peak.
* Note: Bayesian inference not necessarily completely opposed to frequentist statistics. Estimators derived from Bayesian arguments can still be evaluated in a frequentist basis (e.g. James-Stein estimators)
    > (a technique to generate an estimator)
    > (could call this as intervals that contain some positive prob. of containing θ).

---
(Page 52)
> (Last time: gaussian example of bayesian inference (1D, flat prior -> like "hello world" of bayesian inference)
> recap: now slightly more involved example.

### Simple Gaussian Example w/ conjugate prior (board)
$Y_i \stackrel{iid}{\sim} N(\mu, \sigma^2)$ ($ \mu $ unknown, $ \sigma^2 $ known).

Likelihood: $ P(\vec{y}|\mu, \sigma^2) = \prod_{i=1}^{N} N(y_i|\mu, \sigma^2) = \dots $
$= (2\pi \sigma^2)^{-N/2} e^{-\frac{(N-1)S^2}{2\sigma^2}} e^{-\frac{N}{2\sigma^2}(\bar{y}-\mu)^2} $

Defined sufficient statistics:
> all info in data is contained in the likelihood & likelihood depends on sufficient statistics. meaning if two diff. y vectors with same N & $\bar{y}$, S yield same info about parameters. even if y vector is different.

$\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$ "sample mean"

$S^2 = \frac{1}{N-1} \sum_{i=1}^N (y_i - \bar{y})^2$ "sample variance"

Conjugate Prior: $ P(\mu) = N(\mu|\mu_0, \tau_0^2) $
(equivalently can write $ \mu \sim N(\mu_0, \tau_0^2) $) $ \leftarrow \text{prior mean} \leftarrow \text{prior variance} $
$ P(\mu) = \frac{1}{\tau_0 \sqrt{2\pi}} e^{-\frac{1}{2\tau_0^2}(\mu - \mu_0)^2} $

Posterior: $ P(\mu|\vec{y}) \propto P(\vec{y}|\mu) P(\mu) $
> Only stuff w. $\mu$ in will matter.
$ \propto e^{-\frac{N}{2\sigma^2}(\bar{y}-\mu)^2} e^{-\frac{1}{2\tau_0^2}(\mu-\mu_0)^2} $
> Prior: relates to our degree of belief in parameters (a conceptual point)
> will be normally const, so don't need to do integration.
> let's get robust result. will actually be a gauss.

[warm up ex.] - product of gaussian densities
$ P(\mu|\vec{y}) = N(\mu|\mu_N, \tau_N^2) $
$ \qquad \uparrow \qquad\qquad \uparrow $
posterior mean posterior variance

---
(Page 53)
> (from up sheet 1)
reminder: when you have product of gaussian densities, the precision = $\frac{1}{\text{variance}}$ of the resulting gaussian = sum of precision of individual gaussians.

$$ \left[ \frac{1}{\tau_N^2} = \frac{1}{\tau_0^2} + \frac{N}{\sigma^2} \right] \qquad \text{POSTERIOR PRECISION} $$
> e.g. have $N(\mu_1, \sigma_1^2), N(\mu_2, \sigma_2^2)$
> add them: $N(\mu, \sigma^2)$
> multiply them: $N(\mu_{12}, \sigma_{12}^2), \sigma_{12}^{-2} = \sigma_1^{-2} + \sigma_2^{-2}$

Posterior precision = sum of precisions from likelihood & prior.

reminder:
$$ \left[ \mu_N = \frac{\frac{1}{\tau_0^2}\mu_0 + \frac{N}{\sigma^2}\bar{y}}{\frac{1}{\tau_0^2} + \frac{N}{\sigma^2}} \right] \qquad \text{POSTERIOR MEAN} $$
> (divide by weights to get mean)

Posterior mean = precision weighted average of means from likelihood & prior.
> These formulas explicitly tell us how prior information & likelihood are combined to give us the bayesian inference.
> (mean & variance)

For fixed prior $\tau_0$, as $N \rightarrow \text{large}$
> send $N \rightarrow \infty$, second term in $\tau_N^2$ goes to $\infty$. $\tau_N^2 \rightarrow 0$?
$\mu_N = \bar{y}$
$\tau_N^2 \rightarrow \sigma^2/N$
> ($P(\mu|\vec{y}) = N(\mu|\bar{y}, \frac{\sigma^2}{N} + \frac{\sigma^2}{\tau_0^2 \dots})$
(Data $N \rightarrow$ large, likelihood dominates over prior.)
> basically after a lot of data, we get the same result as flat prior.

Side note: What is conjugate prior?
for a given likelihood, if you combine it with a conjugate prior coming from "nice" distribution then the posterior is guaranteed to come from the same class of "nice" distributions. (e.g. gaussian, wishart, laplace)

---
(Page 54)
### Astrophysics Example (slide)
#### BAYESIAN INFERENCE FOR PARALLAX
PAPER: C. Bailer Jones
Estimating distances from parallaxes
- parallax is a way to measure distances
$ \varpi = \frac{\text{parsec}}{r} \qquad $ (apparent position)
$ \text{arcsec} $
$ \varpi_{\text{true}} = 1/r \qquad $ (in units of parsec)
[Diagram showing the Earth at two points in its orbit around the Sun, observing a distant star against a background of even more distant stars. The angle subtended by the Earth's orbit as seen from the star is the parallax angle.]
> earth
> star
> path of late-time

- Gaia satellite makes parallax measurement to measure distances & map stars in our galaxy.
- measurement error in parallax angle. measured parallax is different to true parallax.
$ P(\tilde\omega|r) = \frac{1}{\sigma_{\tilde\omega}\sqrt{2\pi}} \exp\left[-\frac{1}{2\sigma_{\tilde\omega}^2}(\tilde\omega - \frac{1}{r})^2\right] \qquad \sigma_{\tilde\omega} > 0 $
- measured parallax has some distribution: assume gaussian distribution for simplicity
- fractional measurement error $f = \sigma_{\tilde\omega}/\tilde\omega$
> just another way to express error, no more info than $\sigma$
$ r=10 \text{pc}: $
[Plot of a probability distribution P(ω|r). The x-axis is ω, centered at 0.1. There are two Gaussian curves shown. The wider, shorter one is labeled "ω_observed = 0.2, σ_ω = 0.05, f=0.5, less precise instrument". The narrower, taller one is labeled "σ_ω = 0.02, f=0.2, more precise instrument".]
> true value of actual parallax could be anywhere in distribution.
- possible to get negative parallax due to measurement error.

Likelihood
$ P(\tilde\omega|r) $
[Plot of likelihood P(ῶ|r) vs r. The x-axis is r (true distance). The y-axis is Likelihood. Three curves are shown, corresponding to different observed parallaxes ῶ. A dashed blue curve peaks at a low r and is labeled "upper likelihood". A solid blue curve peaks at a higher r and is labeled "true likelihood". A solid purple curve is shown to the right of the blue ones, labeled "lower likelihood".]
> get skewed distribution of likelihoods.
> true value

---
(Page 55)
> what if ῶ < 0? large error? then peak is -ve
- Laplace approximation gives gaussian, incidentally this is mathematically equivalent to doing propagation of error (assume gaussian measurement error propagated to $r=1/\tilde\omega$...)
- this is an approximation. (dotted vs solid line).
> no peak when ῶ < 0
[Plot of likelihood P(ῶ|r) vs r. x-axis from -30 to 30. y-axis is likelihood. Several curves are shown for different values of ῶ. For ῶ=0.1, the curve is a peak in the positive r region. For ῶ=0.05, the peak is further out. For ῶ=0.01, the peak is even further out. For ῶ=-0.01, there is no peak in the positive r region, the curve rises as r approaches 0 from the positive side. A dashed curve shows the symmetric negative part for ῶ=0.05. A label "Futility function" is near the peak of ῶ=0.05. A note says "Likelihood for seeing return a positive parallax given that no prior is used. Just reality."]
- likelihood is positive on negative values of distance (unphysical) -> see all values of mapping $r \rightarrow 1/r$
- negative measurements have no mode (MLE).
> We peak - most likely peak for the but not for -ve distance.

they try to fix this:
- impose our knowledge that we know distances should be positive
$ P(r|\tilde\omega) \propto P(\tilde\omega|r) P(r) $
$ P(r) = \begin{cases} 1 & r>0 \\ 0 & \text{otherwise} \end{cases} $
> note: improper distribution, not normalisable
- this cuts off -ve values of r in our graph
[Two plots are shown. Left plot: P(r|ῶ) vs r. Curves for ῶ=0.01, δ=0.2, δ=0.5 are shown, all peaking at positive r and going to zero at r=0. The y-axis is labeled P(r|ῶ) (unnormalized posterior). Right plot: P(r|ῶ) vs r. Two curves for ῶ=-0.01 and σ=0.02 are shown. One is the likelihood which is non-zero for r<0. The other is the posterior P(r|ῶ) which is a cumulative-like curve for r>0. Note says "for -ve parallaxes, not getting a peak at all -> no mode".]
- improper posterior: not-normalisable no mode, variance etc.
- mode ($r=1/\tilde\omega$) exists for +ve $\tilde\omega$ but undefined for -ve $\tilde\omega$. ($r>0$)
> still have some problems

---
(Page 56)
- impose a proper distance prior (impose limit to max distance of a star in our survey)
$ P(r) = \begin{cases} 1/r_{lim} & 0 < r \le r_{lim} \\ 0 & \text{otherwise} \end{cases} $

$ P(r|\tilde\omega) \propto P(\tilde\omega|r)P(r) $
$ P^*_{\mu}(r|\tilde\omega, \sigma_{\tilde\omega}) = \begin{cases} (1/r_{lim}) P(\tilde\omega|r, \sigma_{\tilde\omega}) & 0 < r \le r_{lim} \\ 0 & \text{otherwise} \end{cases} $ > unnormalized posterior

[Plot of P(r|ῶ) vs r. The x-axis goes up to r_lim=100. Three curves are shown, for ῶ=0.01, ῶ=0.025, and ῶ=0.05. All curves are cut off at r=100. A dashed line shows the continuation of the ῶ=0.01 curve.]
> note: cut off at r_lim
- implies unrealistic stellar density ~ $1/r^2$
- cut off of mode at edge $r_{lim}$ for small or negative measured parallax ($\tilde\omega < 0$) (but better than going to infinity)
$ \text{rest} = \begin{cases} 1/\tilde\omega & 0 < 1/\tilde\omega \le r_{lim} \\ r_{lim} & 1/\tilde\omega > r_{lim} \\ \text{undefined} & \tilde\omega \le 0 \end{cases} \quad \text{(extreme mode)} $

- still have some problems -> can go towards more & more astrophysically motivated priors.
- problem w/ previous prior: volume density $p(r)$.
$ P_r(\text{distance} \in [r, r+dr]) \propto p(r) 4\pi r^2 dr $
> unphysical ->
$ p(r) \propto 1/r^2, r < r_{lim} \Rightarrow p(r) \propto \text{const}. \quad r < r_{lim} $
$ \qquad 0 \text{ otherwise} \qquad\qquad\qquad 0 \text{ otherwise} $
- more physical: copernican principle $p(r) \propto \text{const}. \Rightarrow p(r) \propto \text{const} \cdot 4\pi r^2 $
> sounds reasonable.
> see slides CBJ, some principle as previous priors

---
(Page 57)
## Lecture 13
_21.2.25_
> see slides for reading refs
- astrophysics example
recap: understanding how priors impact inference
- keeps going as before: introduce more & more astrophysically motivated priors. (see slides)

### Conclude: Priors in Bayesian Inference
* Priors can be used to encode background info./external knowledge about parameters.
    - weak mathematical constraints on physical parameters e.g. positivity of distances
    - astrophysical info. e.g. distributions of stars
* Test sensitivity of your inference to the priors (& likelihood). Under various assumptions of the model.
> Note: also depends on your measurement error & precision. not just prior.
$ \underset{\text{likelihood}}{P(\tilde\omega|r)} - \underset{\text{measured}}{P(r|\tilde\omega)} \propto P(\tilde\omega|r) \underset{\text{prior}}{P(r)} - \text{posterior} $

---
(Page 58)
### MULTIPARAMETER BAYESIAN MODELS (slides)
Example: Gelman BDA 3.2 sheet 2
> [Analytic posterior for gaussian model with non-informative prior]
data generating process: $ y_i \stackrel{iid}{\sim} N(\mu, \sigma^2) \quad i=1,...,n $
likelihood: $ p(\vec{y}|\mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{(n-1)s^2}{2\sigma^2}\right) \exp\left(-\frac{n}{2\sigma^2}(\bar{y}-\mu)^2\right) $

sufficient statistics
(non informative) $ p(\mu) \propto 1 $
> independent of ordering of meas.
improper prior: $ p(\log \sigma^2) \propto 1 $ or $ p(\sigma^2) \propto \sigma^{-2} \quad (\sigma^2 > 0) $ JEFFREYS' PRIOR
> even if prior improper, need proper posterior (corresponds to real observations)
joint posterior: $ p(\mu, \sigma^2 | \vec{y}) \propto p(\vec{y}|\mu, \sigma^2) \times p(\mu, \sigma^2) $
$ p(\mu, \sigma^2|\vec{y}) \propto (\sigma^2)^{-n/2-1} \exp\left(-\frac{(n-1)s^2}{2\sigma^2}\right) \exp\left(-\frac{n}{2\sigma^2}(\bar{y}-\mu)^2\right) $
> equivalent, can say: $ P(\sigma^2|\vec{y}) \frac{P(\mu|\sigma^2, \vec{y})}{P(\sigma^2)} \dots $
> joint is useful to find conditional/marginals:
conditional posterior: $ p(\mu|\sigma^2, \vec{y}) = N(\mu|\bar{y}, \sigma^2/n) $
> ex. sheet 2
marginal posteriors: $ p(\sigma^2|\vec{y}) = \int p(\mu, \sigma^2|\vec{y}) d\mu = \text{Inv-}\chi^2(\sigma^2|n-1, s^2) $
$ p(\mu|\vec{y}) = \int p(\mu, \sigma^2|\vec{y}) d\sigma^2 $
> can derive these analytically so useful
$ \propto \left[ 1 + \frac{n(\mu-\bar{y})^2}{(n-1)s^2} \right]^{-n/2} $
$ = t_{n-1}(\mu|\bar{y}, s^2/n) \qquad t \text{ distribution} $

Note: (1) inverse chi-squared distribution
$ \text{Inv-}\chi^2(\theta|\nu, s^2) \propto \theta^{-(\nu/2+1)} \exp\left(-\frac{\nu s^2}{2\theta}\right), \quad \theta>0 $
deg. of freedom $ \nu > 0 $
scale parameter $ s > 0 $
[A sketch of a right-skewed distribution, peaking near zero and having a long tail to the right.]

---
(Page 59)
(2) Same as inverse-gamma
$ \text{Inv-gamma}(\theta|\alpha, \beta) \propto \theta^{-(\alpha+1)} \exp(-\beta/\theta), \quad \theta > 0 $
shape parameter $ \alpha = \nu_0/2 $
scale parameter $ \beta = \nu_0\sigma_0^2/2 $
> useful named distributions have random number generators in common python packages

(3) Student's t distribution
$ t_{\nu}(\theta|\mu, \sigma^2) \propto \left[1 + \frac{1}{\nu} \left(\frac{\theta-\mu}{\sigma}\right)^2\right]^{-(\nu+1)/2} $
deg. of freedom $ \nu > 0 $
location parameter $ \mu $
scale parameter $ \sigma > 0 $
> see these distributions in book i.e. Gelman BDA appendix or just wikipedia for your example sheet
[A sketch of a bell-shaped curve, similar to a Gaussian but with potentially heavier tails.]

What if you can't compute marginals/expectations analytically? -> Bayesian Computation
* Bayesian answer: full posterior $P(\theta|D)$, numerical estimates are attempts to (imperfectly) summarise the posterior e.g. mean, mode.
* Often these are posterior expectations e.g. $E[f(\theta)|D] = \int f(\theta) P(\theta|D) d\theta$ (computationally difficult)
* Bayesian computation: algorithms to "map out"/sample the posterior $P(\theta|D)$ and compute expectations $E[f(\theta)|D]$.
* e.g. MCMC, nested sampling, importance sampling.
> "all models are wrong, some are useful. Even if your computation is 'right'!"

---
(Page 60)
> object of computation is to calculate some integral (posterior expectations) -> can do with monte carlo integration
### MONTE CARLO INTEGRATION (slides)
Typically we want to summarize the posterior & compute expectations of the form
$ I = E[f(\theta)|D] = \int f(\theta) P(\theta|D) d\theta $

Using m samples from the posterior
$ \theta_i \sim P(\theta|D) $
$ \hat{I} = \frac{1}{m} \sum_{i=1}^m f(\theta_i) \rightarrow I $ (LLN for large N) CLT

Monte Carlo Error (derive later on next page)
$ \text{Var}[\hat{I}] = \frac{1}{m^2} \sum_{i=1}^m \text{Var}[f(\theta_i)] = \frac{1}{m} \text{Var}[f(\theta)] \propto \frac{1}{\sqrt{m}} \text{Var}[E_f(\theta_i)] $
'Monte carlo error ~ $1/\sqrt{m}$ indep of dim[$\theta$] - is convenient'

Fundamental Theorem of Monte Carlo -> Bayesian computation "using sampling"
$ \underset{\text{posterior expectation}}{E[f(\theta)|D]} = \int f(\theta) P(\theta|D) d\theta \approx \frac{1}{m} \sum_{i=1}^m f(\theta_i) $
$ \qquad \uparrow \qquad\qquad \text{sample average} $

E.g.
Posterior mean $\mu \qquad f(\theta) = \theta $
> reporting this as if it is Maximum?
Posterior variance $ \qquad f(\theta) = (\theta-\mu)^2 $
Probability in an interval [a,b] $ \qquad f(\theta) = I_{[a,b]}(\theta) $
(indicator function)
= one when value between a & b.

---
(Page 61)
### Monte Carlo Error (board)
suppose: $\theta$ is some invertible fn. of $\theta$
$ \theta_i \stackrel{iid}{\sim} P(\theta) \qquad i=1,...,m $
> monte carlo sample size $m$ is not same as data sample size $n$.
$ \rightarrow f(\theta_i) $ are iid
Def: $ \hat{I} = \frac{1}{m} \sum_{i=1}^m f(\theta_i) $
$ \text{Var}(\hat{I}) = \text{Cov}(\hat{I}, \hat{I}) = \text{Cov}\left[\frac{1}{m}\sum_i f(\theta_i), \frac{1}{m}\sum_j f(\theta_j)\right] $
$ = \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m \text{Cov}[f(\theta_i), f(\theta_j)] $
> $\theta_i, \theta_j$ are iid -> only non zero contributions when $i=j$
$ = \frac{1}{m^2} \sum_{i=1}^m \text{Var}(f(\theta_i)) $
$ = \frac{1}{m} \text{Var}(f(\theta_i)) $
> standard deviation of monte carlo error
> std MC error $\propto \frac{1}{\sqrt{m}}$
GLT: $ m \rightarrow \text{large} \quad \hat{I} \sim N(E[f(\theta)], \text{Var}(f(\theta))/m) $

approximate variance
$ \text{Var}(\hat{I}) \approx \frac{1}{m} \widehat{\text{Var}}(\{f(\theta_i)\}) $
> slow convergence (accuracy of MC error)
> but! -> nice: indep. of dim($\theta$)
> so MC error rate is indep of dimension. oooft!
MC sample variance
$ \widehat{\text{Var}}[\{f(\theta_i)\}] = \frac{1}{m-1} \sum_{i=1}^m (f(\theta_i) - \hat{I})^2 $
> MC error scales as $m^{-1/2}$
> slow but indep of dim($\theta$)!
> an approx.

---
(Page 62)
## Lecture 14
_24.2.25_
> note on useful papers of gaussians on moodle, & for ex sheet 2.
Today: Bayesian Computation, Direct sampling, Importance Sampling

### Monte carlo direct sampling (slides)
> e.g. previous example is gaussian-gaussian model with non-informative prior.
How does one sample from joint? -> it factorises.
Factorise posterior: $ P(\mu, \sigma|y) = P(\mu|\sigma^2, y) P(\sigma^2|y) \leftarrow \text{Factorises!} $
1. $ \sigma_i^2 \sim P(\sigma^2|y) \qquad \text{Inv-}\chi^2 $
2. $ \mu_i|\sigma_i^2 \sim P(\mu|\sigma_i^2, y) \qquad \text{Normal} \rightarrow (\mu_i, \sigma_i^2) \sim P(\mu, \sigma^2|y) $
> from posterior
> Joint-draw from posterior!

[A diagram with two panels. The left panel shows a 2D scatter plot labeled "Joint". The x-axis is μ, and the y-axis is σ². A cloud of blue dots represents the joint samples. Below this is a histogram labeled "Marginal" for μ. To the right is a histogram labeled "Marginal" for σ². Notes are added to the diagram.]
> e.g. $\bar{y}=0$, $s^2=1$, $n=10$. Arbitrary choice of sufficient statistics.
> reminder, summarizing data into sufficient statistics
> Compute posterior summaries from monte carlo.
> $E[\mu|y]=1.10$.
> $Std[\sigma|y]=0.25$
> $E[\sigma|y]=0.00$.
> $Std[\mu|y]=0.14$.

(did not need to solve analytically to get marginals)

### Kernel density estimate = estimate a smooth density from samples.
[A diagram shows a histogram made of many thin vertical blue lines (representing samples), with a smooth, bell-shaped curve drawn over it. The curve is thicker and more pronounced in the center.]
> can think of it as "a smoothed histogram".

---
(Page 63)
### KERNEL DENSITY ESTIMATION (board)
$ \theta_i \stackrel{iid}{\sim} P(\theta|D) \qquad i=1,...,m $
Approx. $ P(\theta|D) $ using $ \hat{P}(\theta|D) \approx \frac{1}{m} \sum_{i=1}^m N(\theta|\theta_i, bw^2) $
$ \int \hat{P}(\theta|D) d\theta = 1 \quad \checkmark $
> nice rule of thumb.
> select bandwidth
Lots of ways to choose bandwidth, nice one is
Silverman's Rule of Thumb:
$ bw = \left(\frac{4\hat\sigma^5}{3m}\right)^{1/5} \quad \hat\sigma^2 = \text{Var}(\theta|D) \approx \widehat{\text{Var}}(\{\theta_i\}) $
> sample variance $ \frac{1}{m-1}\sum(\theta_i - \bar{\theta})^2?? $

### SUMMARISING POSTERIOR UNCERTAINTIES
* Can compute posterior mean (or median) $\pm$ posterior standard deviation.
* If posterior $P(\theta|D)$ is Gaussian, then this contains 68% posterior probability
* But not necessarily for non-gaussian posterior!

### CREDIBLE INTERVALS
> on Bayesian land, talk about "credible intervals"
> prob that $\theta$ true is in interval $L(\vec{x}), U(\vec{x})$ is $1-\alpha$ given that we know data D.
[Two plots of a skewed posterior distribution P(θ|D) vs θ are shown.
Left plot: A "68% central credible interval" is marked. The area under the curve between the 16% quantile and the 84% quantile is shaded. The tails, each containing 16% of the probability, are also indicated. A note says "NOT UNIQUE!".
Right plot: Another interval is shaded, covering 68% of the probability mass, but it is shifted to the right, starting from a low percentile and ending at a higher one. The area is labeled "68%".]

---
(Page 64)
### (HPD) HIGHEST POSTERIOR DENSITY CREDIBLE INTERVAL (board)
- Unique
- Narrowest interval that contains X% posterior probability

[A plot of a unimodal posterior distribution P(θ|D) vs θ. A horizontal dashed line cuts through the peak. The interval on the θ-axis where the curve is above this line, `[l(ρ), u(ρ)]`, is marked. Two lower horizontal lines for ρ₁ and ρ₂ are also shown with their corresponding intervals.]

Find $\rho$ s.t. `[l(ρ), u(ρ)]` contains X% posterior probability.

HPD works for multi-modal posteriors!
[A plot of a bimodal posterior distribution P(θ|D) vs θ. A horizontal dashed line at height ρ cuts through both peaks. The regions under the curve and above the line are shaded. The total shaded area corresponds to the HPD credible interval.]

X% HPD-C.I. = set $\Theta$ (capital $\theta$) (possibly disconnected) with highest $\rho$ s.t.
$ \forall \theta \in \Theta, P(\theta|D) > \rho \text{ and } \int_{\Theta} P(\theta|D) = X $

---
(Page 65)
[ What if you can't directly sample the posterior $\theta_i \sim P(\theta|D)$? (board) ]
$ E[f(\theta)|D] = \int f(\theta) P(\theta|D) d\theta \approx \frac{1}{m}\sum_{i=1}^m f(\theta_i) $

e.g. likelihood might be output of some computer program
* Importance sampling:
draw from an easier "tractable" distribution (importance function) $\theta_i \sim Q(\theta)$ and weight the samples by $w_i = P(\theta_i|D)/Q(\theta_i)$ to compute expectations
* Posterior simulation:
MCMC, Nested sampling etc. generates draws from the posterior density iteratively in long-run
$ p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)} $

---
(Page 66)
### IMPORTANCE SAMPLING (board)
Given a probability density $P(\theta)$ [e.g. the posterior]
Want to estimate
$ I = E_P[f(\theta)] = \int f(\theta)P(\theta)d\theta $

However, $P(\theta)$ is intractable i.e. difficult to directly sample, e.g. $\theta_i \sim P(\theta)$, but can evaluate $P(\theta_i)$ for a parameter value $\theta_i$.

Select an instrumental distribution (importance function) from which it is easy to draw samples!
$ \theta_i \sim Q(\theta) \rightarrow \text{tractable, can easily draw/sample} \rightarrow \text{evaluate} $
and can evaluate $Q(\theta_i)$, and $Q(\theta)>0$ wherever $P(\theta)>0$.

Rewrite $ I = E_P[f(\theta)] = \int f(\theta) \frac{P(\theta)}{Q(\theta)} Q(\theta) d\theta $
$ = E_Q\left[f(\theta) \frac{P(\theta)}{Q(\theta)}\right] $

Draw samples $ \theta_i \stackrel{iid}{\sim} Q(\theta) \quad (i=1,...,m) $
Approximate I with importance sampling estimate:
$ \hat{I}^* = \frac{1}{m} \sum_{i=1}^m f(\theta_i) \frac{P(\theta_i)}{Q(\theta_i)} = \frac{1}{m} \sum_{i=1}^m f(\theta_i) w_i^* $
where $ w_i^* = w^*(\theta_i) = \text{importance weights} = \frac{P(\theta_i)}{Q(\theta_i)} $

---
(Page 67)
$ E_Q[\hat{I}^*] = \frac{1}{m} \sum_{i=1}^m E_Q[f(\theta_i)w^*(\theta_i)] $
$ = \frac{1}{m} \sum_{i=1}^m E_Q\left[ f(\theta_i) \frac{P(\theta_i)}{Q(\theta_i)} \right] $
$ = \frac{1}{m} \sum_{i=1}^m \int f(\theta) \frac{P(\theta)}{Q(\theta)} Q(\theta) d\theta $
$ = \frac{1}{m} \sum_{i=1}^m \int f(\theta) P(\theta) d\theta $
$ = \frac{1}{m} \sum_{i=1}^m E_P[f(\theta)] = E_P[f(\theta)] $
$ = I \quad (\text{unbiased}) $

> reminder: Var(aX+bY) = a²VarX + b²VarY
$ \text{Var}_Q[\hat{I}^*] = \frac{1}{m^2} \sum_{i=1}^m \text{Var}(f(\theta)w^*(\theta)) $
$ = \frac{1}{m} \text{Var}(f(\theta)w^*(\theta)) $
$ \approx \frac{1}{m} \widehat{\text{Var}}(\{f(\theta_i)w^*(\theta_i)\}) $
$ \widehat{\text{Var}}[\{ \cdot \}] = \frac{1}{m-1} \sum \left(f(\theta_i)w^*(\theta_i) - \hat{I}^*\right)^2 $

---
(Page 68)
## Lecture 15
_26.2.25_
> useful papers & gaussian's identities on moodle for sheet 2
Today: case study "Importance sampling the mass of the milky way galaxy"
Continuing: Bayesian computation, importance sampling, MCMC. (Gelman BDA ch10-12)

> Recap: MC integration, want to summarize posterior, estimate w/ samples from the posterior. Replace integral w/ MC sum, which converges to desired expectation for large N. Error on estimate $\propto 1/\sqrt{N}$ and is indep of dim of parameter space!

### Importance sampling recap (slides)
Objective: compute expectation w.r.t. distribution $P(\theta)$
$ I = E_f[\theta] = \int f(\theta)p(\theta)d\theta $

Example: Posterior (suppress "|D" here.)
$ P(\theta) = \frac{L(\theta)\pi(\theta)}{\int L(\theta)\pi(\theta)d\theta} \qquad \frac{\text{likelihood}}{\text{prior}} $
Can evaluate $p(\theta)$ but not sample from it directly. Choose importance function $Q(\theta)$ you can evaluate and sample!
> named distribution
> e.g. normal, student's t

Importance sampling estimate
$ \theta_1, ..., \theta_m \stackrel{iid}{\sim} Q(\theta) $
$$ \hat{I}^* = \frac{1}{m} \sum_{i=1}^m f(\theta_i) w^*(\theta_i) $$
$$ w_i^* = w^*(\theta_i) = P(\theta_i)/Q(\theta_i) $$

---
(Page 69)
### SELF-NORMALISED IMPORTANCE SAMPLING (slides)
Objective: compute expectation
$ I = E_f[\theta] = \int f(\theta)P(\theta)d\theta $
but can only evaluate unnormalised $\tilde{P}(\theta)$.

Example: unnormalised posterior
$ \tilde{P}(\theta) = L(\theta)\pi(\theta) $.
> (evidence, hard to evaluate)
Normalised posterior is $ P(\theta) = \frac{\tilde{P}(\theta)}{Z_p} = \frac{\tilde{P}(\theta)}{\int \tilde{P}(\theta)d\theta} $,
but, cannot easily calculate
$ Z_p = \int L(\theta)\pi(\theta)d\theta $ so cannot evaluate $P(\theta)$.

---
(Page 70)
### Self-Normalised Importance Weighting (board)
Often in Bayesian Analysis, you can only compute/evaluate posterior $P(\theta)$ up to a constant.
$ P(\theta) = \frac{L(\theta)\pi(\theta)}{Z_p} = \frac{\tilde{P}(\theta)}{Z_p} \leftarrow \text{unnormalised} $
$ \qquad \uparrow \qquad Z_p = \int \tilde{P}(\theta)d\theta $
normalised

GOAL: Estimate $I = E_p[f(\theta)] = \int f(\theta) P(\theta)d\theta $
$ = \int f(\theta) \frac{\tilde{P}(\theta)}{Z_p} d\theta $
$ = \frac{\int f(\theta) \frac{\tilde{P}(\theta)}{Q(\theta)} Q(\theta)d\theta}{\int \frac{\tilde{P}(\theta)}{Q(\theta)} Q(\theta)d\theta} $
> if you can't solve a hard problem, solve an easier problem.

instrumental distribution -> probably named distribution
Choose Q s.t. drawing samples $\theta_i \stackrel{iid}{\sim} Q(\theta)$ is easy $(i=1,...,m)$.
> don't need to know normalization of Q either!
Approximate I with
$ I \approx \hat{I} = \frac{\sum_{i=1}^m f(\theta_i) \tilde{w}(\theta_i)}{\sum_{i=1}^m \tilde{w}(\theta_i)} = \sum_{i=1}^m f(\theta_i) w(\theta_i) $ (FTMC)
(Estimate normalisation using same samples drawn from Q)

where the self-normalised weights are,
$ w(\theta_i) = \frac{\tilde{w}(\theta_i)}{\sum_{j=1}^m \tilde{w}(\theta_j)} = \frac{\tilde{P}(\theta_i)/Q(\theta_i)}{\sum_{j=1}^m \tilde{P}(\theta_j)/Q(\theta_j)} $
> nice, works. when $P$ is unnormalised... we have done the bottom integral using importance sampling itself.

---
(Page 71)
(Also works if Q is unnormalised, since const. factor in Q cancel above.)
> whereas importance sampling we looked at before required Q to be proper i.e. normalised.

COST: $\hat{I}$ is slightly biased
$ E[\hat{I}] = I + O(1/m) $
b/c of estimating denominator (walk out yourself)
> biased but consistent? -> as $m \to \infty$, $P(\hat{I}=I) \to 1$

> 'bias arises from non-domination' process
reminder: $E[AB] = E[A]E[B] + \text{Cov}(A,B)$
> since cov(A,B) = E[AB] - E[A]E[B]
'cheat GET attempt':
$ E_Q[\hat{I}] = \sum_i E_Q[f(\theta_i) w(\theta_i)] = \sum_i (E_Q[f(\theta_i)]E_Q[w(\theta_i)] + \text{Cov}(f_i, w_i)) $
> unclear steps
> since $\sum w_i = 1$?
> and $\text{Cov}(f_i, w_i)$ is ignored? O(1/m²)?
$ E_Q[w(\theta_i)] = 1/m + O(1/m^2) $
$ \rightarrow E_Q[\hat{I}] = \sum_i(E_Q[f_i]\theta^i(1/m + O(1/m^2))) $
$ = E_Q[f_i(\theta)] + O(1/m) $

> some kind of perturbation analogy/taylor expansion?

$ \text{Var}[\hat{I}] = \sum_i^m \text{Var}[f(\theta_i)w(\theta_i)] = \frac{1}{m} \text{Var}[f(\theta_i)w(\theta_i)] \qquad ? \text{.ie} $
$ \approx \frac{1}{m} \text{Var}(\{f_i(\theta_i)w_i(\theta_i)\}) \quad ? $

---
(Page 72)
### Contrived example: Gaussian Mixture, normalised PDF (slides)
#### IMPORTANCE SAMPLING EXAMPLE (not self normalised)
[A plot showing two curves on an x-axis. A blue, bimodal curve is labeled "target p(x)". A wider, purple, unimodal curve is labeled "impt. fn. Q(x) - we choose gaussian".]
> mixture of 2 gaussians
> target mean = 0.20
> (have analytic soln but suppose we didn't have this & try to estimate)

Generate 1000 samples from impt. fn.
[A plot of the importance weight function vs x. It shows a dashed line that is low and then rises to a peak where the target PDF has its main peak.]
[A histogram of m=1000 draws from the importance function is shown, roughly matching the shape of Q(x).]
MC estimate of mean
$ \sum w_i^* x_i = 0.215 \pm 0.028 $
> within $1\sigma$ of target

> could also do this for posterior variance etc.
m=10,000 draws: mc estimate of mean is
$ \sum x_i w_i^* = 0.199 \pm 0.009 $
> error does down as $1/\sqrt{m}$

---
(Page 73)
### Parallax Example (slides)
> gaussian measurement error in ῶ?
Likelihood $ \tilde\omega \sim N(1/r, \sigma_{\tilde\omega}^2) $
$ P(\tilde\omega|r) = \frac{1}{\sigma_{\tilde\omega}\sqrt{2\pi}} \exp \left[-\frac{1}{2\sigma_{\tilde\omega}^2}(\tilde\omega - 1/r)^2\right], \quad \sigma_{\tilde\omega} > 0 $
> parallax prop.
True relation (no errors): $ \omega = \frac{\text{parsec}}{r} \leftarrow \text{distance} $
> parallax angle
[Diagram showing the Earth orbiting the sun, observing a star. The parallax angle is defined. The distance to the background stars is effectively infinite.]
> since $\tilde\omega \ll 1$,
> $\tan \tilde\omega \approx \tilde\omega$ to v. good approx.
> parsec is 'parsec-second',
> r is defined as the parsec.

Introduce physically motivated prior
$ P(r) = \begin{cases} \frac{2}{L^3}r^2 e^{-r/L} & \text{if } r>0 \\ 0 & \text{otherwise} \end{cases} $
> as seen before, just some kind of it.
Exponential decrease in density of stars with galactic length scale L.
$ P(r|\tilde\omega) \propto P(\tilde\omega|r)P(r) $
Unnormalised posterior
$ P^*_{post}(r|\tilde\omega, \sigma_{\tilde\omega}) = \begin{cases} \frac{r^2 e^{-r/L}}{\sigma_{\tilde\omega}} \exp\left[-\frac{1}{2\sigma_{\tilde\omega}^2}(\tilde\omega - 1/r)^2\right] & r>0 \\ 0 & \text{otherwise} \end{cases} $
Can't compute normalisation analytically.
> (in reality this is a 1D example so we could actually evaluate this on a grid and do integral numerically, but let's pretend we can't do that)

---
(Page 74)
Want to compute posterior mean. consider posterior of distance
[Plot showing an unnormalized prior (a skewed curve peaking at low r) and two posteriors. One posterior for ῶ=-0.01 is a rising curve. The other posterior for ῶ=0.33 is a peaked curve labeled "unminimalised posterior". A note points to the ῶ=0.33 curve saying it is bimodal, higher signal. A note points to the ῶ=-0.01 curve saying it has a short-tail at low r, low s/n ratio.]
> what is the value of ῶ? we need to test
> e.g. two mock values for ῶ=-0.01. with prior ῶ=0.33, need to use self-normalised importance sampling, or need to choose impt. fn.
> ῶ=-0.33

[Plot showing the unnormalized posterior for ῶ=0.33 and ῶ=0.01. The y-axis is labeled unnormalised PDF. The curve for ῶ=0.33 is the "target posterior p(r|w)". A wider, smoother curve is the "impt. fn. Q(r)".]
Choose Q(r) that covers (is >) (target)
(is should be +ve anywhere the target is +ve.)
> choose, say, student's t-dist with v=3 (low deg of freedom)
> choose because tail of student-t is thicker than gaussian. (gaussian drops as $e^{-x^2}$ but student t as $x^{-(\nu+1)}$). have to scale it up more than the target if you have but we want a named distribution.

M=10⁴ draws from Q(r):
[Plot of the importance weight vs r. It is noisy and has large spikes.]
MC estimate of mean
$ \sum x_i w_i = 2391.263 \pm 3.918 $

M=10000 draws from impt. fn.
[Histogram of the number of draws vs distance r.]
> think of weights like adjusting histogram of Q(r) to match P(r|D).
> from wikipedia, a conceptual introduction to MCMC.
> Imagine approximately integral we want to do over a grid of points, say 10⁵ cycles in low-prob areas, the algorithm would terminate a lot to the true mean. It is reject-dominated so small no. of points is in high prob areas. The probability to do this is high so, yes we are in regions that don't add much -> equivalent to 'concept' of 'bores' sample weights everywhere. So small sample size.

> weights of medium variance -> not bad
> large variance in weights
> optimal goal is no variance in weights.
> idea of ESS: how many draws from posterior do we get that are effective from original grid of non-uniform samples, we want to get more effective samples from each grid. good is has low square-root of err, low confidence.

---
(Page 75)
### CHOOSING A GOOD IMPORTANCE FN. (slides)
> (most important part of importance sampling)
Within chsq named distributions.
* Can be shown (sheet 2) that theoretically optimal (minimum variance) importance fn. is
$ Q^*(\theta) = \frac{|f(\theta)|P(\theta)}{\int |f(\theta)|P(\theta) d\theta} $
> high variance in importance weights -> company, estimates -> small effective sample size.
* However, if we can't directly sample from $P(\theta)$, then we probably can't sample from $|f(\theta)|P(\theta)$.
* Want to keep the importance weights roughly const, otherwise large variations in $P(\theta)/Q(\theta)$ will lead to high variance of estimate, smaller ESS.
* Effective sample size
> 'importance fn similar shape to dist we are trying to sample'
$$ ESS = \frac{m}{1+\text{Var}[\{w^*(\theta_i)\}]} $$
> e.g. as in parallax example
* In practice, find thick-tailed distribution $Q(\theta)$ that is positive everywhere and similar in shape to $|f(\theta)|P(\theta)$.
* Don't want $Q(\theta)$ small when $|f(\theta)|P(\theta)$ large!
> problem: small sample size, high variance.

> -> ideal importance fn. is proportional to $|f(\theta)|P(\theta)$
> want fn that resembles target $P(\theta) \times f(\theta)$ (i.e. importance weights const -> small variance)
> weights -> large ESS.
> -> in practice want positive, everywhere, covers the tail (thicker tail), and... (misc)
> weight roughly skewed distribution.---
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
> use laplace approx, find MAP to estimate to sample posterior---
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
> find for this case its not obvious that this has 10 significant parameters---
(Page 126)

> remember
> we had a image shifted by a time delay
> each has indep. noise. How do we deal w. this using GP?

### Case study: Construct Bayesian Model

One latent fn. modelled as a draw from a GP
$f(t) \sim GP(c, k(t, t')) \quad k(t, t') = A^2 \exp[-(t-t')^2 / \tau^2]$
Data of each image are measurements of shifted copies of this latent function
$(\Delta t_i, \Delta m_i)$

For image $S1$ w/ observations $y_i$ at times $t_i$:
$y_i(t_i) = f(t_i) + \epsilon_i \quad \epsilon_{ij} \sim N(0, \sigma^2_{ij})$ is...
($j$ indexes the observations at times $t_i$)

Images $S\#i=2,3,4$ w/ observations $y_i$ at times $t_i$ w/ unknown time delays and magnitude shifts (relative to $S1$): $(\Delta t_i, \Delta m_i)$
$i=2,3,4 \quad y_i(t_i) = \Delta m_i + f(t_i - \Delta t_i) + \epsilon_i \quad \epsilon_{ij} \sim N(0, \sigma^2_{ij})$

Sample Posterior: $P(\Delta m, \Delta t, A, \tau^2 | D)$

> e.g. for results, sample w/ metropolis-within-gibbs
> one could just sample parameters for this example as well

[A diagram with four subplots arranged in a 2x2 grid. Each subplot shows a histogram.
The top-left subplot is labeled 'marginal posterior $\Delta t_2$'. The x-axis is labeled '$\Delta t_2$'.
The top-right subplot is labeled 'marginal posterior $\Delta t_3$'. The x-axis is labeled '$\Delta t_3$'.
The bottom-left subplot is labeled 'marginal posterior $\Delta t_4$'. The x-axis is labeled '$\Delta t_4$'.
The bottom-right subplot is labeled 'your best estimate for each time delay'. The x-axis is labeled 'delay'.
This indicates that the posteriors for the time delays Δt₂, Δt₃, and Δt₄ have been estimated.]

> Shifting by estimated $\Delta t_i$, $\Delta m_i$ datasets line up pretty well as hoped!

---
(Page 127)

## Lecture 22
_14.3.25_

> gelman BDA ch 21, 5

[Today: Finishing Gaussian processes, starting hierarchical Bayes / Probabilistic Graphical Models]

### Gaussian Processes cont.

Previously: used "squared exponential kernel"
$\rightarrow \text{Cov}[f(t), f(t')] = k(t, t') = A^2 \exp(-(t-t')^2 / \tau^2)$
$f|A,\tau \sim N(\underline{1}c, K)$ > (characteristic amplitude, characteristic timescale or duration)

this kernel gives very smooth curves (infinitely differentiable)
> also has properties

*   Stationary: $k(t, t')$ invariant to $t \rightarrow t+k, t' \rightarrow t'+k$ > (property depends only on difference of inputs $t-t'$)
*   Symmetric: $k(t,t') = k(t',t)$

### OTHER COV FUNCTIONS:
"Ornstein Uhlenbeck Process" (damped random walk)
Exponential covariance function.
> (often used to model fn that follow)

$k(t, t') = A^2 \exp(-|t-t'|/\tau)$

Long-run dist. of stochastic differential eq. > mean-reversion long term random walk
$df(t) = \tau^{-1}[\mu - f(t)]dt + \sigma dw_t$
> long-term mean ↑
> A = $\tau \sigma^2 / 2$ volatility
> mean-reversion timescale ↑

---
(Page 128)

[A small sketch of a jagged path] > more jagged -> squared exponential

this kernel is everywhere continuous but not differentiable > it might be more jagged than what you want (or depends on what you want your use case)

"Matern class" of cov functions
> somewhere in between smooth & jagged

$V=1/2$ special case of Matern kernel:
$K_{\text{Matern}}(r) = \frac{2^{1-V}}{\Gamma(V)} (\frac{\sqrt{2V}r}{L})^V K_V(\frac{\sqrt{2V}r}{L})$
with five parameters $V, L, c$, where $K_V$ is a modified Bessel fn.

When $V = p + 1/2$ is half-integer; exponential x p-polynomial.
$k_{V=p+1/2}(r) = \exp(-\frac{\sqrt{2V}r}{L}) \frac{\Gamma(p+1)}{\Gamma(2p+1)} \sum_{i=0}^p \frac{(p+i)!}{i!(p-i)!} (\frac{\sqrt{8V}r}{L})^{p-i}$

"Periodic cov functions" > (ex sheet 3)
e.g. $k(t,t') = A^2 \exp(-\frac{2\tau^2}{L^2} \sin^2(\pi(t-t')/T))$

[Two sketches of periodic functions]

> End of GP's

[Scribbled out math]

---
(Page 129)

(board)

### PROBABILISTIC GRAPHICAL MODELS
> conditional independence

(Depicting complex probability distributions)

Suppose a,b,c are r.v.s with a joint pdf:
$P(a,b,c) = P(c|a,b)P(a,b)$
$= P(c|a,b)P(b|a)P(a)$ (general factorisation)

Directed Acyclic Graph:
> (parent)
[A diagram shows three nodes labeled a, b, and c. Node 'a' has arrows pointing to 'b' and 'c'. Node 'b' has an arrow pointing to 'c'.]
> nodes are connected by one-way edges that do not form any cycles

> DAG

> why care about directed graphs? Everything is bayesian statistics is directed. It tells us how to construct joint distribution.

Generally, if $\vec{x}$ is a collection of k r.v.s, then given a DAG, the factorisation is written as
$P(\vec{x}) = \prod_{k=1}^K P(x_k | \text{Parents of } x_k)$ > if $x_k$ has no parents then just $P(x_k)$

(Eq)
[A directed acyclic graph with 7 nodes, x₁ to x₇.
x₁ points to x₂ and x₄.
x₂ points to x₃ and x₅.
x₃ points to x₅.
x₄ points to x₆.
x₅ points to x₇.
x₆ points to x₇.]

$P(x_1, ..., x_7) = P(x_1) P(x_2 | x_1) P(x_3)$
$\times P(x_4|x_1, x_2, x_3)$
$\times P(x_5|x_2, x_3) P(x_6|x_4)$
$\times P(x_7|x_5, x_6)$

> we call this CONDITIONAL INDEPENDENCE (the lack of connections implies some model STRUCTURE!)
> fact it isn't fully connected implies conditional independence (structure)

---
(Page 130)

> imagine you want to simulate an image with lots of galaxies in it

### Generative Models (slides)
> non-sky

Causal process for generating images
[A diagram with four nodes. Three nodes 'object', 'position', and 'orientation' point to a central node 'image'.]
> no. of galaxy or star w/ prior from cosmology or sth.
> how elliptically round galaxy is

> e.g. sample joint dist of (object, pos, orientation) can produce an image of galaxy/star
> can think of this joint dist. as a graph (?)

### Notation (board)

[A diagram of an empty square or an empty circle.]
open node = latent parameter/unobserved data

[A diagram of a shaded square.]
shaded node = observed parameter or data (conditioned on)

[A diagram of a small filled dot.]
filled dot = fixed and known constant

[A diagram of a square with "i=1...N" inside.]
plate: N independent replications of what's inside

---
(Page 131)

> representing bayesian relationships with graphs

(Eg) Gaussian i.i.d. data $y_i \stackrel{iid}{\sim} N(\mu, \sigma^2=1)$
$i=1,..,N$

[A diagram shows a filled dot for $\sigma^2=1$. A node for $\mu$ points to a plate containing node $y_i$ for $i=1...N$. A dotted line expands the plate to show individual nodes $y_1, ..., y_N$ each with an arrow from $\mu$.]
=
[A simplified diagram with the plate notation.]
$= P(\mu, \vec{y}) = P(\mu) [\prod_{i=1}^N P(y_i|\mu)]$

If y observed, (shaded)
[A diagram with a filled dot for $\sigma^2=1$ and a node for $\mu$. An arrow from $\mu$ points to a plate containing a shaded node $y_i$ for $i=1...N$.]

$\propto P(\mu|\vec{y}) \propto P(\mu) [\prod_{i=1}^N P(y_i|\mu)]$

(Eg) Both $\mu, \sigma^2$ unknown
[A diagram with nodes for $\mu$ and $\sigma^2$. Both nodes have arrows pointing to a plate containing node $y_i$ for $i=1...N$.]
$= P(\mu, \sigma^2, \vec{y})$
$= P(\mu) P(\sigma^2) [\prod_{i=1}^N N(y_i; \mu, \sigma^2)]$
$P(\mu, \sigma^2)$

---
(Page 132)

### Conditional Independence and Probabilistic Graphical Models

Let a,b,c be r.v.s

(Marginal) Independence
$P(a,b) = P(a)P(b) \rightarrow a \perp b | \emptyset$
> (not conditioning on anything)

Conditional Independence
$P(a,b|c) = P(a|c)P(b|c) \rightarrow a \perp b | c$
i.e. $P(a|b,c) = P(a|c)$ ← > if I know c then b gives me no more info about a

(Eq)
[A diagram showing node c with arrows pointing to nodes a and b.]
$P(a,b,c) = P(c)P(a|c)P(b|c)$
$P(a,b) = \int P(c)P(a|c)P(b|c) dc \neq P(a)P(b)$
$\rightarrow a \not\perp b | \emptyset$ > "a is not indep of b marginally"

[A diagram showing node c (shaded) with arrows pointing to nodes a and b.]
$P(a,b|c) = \frac{P(a,b,c)}{P(c)} = P(a|c)P(b|c)$
$\rightarrow a \perp b | c$
> "a is indep of b conditionally"

---
(Page 133)

(Eq)
[A diagram showing node a pointing to c, and node b pointing to c.]
$P(a,b,c) = P(a)P(b)P(c|a,b)$
$P(a,b) = P(a)P(b) \int P(c|a,b) dc$
$= P(a)P(b) \rightarrow a \perp b | \emptyset$ > "marginal independence between a & b"

[A diagram showing node a pointing to c (shaded), and node b pointing to c (shaded).]
$P(a,b|c) = \frac{P(a,b,c)}{P(c)} \neq P(a|c)P(b|c)$
$\rightarrow a \not\perp b | c$
> i.e. observing c, the child of a & b, implies a, b are now correlated
> a is no longer indep. of b if you condition on c

### D-separation Example
(slides)

[A diagram with nodes a, e, c, f, b. Arrows from a to e, f to e, c to e, e to b.]
$a \not\perp b | c$
> path a->e->b is not blocked so a is not perp to b|c

[A diagram with nodes a, e, c, f, b. a, c, and f point to e, which points to b. Node f is shaded.]
$a \perp b | f$
> path a->e->b blocked by f, so a is perp to b|f

> want to test which nodes are indep. given other nodes.
> read off independent properties from graph to generalise our models

---
(Page 134)

### D-separation (slides)

*   A, B and C are non-intersecting subsets of nodes in a directed graph.
*   A path from A to B is blocked if it contains a node such that either
    a) The arrows on the path meet either head-to-tail $\leftarrow \rightarrow$ or tail-to-tail $\leftarrow \rightarrow$ at the node, and the node is in the set C, or
    b) The arrows meet head-to-head $\rightarrow \leftarrow$ at the node, and neither the node, nor any of its descendants are in the set C.
*   If all paths from A to B are blocked, A is said to be d-separated from B by C.
*   If A is d-separated from B by C, the joint over all variables in the graph satisfies $A \perp B | C$.

> example, see slides
> (very brief, didn't dwell on this in lecture)

e.g.
[A diagram with 7 nodes. 1->2, 1->3, 2->4, 3->4, 4->5, 6->5, 7->5.]
$x \not\perp y$
$x \perp y | F$
$x \not\perp y | F, G$

---
(Page 135)

## Lecture 23
_17.3.25_

> very to think about statistics: model -> how do I generate parameters? -> gets data -> story starts with DGP
> gelman BDA
> why are the models/methodologies we so stupid/parametrised?

Today: Hierarchical Bayes / Bayesian Model Selection

e.g. generative model (slides)
[A diagram shows a causal process for generating an image. A node "light source" points to "pixel", which is inside a plate labeled "pixels in image". A node "object" and "camera" both point to the "light source" node. The "object" node is in a plate labeled "objects".]
> data generating process:
> light source is some type of object
> properties for a source
> from light source to each pixel
> data: probabilistic model for astronomical images

PGM linked to hierarchical bayesian models
(all g. surveys research is basically hierarchical bayesian model)
Supernovae -> complex, more complex hierarchical bayesian model
[A complex PGM diagram for supernovae.
Nodes for 'dist pop. Rᵥ, E' and 'dust pop.' point to a 'dust effect, Aᵥ, Eᵥ' node.
A node 'SNIa feature Tᵥ' points to a 'dist effect, Aᵥ, Eᵥ' node.
A node 'model inherent properties e.g. how populations of these were modelled' has an arrow pointing to 'SNIa feature Tᵥ'.
The 'dist pop. Rᵥ, E' node also points to the 'SNIa feature Tᵥ' node.
The 'dust effect' and 'SNIa feature' nodes point to a node 'φ'.
'φ' is inside a plate for S=1,...,N_SN.
Inside the plate, φ points to m_s.
Also inside the plate, a node for 'dist metric SED, intrinsic f' points to m_s.
A node for "correlate properties that effect SN brightness" points to 'φ'.
A node for "model/statistically describe SNIa" points to the 'dist metric' node.]
> (dist closing) and galaxy

> model spectral energy distribution SED:
> SED: model SN brightness (flux) as fn. of time & wavelength
> how diff SN Ia's have diff time varies as a fn of wavelength. trace it to variations in physical origin of time.

> slides discussing N SN (plate)

> stuff outside plate describes pop distribution / cosmological parameters (things not related to individual SN) and stuff inside plate describes stuff related to individual supernovae.

[A scribble showing a plate with nodes and arrows inside and outside.]
> params

---
(Page 136)

> this lecture is more important
> while discussing example. very complicated, hierarchical bayesian model. see slides
> you're interested.

### Common Problems in Astronomy (slides)

*   Want to learn about a population of objects from a finite sample of individuals, each measured with error.
> i.e. not just one supernova. (one star is fit, many are to get a full picture for that population)

*   Observed data is actually a combination of uncertain astrophysical & instrumental & selection effects. Need to model them to infer the "intrinsic" properties of the object or population of objects ("deconvolve").

### What is Hierarchical Bayes?
(slides)

"simple" Bayes: $D|\theta \sim \text{Model}(\theta)$ [A diagram shows a node θ pointing to a node D]
Posterior (Bayes' theorem): $P(\theta|D) \propto P(D|\theta)P(\theta)$

Hierarchical Bayes: $\theta_i$ parameter of individual
$\alpha, \beta$ hyperparameters of population
$D_i|\theta_i \sim \text{Model}(\theta_i)$
$\theta_i|\alpha,\beta \sim \text{Pop Model}(\alpha, \beta)$
[A diagram shows nodes α,β pointing to a plate containing θ_i, which in turn points to D_i.]

Joint posterior:
$P(\{\theta_i\}, \alpha, \beta | \{D_i\}) \propto [\prod_{i=1}^N P(D_i|\theta_i) P(\theta_i|\alpha, \beta)] P(\alpha, \beta)$

build up complexity by layering conditional probability

> def. hierarchy. e.g. population level parameters. (pop. dist. of host galaxy dist of systems).
> object level parameter. e.g. property of individual SN.

---
(Page 137)

Forward model:
[A diagram shows nodes α,β pointing to a plate containing θ_i, which points to D_i. The plate is labeled θ₁..θₙ and D₁..Dₙ.]

Plate notation:
(loop over individuals in sample)
[A simplified diagram with a plate containing "θ_i -> D_i", labeled i=1...N. Nodes α,β point to the plate.]

### Advantages of Hierarchical Bayesian Models

*   Common problem in astronomy: infer properties of population from finite sample of individuals with noisy measurements.
*   Incorporate multiple sources of randomness & uncertainty as "latent variables" with distributions underlying the data.
*   Express structured probability models adopted to data-generating process ("forward model")
*   Bayesian: Full (non-gaussian) probability distribution = global, coherent quantification of uncertainties.
*   Completely explore & marginalise posterior trade-offs / degeneracies between parameters/hyperparameters.

---
(Page 138)

(slides)

Simplest Hierarchical Bayesian/Multi-level Model:
"Normal-Normal" for Standard Candle Mag.
$s=1,...,N$

Level 1: Pop. distribution of latent variables (absolute mags.)
> "population dist/prior"
$M_s \sim N(M_o, \tau^2)$
> latent variables (pop. means, pop. variance) -> hyperparameters

Level 2: Measurement error process
> "measurement likelihood"
$D_s|M_s \sim N(M_s, \sigma^2_s)$
> measurements (data)
> heteroskedastic meas. error variance (known)

Joint probability density of data, latent variables, hyperparameters
$H \equiv \{M_o, \tau^2\}$

> "actually PGMs for this joint"

$P(\{D_s\}, \{M_s\}, H)$

[A PGM diagram. Nodes M_o, τ² point to a plate. Inside the plate is a node M_s. A filled dot for σ_s² also points to a node D_s inside the plate. M_s points to D_s. The plate is labeled s=1,...,N.]
> joint factors into conditional and marginal pdfs based on model assumptions

Joint probability density of all the things: data, latent variables, hyperparameters
$P(\{D_s\}, \{M_s\}, H) = [\prod_{s=1}^N P(D_s|M_s) P(M_s|M_o, \tau^2)] \times P(H)$
> measurement likelihood
> population distribution/prior
> hyperprior

---
(Page 139)

### Putting the Bayesian in Hierarchical Bayesian

Joint posterior of all unknowns given the data
$P(\{M_s\}, H | \{D_s\}) = \frac{P(\{M_s\}, H, \{D_s\})}{P(\{D_s\})}$ <-- ignorable normalisation const. "ignore until next sheet"

(posterior on N+2 dim. parameter space)
$P(\{M_s\}, H | \{D_s\}) \propto [\prod_{s=1}^N P(D_s|M_s) P(M_s|M_o, \tau^2)] \times P(H)$

[A PGM diagram. Nodes M_o, τ² point to a plate. Inside the plate is a node M_s. A filled dot for σ_s² also points to a shaded node D_s inside the plate. M_s points to D_s. The plate is labeled s=1,...,N.]

### Hierarchical vs Regular Bayes

*   Could regard as just a general Bayesian inference problem in a very high dim. parameter space, e.g.
    $\theta = \{M_1, ..., M_N, M_o, \tau^2\} = \{M_i, M_o, \tau^2\}$
    $P(\theta|D) \propto P(D|\theta)P(\theta)$
    $P(\theta|D) \propto P(D|M) P(M|M_o, \tau^2) P(M_o, \tau^2)$
    $P(\theta|D) \propto [\prod_{s=1}^N P(D_s|M_s) P(M_s|M_o, \tau^2)] P(M_o, \tau^2)$
    > helpful in bayesian computation but if you can take advantage of the structure of a problem

*   However, special hierarchical structure is useful for modelling, estimation & computation
*   For large N, wouldn't want to do N+2 dim. Metropolis MCMC!

---
(Page 140)

### Gibbs Sampling & Hierarchical Bayes
> run down to N+2 gibbs sample. let's treat that in 2 different ways

Utilises conditional independence structure of PGM/posterior to derive conditional posterior densities
$P(\{M_s\}, H | \{D_s\}) \propto [\prod_{s=1}^N P(D_s|M_s) P(M_s|M_o, \tau^2)] \times P(H)$
Gibbs: use $P(\{M_s\}|H, \{D_s\})$ & $P(H|\{M_s\}, \{D_s\})$ asked to sample
> need 2 conditional posterior densities

1.  For $s=1,...,N$ Sample latent variables conditional on data and hyperparameters
    $P(M_s|H, D_s) \propto P(D_s|M_s) \times P(M_s|M_o, \tau^2)$ > indep!
    (conditional independence) (gaussian) (gaussian)
    > product of 2 gaussians is gaussians -> Maths to derive posterior is just mean/variance

2.  Sample hyperparameters from conditional on data and latent variables
    $P(M_o, \tau^2|\{M_s\}, \{D_s\}) = P(M_o, \tau^2 | \{M_s\})$
    (conditional independence) > depending on D_s not on M_s from graph?
    $= P(M_o|\{M_s\}) P(\tau^2|\{M_s\})$
    (gaussian) (inv-$\chi^2$)

Reduces to the familiar posterior for unknown mean and variance of gaussian data (sheet 2 prob. 1 BDA 3.2-3.3)

> for data (condition on D always)
> input some guess, say $M_s, \tau^2$, and condition on that, sample latent variables
> turns out that $D_s$ is condition on $M_s$, M of SN s is indep to M of SN s', D_s etc.
> so samples N indep. to. two terms rather than N+2 terms. which depend on D.
> so can keep them each s which is easy to sample from b/c gaussian
> can cut out read out indep relationships of graph.
> is about easy to worry about data, only these N factors. but those look like gaussian, which only factors into a gaussian. is Inv-X² -> much easier to sample from
> (w/ unknown μ, σ²)

---
(Page 141)

Full posterior:
$P(\{M_s\}, H | \{D_s\}) \propto [\prod_{s=1}^N P(D_s|M_s) P(M_s|M_o, \tau^2)] \times P(H)$
[A PGM is shown for the full posterior. M_o, τ² point to a plate containing M_s. M_s and a filled dot σ_s² point to a shaded D_s. The plate is for s=1...N.]

1.  [A PGM shows M_o, τ² (shaded) pointing to M_s. M_s and σ_s² (filled dot) point to D_s (shaded). An arrow points from M_s out of a box around it.]
    $P(M_s | M_o, \tau^2, \{D_s\}) \propto P(D_s|M_s)$
    $P(M_s|M_o, \tau^2)$
    $N(D_s; M_s, \sigma_s^2) \times N(M_s; M_o, \tau^2)$

2.  [A PGM shows M_o, τ² pointing to a plate with M_s (shaded). An arrow points out from M_o, τ².]
    $P(M_o, \tau^2 | \{M_s\}, \{D_s\}) = P(M_o, \tau^2 | \{M_s\})$
    $= P(M_o | \tau^2, \{M_s\}) P(\tau^2 | \{M_s\})$
    gaussian      inv-$\chi^2$
> hyperprior
> $P(M_o, \tau^2) = P(M_o| \tau^2) P(\tau^2) \propto 1$
> $P(M_o, \tau^2) \propto 1$

(Similar to sheet 2 problem 1)
> "think about this on your own"

Hyperprior $P(M_o, \tau^2) \propto 1$
Draw from $\tau^2 | \{M_s\} \sim \text{Inv-}\chi^2(N-3, \frac{(N-1)}{(N-3)} S^2)$
$M_o | \tau^2, \{M_s\} \sim N(\bar{M}, \tau^2/N)$
$\bar{M} = \frac{1}{N} \sum M_s \quad S^2 = \frac{1}{N-1} \sum (M_s - \bar{M})^2$

e.g. 100 SN
[Histogram labeled "obs abs mag, D_s".]
> can make histogram of obs data
> $D_s = \hat{M}_s - \hat{\mu}_s$
> data with some error
> see slides

[Histogram labeled "gibbs sampling sample M_o, τ²".]
> in N+2 dim space, but then we want to reverse sample M_o, τ² | M_s
> in N+2 dim, but we can sample M_s | ...

[Histograms for μ_o and τ², labeled "Marginal posterior estimate".]

[Scatter plot of M_s vs D_s with an upward trend.]
> scatterplot from Z-dist, not by data in conditional posteriors, can combine data from all sources (this is useful) to truth on speed
> measurement error dilutes/spread these points

[Histogram for M_s labeled "the individual SN calc shallow abs mag".]
> observed

[Histogram for M_s labeled "individual estimates have narrow spread from posterior estimation (partially corrected for measurement error)".]
> marginal posterior population parameters

---
(Page 142)

> HB models, important part
> partial pooling -> "borrowing strength"
(slides)

### HB Models: Partial Pooling, "Shrinkage" & "Borrowing of Strength"
> each is-a-verse Di
> individual (no-HB)
> find out M-L estimate

*   Common sense procedure:
    *   Analyse each individual object's data Di separately and get each individual MLE estimate (with error).
    *   Plug in all $\{ \hat{\theta_i} \}$ to estimate population hyperparameters

*   PROBLEM: Each individual $\hat{\theta_i}$ estimate may be unbiased but collectively give a biased estimate of population (e.g. variance) because of errors.

*   SOLUTION: Use HB to model & infer individuals & population simultaneously and get better estimates of both.

> "notes to self": (attempt to develop intuition).
> HB suff posterior -> $P(M_s|M_o, \tau, D) \propto P(D_s|M_s) P(M_s|M_o, \tau^2)$
> individual estimate as prior
> pop dist acts as prior

[A plot shows two distributions for M_s. A wide one is labeled 'pop dist acts as prior'. A narrower one is labeled 'individual estimate just from what we measure from D_s'.]
> say we want to infer Ms. MLE and take peak as our best estimate of Ms.
> Individual estimate is just from what we measure from D_s. observing our data(MLE)
> But HB model full posterior incorporates 'prior' on Ms. so e.g. Ms was drawn from N(M_o), but M_s is very unlikely. This will sway our estimate of Ms. throwing this prior balance between measuring part of data but also probability that interest value may be existing in the population in the first place.
> How is this diff to MAP?
> In general, HB gets p(M_s|{D_s}). So can infer true hyper params instead of using MLE. check how this correlates. (I think)

---
(Page 143)

(slides)
> "ester not unbiased but their biases helps you out in away"

### Shrinkage Estimators
> closer to truth on avg than considering individual estimates for each individual in isolation

*   Bias estimator of individual towards the population
*   Leads to overall lower MSE than individual unbiased estimators
*   Allows "sharing of information" between individuals to improve overall estimation
example: slides (bring explanation)

### Shrinkage with Hierarchical Model

Return to our previous example:
level 1: pop. dist. of latent variables $M_s \sim N(M_o, \tau_s^2)$ > "pop. dist." / "prior"
Level 2: measurement error process $D_s|M_s \sim N(M_s, \sigma_s^2)$ > "measurement likelihood"
> chapter your highness, you shall become (D_s|M_s)

Individual MLE of SN is alone: $\hat{M}_s = D_s$ (unbiased)
> individual estimate comes from here

Population dist. acts as prior
$P(M_s|M_o, \tau^2; D_s) \propto P(D_s|M_s) P(M_s|M_o, \tau^2)$
and pulls posterior estimate of individual closer to population mean estimate
> think of pop dist as sort of a prior
> to counter-balance and shrink posterior estimate towards pop. mean estimate

Pull/shrinkage controlled by population variance $\tau^2$

> estimate from full posterior:
> $P(M_s, M_o, \tau^2 | \{D_s\}) \propto P(\{D\}) [P(D_s|M_s)P(M_s|M_o, \tau^2)]$
> s=1..N
> just ignore terms not in Ms, then in simple gaussian example

---
(Page 144)

How does HB implement shrinkage?
example (dataset of 8 SN)

[A plot showing 8 data points with error bars, labeled 1 to 8 on the x-axis, and M̂_s=D_s on the y-axis. The y-axis is labeled "abs mag".]
> if you knew M_o, τ of individual Ms. How did it depend on τ.

[A plot with τ on the x-axis. Several curves originate from different y-values at τ=0 and converge towards a single value M_o as τ increases. The y-axis is labeled "conditional posterior means", E(M_s|M_o,τ,D). One curve is labeled "one thick for each SN, showing mean estimate". One y-value is labeled M̂_s=D_s. The horizontal line is labeled M_o.]
> we can take τ to zero, converge to M_o
> τ controls degree of shrinkage towards M_o
> (increase τ, wide range is ok, prob not so, wide spread)

but we don't know M_o, τ to estimate that
$P(M_o, \tau | D) = P(M_o| \underline{\tau, D}) P(\tau|D)$
> form marginalised posteriors
> individual Ms

[Two plots. Left plot shows P(M_o|τ,D) vs M_o for different τ, labeled "posterior". Right plot shows P(τ|D) vs τ, labeled "posterior mean", showing a peak.]
> poster. dist depends on τ

[A plot of E(M_s|M_o,τ,D) vs τ. Several curves converge as τ increases. The x-axis is labeled "population variance". The y-axis is "population mean".]
> these curves shrink towards population mean with decreasing τ.

hierarchical bayes gives the posterior density P(τ|D) -> way to estimate τ
(don't request τ, have to take post. estimates of τ & integrate?)

[A plot of p(τ|D) vs τ with a peak at τ=0.134. The label is E(τ|D)=0.134]
> how can work out best trade-off between having these same pop. values vs allowing them to be independently estimated? M_o is at about τ=0.134

[A final plot of E(M_s|M_o,γ,D) vs τ. Several curves converge. A vertical line at τ=0.734 is shown. The y-axis is labeled "conditional post. mean".]
> on avg HB estimate closer to truth than individual MLE
[A small plot below shows individual MLEs vs SN #, and HB estimates vs SN #.]

---
(Page 145)

## Lecture 24
_19.3.25_

> recommend reading (will's book!)
> Mackay "information theory inference & learning algorithms" ch. 28

Today: Bayesian Model selection

### Model Comparison & Selection
> main issues:
> *just need 9* inference
> which assumes model true
> & check model against the data (P(D|θ))
> this lecture is on comparing diff. competing models

*   No. of spectral lines in a (noisy) spectrum?
*   Clustering/mixture models - how many clumps?
*   Time series - curve fitting
    *   Is there a trend?
    *   Complexity/order/degree of best model.
    *   which GP kernel best explains the data?
*   Cosmology - standard (8-parameter) cosmological model vs more exotic (more parameters) models?
    *   8 in standard model (ΛCDM)
    *   but people looking for evidence of more exotic variations of the model (f(t) any, and figure out dark energy)
    *   but then we need to introduce more parameters
    *   (i) which extra parameters are warranted by data (adding more params will always give better fit. but how do we know if adding extra params is just fitting to noise or actually warranted by data?) -> hope to answer through bayesian model comparison

---
(Page 146)

### Bayesian Model Comparison
"Max"

Parameter estimation: posterior on parameters
$P(\theta|D, M_1) = \frac{P(D|\theta, M_1) P(\theta|M_1)}{P(D|M_1)}$

Model selection: posterior on models
$P(D|M_1) = \int P(D|\theta, M_1) P(\theta|M_1) d\theta_1$
> evidence or marginal likelihood
> needs proper prior!
> "cost of improper"

Posterior odds ratio:
> "betting" on two competing models

$\frac{P(M_1|D)}{P(M_2|D)} = \frac{P(D|M_1)}{P(D|M_2)} \times \frac{P(M_1)}{P(M_2)}$
> BF ↑
> Prior odds ↑
> prior importance on the model -> model choice in cosmological studies. based on theory, philosophical/ideological preference for "equal probabilities"

Bayes Factor$_{12}$ = $\frac{P(D|M_1)}{P(D|M_2)}$

---
(Page 147)

> let's develop some intuition

For a given model with a single parameter, $\omega$, consider the approximation
$p(D) = \int p(D|\omega) p(\omega) d\omega \approx p(D|\omega_{MAP}) \frac{\Delta\omega_{posterior}}{\Delta\omega_{prior}}$

Where the posterior is assumed to be sharply peaked.
[A diagram shows two distributions on the ω axis. A wide, flat distribution is labeled $\Delta\omega_{prior}$. A narrow, peaked distribution inside it is labeled $\Delta\omega_{posterior}$.]
> cotton approximation.
> examples
> $p(D) \approx p(D|\omega_{MAP})$. $\frac{P(\omega_{MAP})}{p(\omega_{MAP}|D)}$
> assume post. pdf on gaussian
> $p(\omega | D) \approx N(\omega_{MAP}, \Sigma_{post})$
> $p(D|\omega) \propto N(\omega_{MAP}, \Sigma_{post}^{-1})$
> $p(D) \approx p(D|\omega_{MAP}) \frac{\Sigma_{post}^{1/2}}{Z_{prior}(\omega_{MAP})} \frac{\Delta\omega_{post}}{\Delta\omega_{prior}}$

Taking logarithms, we obtain
$\ln p(D) \approx \ln p(D|\omega_{MAP}) + \ln (\frac{\Delta\omega_{posterior}}{\Delta\omega_{prior}})$
> negative
> cause we hope posterior always large, simpler problem w/ less params

generalize:
With M parameters, all assumed to have the same ratio $\Delta\omega_{posterior}/\Delta\omega_{prior}$, we get
> term related to fit of data under best fit parameters
> term that discounts for no. of params & ratios of prior/posteriors

$\ln p(D) \approx \ln p(D|\omega_{MAP}) + M \ln (\frac{\Delta\omega_{posterior}}{\Delta\omega_{prior}})$ → OCCAM FACTOR
> negative ↓
> thin line indicates a problem
> i.e. adding more parameters will always give a better fit to the model, but this might be because it is overfitting to noise. Occam's razor (2nd term) penalizes overfitting
> with this term, penalizes you for adding more params.
> want to find sweet spot, enough parameters w/ not overfitting

---
(Page 148)

### Matching data and model complexity

[A plot of P(D) vs D. Three curves for models M₁, M₂, M₃ are shown. M₁ is a narrow, high peak. M₂ is a wider, lower peak. M₃ is a very wide, very low distribution. An orange line is drawn for a specific data point D₀.]
> P(D|M)
> if we get this data D₀, we can read off P(D₀|M₁), P(D₀|M₂), P(D₀|M₃) and remember

*   more complex models can predict a greater variety of data sets.
*   M₁ can only predict a limited amount of data (e.g. linear model)
*   M₂ can predict slightly more complex data sets (e.g. might be a quadratic)
*   M₃ e.g. if a neural network may be able to predict arbitrary functions
*   p(D) is sets a distribution over potential data sets
*   the more complex datasets a model can predict, the more "spread out" p(D) will be. It is normalised -> for any particular dataset it will have a low probability overall
*   so simpler models will have more limited range of data sets they can predict, but because of that, overall p(D) is higher
*   whereas, complex models predict greater range but p(D) lower b/c of that normalisation factor (paying a price)

So this gives a good way to weigh for good models to dataset vs the overall complexity.
e.g. for dataset D₀, M₁ is v. low probability, M₃ decent, M₂ is slightly more probable. But M₂ has best balance between complexity of model (more complex than M₁) but also not too complex. (/M₃)

---
(Page 149)

> recommend read ch 28 of Mackay

### Marginalising Joint Distribution of Parameters and Data
hypothesis H
($\omega \sim P(\omega|H)$ & $D \sim P(D|\omega, H) \rightarrow (D,\omega) \sim P(D, \omega|H)$)
$\theta \sim P(\theta|M)$ & $D \sim P(D|\theta, M) \rightarrow (D, \theta) \sim P(D, \theta|M)$

[A large diagram with multiple plots.
Top-left plot: D vs. P(D|M) for three models M₁, M₂, M₃. A horizontal line is drawn for D_obs.
Bottom-left plot: θ vs. P(θ|M) for the three models, showing their prior distributions.
A series of scatter plots in the center show P(D, θ|M) for M₁, M₂, M₃.
Right column shows plots of P(θ|D, M) vs. θ for the three models.]

> draw θ from prior, then given θ, draw D from sampling distribution. (what likelihood is this? just)
> This gives θ and D pairs from the joint distribution e.g. P(D, θ|M₁)
> The different data points (i.e. samples from joint around data set here is just a single value shown by dotted line)

> draw from prior P(θ|M)
> given this draw, draw from sampling dist. P(D|M,θ)
> gives pairs D,D from joint.

[A side diagram shows a plot of height vs. a distribution. It illustrates that for a given height=175cm, a sample is drawn, forming a joint data point.]

---
(Page 150)

> used to be director of IoA

### Interpreting the Bayes Factor / Evidence Ratio: The Jeffreys Scale

$\Delta \ln E = \ln BF$
> these thresholds are up for debate & not rigorous

| $\Delta \ln E < 1$ | "not worth more than a bare mention" |
|---|---|
| $1 < \Delta \ln E < 2.5$ | "significant" |
| $2.5 < \Delta \ln E < 5$ | "strong/very strong" |
| $5 < \Delta \ln E$ | "decisive" |

### How to calculate Evidence $P(D) = \int P(D|\theta)P(\theta) d\theta$

*   Analytic (really simple, nice problems)
> analytical integral

*   Laplace approximation
> (accurate)

*   Savage-Dickey Ratio (Nested Models)
> likely slow "almost" strongly peaked if not all parameters are "interesting"

*   Monte-Carlo: probably slow if likelihood is peaked
    $\theta_i \sim P(\theta_i)$
    $P(D) \approx \frac{1}{m} \sum_{i=1}^m P(D|\theta_i)$
    > fundamentally approx
    > need an intelligent integration sum
    > approach with MC avg of likelihood

*   Harmonic Mean Estimator (unstable)
    $\theta_i \sim P(\theta|D)$
    $P(D) \approx [\frac{1}{m} \sum_{i=1}^m P(D|\theta_i)^{-1}]^{-1}$
    > infinite variance
    > MC avg of inverse likelihood from posterior

*   Nested Sampling***---
(Page 151)

Calc. evidence via Laplace approx:

(Recall: The Laplace Approximation.)

Evidence $$P(D) = \int P^*(\theta|D) d\theta$$
Unnormalised posterior $$P^*(\theta|D) = P(D|\theta)P(\theta)$$
Find MAP estimate: $$\theta_0 = \underset{\theta}{\operatorname{argmax}} \ln P^*(\theta|D)$$
Taylor expansion:
$$ \ln P^*(\theta|D) \approx \ln P^*(\theta_0|D) - \frac{1}{2}(\theta-\theta_0)^T A (\theta-\theta_0) + \dots $$
Hessian at mode: $$ A_{ij} = - \frac{\partial^2}{\partial\theta_i \partial\theta_j} \ln P^*(\theta|D)|_{\theta=\theta_0} $$

$$ P^*(\theta|D) \approx P^*(\theta_0|D) \times \exp(-\frac{1}{2}(\theta-\theta_0)^T A (\theta-\theta_0)) $$

Not always accurate:
e.g.

[Two graphs are shown side-by-side. The left graph plots -ln P*(Θ|D) against Θ. A dark blue line shows the actual function, which is a curve. A light blue parabola is overlaid, representing the Taylor expansion approximation. The right graph plots P*(Θ|D) against Θ. The dark blue line shows the actual posterior, which is a skewed bell curve. A light blue symmetric bell curve represents the Laplace/Gaussian approximation. The area under the dark blue curve is shaded.]

> e.g. laplace unnormalised posterior...
> laplace approx is good

> but if you trust your approximation on an integral we 'praus' & get this?

\* Evidence from Laplace Approximation: \*
$$ P(D) = \int P^*(\theta|D) d\theta \approx P^*(\theta_0|D) \times |2\pi A^{-1}|^{1/2} \int N(\theta|\theta_0, A^{-1}) d\theta $$

> w.r.t P(D|Θ) x det(2πA)⁻¹/² ?

$$ P(D) \propto P^*(\theta_0|D) \times \det(A/2\pi)^{-1/2} $$
$$ P(D) \approx P(D|\theta_0) \times P(\theta_0) \times \det(A/2\pi)^{-1/2} $$

> having wide prior?
> unnormalised?
> need to be able to evaluate posterior at least 2x to do this?
> in many cases just use (unstable posterior)

$$ A = - \nabla \nabla \ln P(\theta|D) $$

---
(Page 152)

this implements Occam's razor

### Evidence implements Occam's Razor
> samples to estimate better

$$ \underbrace{P(D)}_{\text{Evidence}} \approx \underbrace{P(D|\theta_0)}_{\text{best fit likelihood}} \times \underbrace{P(\theta_0) \times \det(A/2\pi)^{-1/2}}_{\text{Occam factor}} $$
> (?? or just A⁻¹/²)

$$ \Sigma_{post} = A^{-1} \quad \text{roughly} $$

how can we see this?
### Suppose:
$$ P(\theta) = N(\theta|\theta_{prior}, \Sigma_{prior}) $$
> Suppose prior is Gaussian

### Occam Factor:
> assume prior centered near MAP to do so or use 'generalised' version? (shan't get)
$$ \propto |2\pi \Sigma_{prior}|^{-1/2} \times |2\pi \Sigma_{post}|^{1/2} $$
$$ \propto \frac{|\Sigma_{post}|^{1/2}}{|\Sigma_{prior}|^{1/2}} \quad \text{(ratio of posterior to prior width)} $$

[A diagram shows two bell curves on the same axis. One is wide and labeled "prior". The other is much narrower, taller, and centered at the same point, labeled "post".]

> From inference book:
> * factor by which our model's hypothesis space collapses when the data arrive
> * magnitude of Occam factor is a measure of complexity of the model
> * depends on both
>     * no. of free params in model
>     * prior prob model assigns to those params
>
> more params -> multiply factor by N -> more complex model? less simple model

> makes sense by itself?
> ($P(D|M_1) / P(D|M_2)$)
> volume of params space for each model. Balance between maximising data fit so we minimise complexity comparisons.

> Simple models concentrate probability around a limited no of datasets. Complex models must make range of datasets - best probability of a given dataset will be smaller i.e. even if both simple & complex model can predict data, complex will have much smaller prob of

---
(Page 153)

> for nested models?
> useful method that simple model $M_0$ is sub-model of complex model.

### Savage-Dickey Ratio

Suppose the parameters are $\phi, \psi$ and complex model $M_1$ reduces to simpler model $M_0$ when $\psi=0$. (Nested)
> e.g. cosmological params if some set to zero, then we get simple params (ΛCDM?)
> or e.g. how many terms for polynomial fit? constants=0

And the prior is separable: $P(\phi, \psi | M_1) = P(\psi | M_1) P(\phi | M_1)$
> nuisance parameters? (dependent?)
> assume independence (for this formulation)
> see parameters in common between model and submodel (unknown)

And the prior on $\phi$ is the same for each model:
$$ P(\phi|M_1) = P(\phi|M_0) $$
> prior on nuisance parameters

Then the Bayes Factor reduces to:
$$ B_{01} = \frac{P(D|M_0)}{P(D|M_1)} = \frac{P(\psi|D,M_1)}{P(\psi|M_1)} |_{\psi=0} $$
> Show this:
> * consider numerator: $P(D|M_0) = \int P(D|\phi, M_0) P(\phi|M_0) d\phi = \int P(D|\phi, \psi=0, M_1) P(\phi|M_1) d\phi$.
> * $P(\psi=0|D, M_1) = P(\psi=0|D,M_1) P(D|M_1) / (P(D|M_1))$ (wrong)
> * num part becomes $P(\psi=0|D, M_1) P(D|M_1)$

[A complex diagram with several plots. On the left, a plot with vertical axis $\psi$ and horizontal axis $P(D|\psi,M_i)$ shows two overlapping vertical distributions for $P(D|M_1)$ and $P(D|M_2)$. On the right, a horizontal axis labeled $\theta$ has four sets of distributions above it for models $M_0, M_1, M_2, M_3$. Each set shows a wider prior distribution (e.g., $P(\theta|M_1)$ prior) and a narrower posterior distribution (e.g., $P(\theta|D,M_1)$ posterior). The posteriors are shifted relative to the priors. A dashed line connects the peaks of the posterior distributions. Annotations are scattered around the plots.]

> FROM INFERENCE BOOK. How does... without... relate to model complexity?

> For this data, evidence of model $M_0$ is greater than for $M_3$ because area under curve for $M_3$ is greater.
> i.e. $M_0$ has Occam factor $\psi$? simple model but has the worse fit. $M_1$ has the best fit but posterior is still poor in comparison to prior.

> if range of $\psi$ for $M_3$ is small we use wider prior for complex model

---
(Page 154)

### NESTED SAMPLING
> for evidence calculation but also for posterior samples with byproduct!

* An algorithm designed to compute Bayesian evidence (John Skilling, 2004)
* Can be better than MCMC for multi-modal distributions
* Get weighted samples from posterior for free
* Evolve an ensemble of "live points" successively sampling prior volume above a likelihood-level constraint
* Perform evidence integral over contours of equal likelihood
* Uses statistical estimate of prior mass within likelihood-level

---
(Page 155)

Want to calculate multi-dim. evidence integral
$$ Z = \int L(\Theta)\pi(\Theta)d\Theta $$

Define "prior mass enclosed above likelihood $L^*$"
$$ X(L^*) = \int_{L(\Theta)>L^*} \pi(\Theta) d\Theta $$

Inverse: Likelihood at enclosed prior mass: $L(X)$

1D evidence integral: $$Z = \int L(X) dX$$
> (= likelihood value (from X=0) start high initially?)

[Two diagrams are shown side-by-side. The left diagram is labeled "PARAMETER SPACE $\Theta$". It shows a square with three nested, irregular, concentric contour lines labeled $\Theta_1, \Theta_2, \Theta_3$ from outermost to innermost. An arrow points from outside the contours to just inside the $\Theta_3$ contour, labeled $X(L_3)$. The right diagram has a vertical axis labeled "ENCLOSING LIKELIHOOD L" and a horizontal axis labeled "ENCLOSED PRIOR MASS X", with X ranging from 0 to 1. A monotonically decreasing curve labeled $L(X)$ is drawn. The area under the curve is divided into vertical colored bands corresponding to likelihood levels $L_1, L_2, L_3$ and prior mass values $X_1, X_2, X_3$.]

Nested likelihood contours sort to enclosed prior mass X.
* e.g. 2D example take 3 random points from prior $\Theta_1, \Theta_2, \Theta_3$ and calc. their likelihood, each will be associated with some likelihood contour. each contour is associated with some prior value & the prior volume X(L?).
* So can associate with each point a likelihood contour and also a prior mass enclosed above that likelihood.
* If we can map from points in param. space to L(X), I can turn a previously multi-dim integral in param space to a 1-Dim integral in X from 0 to 1.

---
(Page 156)

### Nested Sampling: How?
> how do we achieve this?

[NB: For the rest of this lecture assume prior is uniform box in parameter space (w/ total integral 1)]

1. Begin by sampling $N_{live}$ points $\{\Theta_i\}$ from the prior ($X_0=1$), evaluate likelihood for each live point, initialise evidence $Z=0$.

2. Find the live point with smallest likelihood $L^*$.

3. Estimate compression factor $t_i = X_i/X_{i-1}$.
> volume is estimated statistically. $t_i \in (0, 1)^{N_{live}}$

4. Accumulate evidence as a 1D integral
$$ \Delta Z = L^* \times (X_{i-1} - X_i) = L^* \times (1-t_i)X_i $$

5. Sample a new live point from prior constrained to $\{\Theta | L > L^*\}$, compute its likelihood value.
> (e.g. with MCMC or otherwise) (easy to draw points?)
> reject anything that does not satisfy constraint
> simple reject & get?

6. Repeat steps 2-5 until convergence (i.e. evidence $Z$ stops changing within some tolerance).

7. Add final evidence estimate of remaining live points $\Delta Z = \bar{L} \times X_{end}$.

8. List of dead points $\{\Theta_i\}$ give weighted posterior samples. $$w_i \propto (1-t_i)X_i$$