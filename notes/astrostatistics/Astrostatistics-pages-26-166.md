# Astrostatistics - Pages 26-166
*Transcribed from handwritten lecture notes using Gemini AI*

This file contains the continuation of astrostatistics lecture notes starting from page 26.

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
$M_s = M_s + \mu_s$ (**latent variable equation**)

Define: $\hat{M}_s = \hat{M}_s - \hat{\mu}_s$ (Estimated Abs Mag)
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
> Begin to think about nuisance, see how things begin to become tricky
Same eq. but now
Assume **measurement error = 0** = $\sigma_{ms} \geq \sigma_{\mu s}$.
Population variability $\sigma_{int} \rightarrow \sigma$.
**Only intrinsic population** is source of variability $(\sigma^2 = \sigma^2_{int})$.
> e.g. sampling q. of quasars?

Suppose **distance** is the **same** for an entire sample of SN $\mu_s = \mu$.

population dist. iid.
$M_s \sim N(M_0, \sigma^2) \quad$ (latent variable eq. $M_s = S + \mu_s$)
combine.
$\rightarrow \hat{M}_s \sim N(M_0 + \mu, \sigma^2) \quad s=1,...,N$

[Image of a plot with x-axis 'm (dimmer)' and y-axis '$\mu$ (fainter)'. A diagonal dashed line represents $\mu = m - M_0$. Two Gaussian distributions are shown. The top one, at $\mu_2$, has some of its right tail cut off by a vertical dashed line at $m_{lim}$ labeled "survey limit". The observed SN are shown as 'x's to the left of the limit, and unobserved SN are 'x's to the right. The peak of this Gaussian is at $m_2 = M_0 + \mu_2$. The bottom Gaussian, at $\mu_1$, is fully to the left of the limit. Its peak is at $m_1 = M_0 + \mu_1$. A region to the right of $m_{lim}$ is labeled "to faint to be observable".]
> observed SN
> unobserved SN

> e.g. scope is only going to detect up to certain magnitude limit, can't see anything dimmer than this. Surveys looking for SNe at any magnitude look similar, but in the end we don't see the one above some limit, what do we do?

---
(Page 32)

**Data**: $\{\hat{M}_s\} = \hat{\mathbf{m}}$
fixed $\mu$ for sample

### Naive Likelihood:
$$ P(M_s|M_0, \sigma^2) = N(M_s|\mu+M_0, \sigma^2) $$
$$ L(M_0, \sigma^2) = \prod_{s=1}^N P(M_s|M_0, \sigma^2) $$

> naively apply naive likelihood
If $M_0+\mu$ close to $m_{lim}$, MLE will be **biased**:
$$ \left\{ \begin{array}{l} \hat{M}_0 \text{ too bright} \\ \hat{\sigma} \text{ too small} \end{array} \right. $$
> (we didn't see m,dim everywhere)
> (b/c tail of dist. is cut off. different width, smaller than true criteria)

> how do we resolve this?
### Accounting for selection Effects (sheet 1, Q2)

Let $I_s = \begin{cases} 1, & \text{if SN observed} \\ 0, & \text{if SN NOT observed} \end{cases}$
> indicator variable

Define **Selection Function**:
> prob. indicator given data
$$ P(I_s|M_s) = \begin{cases} 1, & M_s < M_{lim} \\ 0, & M_s \ge M_{lim} \end{cases} $$
$$ = 1 - H(M_s - M_{lim}) $$
> heaviside step fn

[Image of a step function plot. X-axis is $m_{lim}$, Y-axis is unlabeled. The function is 1 for values less than $m_{lim}$ and 0 for values greater than or equal to $m_{lim}$.]

---
(Page 33)


## Lecture 9
_12.2.25_

> (beg recap SN problem setup on board - useful for sheet!)

### SELECTION EFFECTS cont.
recap:
$M_s = N(M_0, \sigma^2)$
$s=1,...,N$ (known)
$\mu_s = \mu$ (known)
> known, effectively no measurement error

[Image of a Gaussian distribution on an x-axis labeled 'm'. The peak is at M. The right tail of the Gaussian is cut off by a vertical line at $M_{lim}$, and observed points ('x's) are to the left of the line.]

Formulate likelihood to account for selection effects:
Let $I_s = \begin{cases} 1, & \text{if SN observed} \\ 0, & \text{if SN NOT observed} \end{cases}$

**Selection function**: $P(I_s=1|m_s)$:
$$ S(m_s) = P(I_s|m_s) = \begin{cases} 1, & m_s < M_{lim} \\ 0, & m_s \ge M_{lim} \end{cases} $$

[Image of a step function plot. X-axis is 'm', y-axis is $S(m_s)$. The function is 1 for values up to $M_{lim}$ and 0 after. Labeled $1-H(m_s-M_{lim})$.]

---
(Page 34)

### Observed Data Likelihood:
$$ P(m_s|I_s=1, \theta) = \frac{P(I_s=1, m_s|\theta)}{P(I_s=1|\theta)} $$
$$ = \frac{P(I_s=1|m_s, \theta)P(m_s|\theta)}{\int P(I_s=1|m_s, \theta)P(m_s|\theta) dm_s} $$
$$ = \frac{S(m_s)N(m_s|M_0+\mu, \sigma^2)}{\int S(m_s)N(m_s|M_0+\mu, \sigma^2) dm_s} $$
> cut off above $M_{lim}$
$$ = \frac{[1-H(m_s-M_{lim})]N(m_s|M_0+\mu, \sigma^2)}{\int_{-\infty}^{M_{lim}} N(m_s|M_0+\mu, \sigma^2) dm_s \quad \rightarrow \Phi \text{ Gaussian CDF}} $$
> to normalise
> scale this upon denominator
$$ (\text{TRUNCATED}) = \text{TN}(m_s|M_0+\mu, \sigma^2, -\infty, M_{lim}) $$
**NORMAL**
$P(m_s|I_s=1, \theta)$
[Image of a truncated normal distribution. A full Gaussian is shown dashed, and the part to the left of a vertical line at $M_{lim}$ is solid. The peak is at $M_0+\mu$. Labels: untruncated mean & variance, lower truncation limit, upper truncation limit.]

### Challenge: what if $S(m_s) = P(I_s=1|m_s) = \Phi\left(\frac{M_{lim}-m_s}{\sigma_{lim}}\right)$?
> limit but a broad boundary, to take into account e.g. observer bias / seeing conditions / cloudy days?
[Image of a sigmoid function (CDF of a Gaussian). X-axis is 'm', y-axis is not labeled. The function decreases from 1 to 0 around $M_{lim}$. A Gaussian centered at $M_{lim}$ with width $\sigma_{lim}$ is shown below the curve. It is labelled $S(m_s)$ and 'A' on the y-axis, and $m_{lim}$ on the x-axis.]

---
(Page 35)

> (Slides)
> Ex sheet 1 q 2: star formation in Perseus
> 3 in densest cloud regions, stars are hidden
> 3 in densest regions will be where target stars are -> selection effect
> Pareto or power law distribution $P(m) \propto m^{-\alpha}$ (for $m>m_0$)
> [A sketch of two likelihood curves on an axis labeled 'power law exponent'. One curve, labeled 'true likelihood', has a peak. The other, labeled 'naive MLE -> biased', is shifted, indicating it "does not account for selection effects".]

### QUANTIFYING UNCERTAINTY USING BOOTSTRAP

### Frequentist interpretation:

Consider **variability** of your estimator $g(\vec{x})$ for $\theta$ under (imaginary) repetitions of your experiment. (Random realisations of the potential data).

How does $g(\vec{x})$ behave under the potential datasets you **did not** observe?
e.g. $\mathrm{Var}[g(\vec{x})] = E[(g(\vec{x}) - E[g(\vec{x})])^2]$
> under $P(x|\theta)$

If $g(\vec{x})$ is approximately Gaussian distributed.
$\Rightarrow$ 68% **confidence interval** $g(\vec{x}) \pm \sqrt{\mathrm{Var}(g(\vec{x}))}$
> standard deviation interval
> $[g(\vec{x}) - \sqrt{\mathrm{Var}(g(\vec{x}))}, g(\vec{x}) + \sqrt{\mathrm{Var}(g(\vec{x}))}]$

$(1-\alpha)\%$ confidence interval $[L(\vec{x}), U(\vec{x})]$ contains the true value $\theta_{true}$ in at least $(1-\alpha)\%$ of the realisations.
> across repeated experiments, fraction of intervals that contain the parameter will approach $(1-\alpha)\%$
> NOT this -> probability interval contains the true parameter (common misconception)

---
(Page 36)

> get diff interval for each experiment -> CI encapsulates interval -> 68% of those realisations contain $\theta$.
[Image of three confidence intervals as horizontal lines. A vertical line represents the true value $\theta$. Two of the intervals cross the line, one does not.]

$[L(x), U(x)]$ is a **random interval** vs. $[L(x_{obs}), U(x_{obs})]$ evaluated on observed numerical values, only one dataset $x_{obs}$! (either contains $\theta_{true}$ or doesn't).

### Bootstrap:
Use the **observed dataset** to **simulate** the variability of the **unobserved** (imaginary) data sets.

**BOOTSTRAP SAMPLE** = **sample with replacement** from the observed dataset to the sample size.

e.g. $X_1, ..., X_5 \stackrel{\text{iid}}{\sim} \text{Poisson}(\lambda)$
$\rightarrow P(X_i) = \frac{\lambda^{X_i} e^{-\lambda}}{X_i!}$
> could do w/ max. likelihood but suppose don't know dist.

Real data (observed) $\vec{X}_{obs} = (3, 8, 2, 4, 5)$.

Suppose you want to estimate the **skewness** of $P(x)$ (asymmetry).
> true mean -> = $\lambda$
i.e. skewness $= \frac{E[(x-\mu)^3]}{(\sigma^2)^{3/2}}$
> true variance

Sample skewness $g(\vec{x}) = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^3}{\left(\sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2}\right)^3}$

---
(Page 37)

Bootstrap B "replicate" datasets from observed dataset:
> "sample w/ replacement from original eg. sample 5 times from 3,8,2,4,5 gives..."

$\vec{X}^{obs} = (3, 8, 2, 4, 5) \rightarrow \hat{g}_{obs} = g(\vec{x}^{obs}) = 0.6927$
$\vec{X}^{b=1} = (2, 5, 4, 4, 4) \rightarrow \hat{g}_1 = g(\vec{x}^{b=1}) = -0.8625$
$\vec{X}^{b=2} = (2, 4, 2, 8, 8) \rightarrow \hat{g}_2 = g(\vec{x}^{b=2}) = 0.2115$
$\vec{X}^{b=3} = (5, 2, 8, 2, 5) \rightarrow \hat{g}_3 = g(\vec{x}^{b=3}) = 0.3436$
...
$\vec{X}^{b=B} = \dots \rightarrow \hat{g}_B = \dots$

Can now compute sample variance
$\widehat{\mathrm{Var}}(\{\hat{g}_1, ..., \hat{g}_B\})$
Standard error $= \sqrt{\widehat{\mathrm{Var}}}$
$$ \hat{g} = 0.6927 \pm 0.635 \approx 68\% \text{ C.I. } \checkmark $$

> (slides)
> back to supernova plus example:
> how do I get standard error on param?
> can try Fisher matrix approach and compare this against bootstrap
> get model to each bootstrapped realisation
> plot max likelihood estimates for each as histograms
> [Image of two small histograms. The left one is labeled "true" and shows a wide distribution. The right one is labeled "naive" and shows a narrower distribution, with an arrow pointing to it labeled "(underestimates)". Another arrow points to the naive histogram with the comment "(this is a sketch, not necessarily true)".]

---
(Page 38)


## Lecture 10
_14.2.25_

### REGRESSION (slides)

* Fit a function $E[y|x] = f(x;\theta)$ for the **mean relation** between $y$ and $x$.
* Basic approaches
  - $\rightarrow$ **ordinary least squares** (homoskedastic scatter)
  - $\rightarrow$ **generalized least squares** (heteroskedastic, correlated scatter)
  - $\rightarrow$ **weighted least squares** (minimum $\chi^2$, known variance)
  - $\rightarrow$ **maximum likelihood**
* Real data problems require more complex modelling
  - $\rightarrow$ **regression dilution** from covariate measurement errors

[A scatter plot with x-axis "luminosity" and y-axis unlabeled. Data points with error bars are shown. A straight line is fitted through them.]
> e.g. each point has diff measurement error. regression dilution; naively applying OLS gives biased fit when there are errors in x (sheet 2)

### Ordinary least squares (OLS):
Linear model $y_i = \beta_0 + \sum_{j=1}^{k-1} \beta_j X_{ij} + \mathcal{E}_i$
$$ Y = X\beta + \mathcal{E} $$
$i=1, ..., N$ objects, $E[\mathcal{E}_i]=0$, **homoskedastic**
$\mathrm{Var}[\mathcal{E}_i] = \sigma^2$ (known).

---
(Page 39)

