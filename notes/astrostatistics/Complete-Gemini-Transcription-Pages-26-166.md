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
