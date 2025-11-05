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
> weight roughly skewed distribution.