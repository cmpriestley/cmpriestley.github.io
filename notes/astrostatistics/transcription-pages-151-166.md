---
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