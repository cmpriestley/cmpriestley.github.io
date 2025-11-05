---
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

*   Nested Sampling***