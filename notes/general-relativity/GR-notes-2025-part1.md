An OCR of the provided PDF document is below.

***

### Page 1

==Start of Transcription for page 1==

# GENERAL RELATIVITY
<div style="text-align:right">
office hours <br>
8.40 (20 mins before lecture S)
</div>

Prof Claude Warnick (Printed notes by Prof Harvey Reall)
<div style="text-align:right">
(Is also look at Tony notes y although slightly dif.).
</div>

General relativity is our best theory of gravitation on the largest scales!
It is:
- **CLASSICAL**: No quantum effects
- **GEOMETRICAL**: Space + time are combined in a curved spacetime
- **DYNAMICAL**: In contrast to Newton's theory of gravity, Einstein's gravitational field has its own non-trivial dynamics

## Differentiable Manifolds

The basic object of study in differential geometry is the (differentiable) manifold. This is an object which "locally looks like $\mathbb{R}^n$" and has enough structure to let us do calculus.

**DEF:** A differentiable manifold of dimension n is a set M, together with a collection of coordinate charts $(O_\alpha, \phi_\alpha)$ where
- $O_\alpha \subset M$ are subsets of M such that $\bigcup O_\alpha = M$.
- $\phi_\alpha$ is a bijective map (one-to-one and onto) from $O_\alpha$ to $U_\alpha$, an open subset of $\mathbb{R}^n$.
- If $O_\alpha \cap O_\beta \neq \emptyset$, then $\phi_\beta \circ \phi_\alpha^{-1}$ is a smooth (infinitely differentiable) map from $\phi_\alpha(O_\alpha \cap O_\beta) \subset U_\alpha$ to $\phi_\beta(O_\alpha \cap O_\beta) \subset U_\beta$.
> *Annotation on "If $O_\alpha \cap O_\beta \neq \emptyset$":* their intersection is not empty
> *Annotation on "open subset":* originally

*Diagram depicting a manifold M with two overlapping charts, Oα and Oβ, mapping to subsets Uα and Uβ in Rⁿ via maps φα and φβ respectively.*
> *Annotations on the diagram:*
> - *Condition between these regions -> $\phi_\alpha$ inverse then $\phi_\beta$ I want this to be smooth (shortcut the loop)*
> - *image of $O_\alpha \cap O_\beta$ when mapped with $\phi_\alpha$*
> - *image of $O_\alpha \cap O_\beta$ when mapped with $\phi_\beta$*
> - *one-to-one and onto: each point from LHS maps to exactly one point on RHS*

**REMARKS:**
- Could replace smooth with finite differentiability (e.g. k-times differentiable)
- The charts define a topology on the original manifold M: a subset $\Omega \subset M$ is open iff $\phi_\alpha(\Omega \cap O_\alpha)$ is open in $\mathbb{R}^n$ for all $\alpha$.
> *Annotation:* I think this is maybe not quite right, maybe just interesting that you do have a topology
- Every open subset of M is itself a manifold (Restrict charts to $\Omega$)
> *Annotation:* you don't need to do topology?

---
### Page 2

==Start of Transcription for page 2==

The collection $\{(O_\alpha, \phi_\alpha)\}$ is called an **atlas**. Two atlases are **compatible** if their union is an atlas.

An atlas A is **maximal** if there exists no atlas B with A $\subsetneq$ B (subset but not equal to). Every atlas is contained in a maximal atlas (consider the union of all compatible atlases). We can assume without loss of generality that we work with a maximal atlas.

### EXAMPLES

1. If $U \subset \mathbb{R}^n$ is open, we can take $O=U$, $\phi(x^1, ..., x^n) = (x^1, ..., x^n)$. $\phi: O \to U$. $\{(U, \phi)\}$ is an atlas.

2. ($2D$ sphere?)
$S^1 = \{p \in \mathbb{R}^2 \mid \|p\|=1\}$
*Diagram showing a circle with a red arc and a blue arc, mapping to two separate intervals on the real line.*
> *Annotations:*
> - *R² (the circle, one can... describe coords, chains using angles around the circle)*

If $\{p \in S^1 \setminus \{(1,0)\}\} = O_1$, there is a unique $\theta_1 \in (-\pi, \pi)$ s.t. $p = (\cos\theta_1, \sin\theta_1)$.
> *Annotation:* open interval (excluding $-\pi, \pi$)
If $\{p \in S^1 \setminus \{(1,0)\}\} = O_2$, then there is a unique $\theta_2 \in (0, 2\pi)$ s.t. $p = (\cos\theta_2, \sin\theta_2)$.
> *Annotation:* open interval (minus/not including that point)

$\phi_1: p \to \theta_1$, $p \in O_1$, $U_1 = (-\pi, \pi)$
$\phi_2: p \to \theta_2$, $p \in O_2$, $U_2 = (0, 2\pi)$

EXERCISE
$\phi_1(O_1 \cap O_2) = (-\pi, 0) \cup (0, \pi)$ (open set)
$\phi_2 \circ \phi_1^{-1}(\theta) = \begin{cases} \theta & \theta \in (-\pi, 0) \\ \theta + 2\pi & \theta \in (0, \pi) \end{cases}$
> *Annotations:* transition function on this set, (composite function?), where defined?
Smooth where defined. Similarly for $\phi_1 \circ \phi_2^{-1}$.
$S^1$ is a manifold.

3. higher dimensional sphere
$S^n = \{p \in \mathbb{R}^{n+1} \mid \|p\|=1\}$.
*Diagram of a sphere.*
Define charts by stereographic projection. If $\{E_1, ..., E_n\}$ is a standard basis for $\mathbb{R}^{n+1}$ and $\{e_1, ..., e_n\}$ the basis for $\mathbb{R}^n$, write $p = p^1 E_1 + ... + p^{n+1} E_{n+1}$.
> *Annotation:* point on surface of sphere

---
### Page 3

==Start of Transcription for page 3==

Set $O_1 = S^n \setminus \{E_{n+1}\}$
> *Annotation:* sphere minus north pole
$\phi_1(p) = \frac{1}{1-p^{n+1}}(p^1 e_1 + ... + p^n e_n)$

$O_2 = S^n \setminus \{-E_{n+1}\}$
$\phi_2(p) = \frac{1}{1+p^{n+1}}(p^1 e_1 + ... + p^n e_n)$

*Diagram illustrating stereographic projection from the north pole of a sphere onto a plane.*
> *Annotation:* 1st map takes point p & projects onto plane transverse to n+1 direction through the north pole. Other map does the same through the south pole.

Claim: $\phi_1(O_1 \cap O_2) = \mathbb{R}^n \setminus \{0\}$ and $\phi_2 \circ \phi_1^{-1}(x) = \frac{x}{\|x\|^2}$.
> *Annotation:* excluding the origin
> *Derivation in margin:*
> Let $\phi_1(p) = x$, then we see $p = \frac{2x}{1+\|x\|^2} + \frac{\|x\|^2-1}{\|x\|^2+1}E_{n+1}$.
> Then since point p on $S^n$ sphere has both $p^{n+1} = \frac{\|x\|^2-1}{1+\|x\|^2}$ and $(p')^2 + (p^{n+1})^2 = 1$
> and so $\|p'\| = \frac{2\|x\|}{1+\|x\|^2}$.
> $\phi_2 \circ \phi_1^{-1}(x) = \frac{1}{1+p^{n+1}} p' = \frac{1}{1+\frac{\|x\|^2-1}{1+\|x\|^2}} \frac{2x}{1+\|x\|^2} = \frac{1+\|x\|^2}{2\|x\|^2} \frac{2x}{1+\|x\|^2} = \frac{x}{\|x\|^2}$
> which is smooth

smooth on $\mathbb{R}^n \setminus \{0\}$.
similar for $\phi_1 \circ \phi_2^{-1}$. $S^n$ is an n-manifold.

$p = p^1 E_1 + p^2 E_2 + p^3 E_3$.
$v_1(p) = \frac{1}{1-p^3}(p^1 e_1 + p^2 e_2)$.

*Diagrams illustrating the 2-sphere case.*

---
### Page 4

==Start of Transcription for page 4==

### Lecture 2
<div style="text-align:right">14.10.24</div>

#### Smooth functions on manifolds

Suppose M,N are manifolds of dim n, n' respectively.
Let $f: M \to N$.
Then let $p \in M$ and pick charts $(O_\alpha, \phi_\alpha)$ for M and $(O_\beta, \phi_\beta)$ for N with $p \in O_\alpha$, $f(p) \in O_\beta$. Then $\phi_\beta \circ f \circ \phi_\alpha^{-1}$ maps an open neighbourhood of $\phi_\alpha(p)$ in $U_\alpha \subset \mathbb{R}^n$ to $U_\beta \subset \mathbb{R}^{n'}$.
If this function is smooth for all possible choices of chart, we say $f: M \to N$ is **smooth**.

*Diagram showing manifolds M and N, with charts Oα, Oβ and their mappings to Uα, Uβ via φα, φβ. A point p in M maps to f(p) in N. The composition map $\phi_\beta \circ f \circ \phi_\alpha^{-1}$ goes from Uα to Uβ.*
> *Annotations:*
> - f: M -> R ... says F=foφα⁻¹: Uα -> R all f on notes is smooth
> - f: M -> N smooth if for all choices of chart, φβ o f o φα⁻¹ smooth

**COMMENTS:**
- A smooth map $\psi: M \to N$ which has a smooth inverse is called a **diffeomorphism** ($n=n'$)
- If $N = \mathbb{R} / \mathbb{C}$ we sometimes call $f$ a **scalar field**
- If $M = I \subset \mathbb{R}$ an open interval then $f: I \to N$ is a **smooth curve** in N
- If f is smooth in one atlas, it is smooth in all compatible atlases.

**EXAMPLES:**
1. Recall $S^1 = \{x \in \mathbb{R}^2 \mid \|x\|=1\}$. Let $f(x,y) = x$. $f: S^1 \to \mathbb{R}$.
Using previous charts
$f \circ \phi_1^{-1}: (-\pi, \pi) \to \mathbb{R}$
*Diagram of a circle on the xy-plane.*

---
### Page 5

==Start of Transcription for page 5==

$f \circ \phi_1^{-1}(\theta_1) = \cos\theta_1$
Similarly $f \circ \phi_2^{-1}: (0, 2\pi) \to \mathbb{R}$
$f \circ \phi_2^{-1}(\theta_2) = \cos\theta_2$. $\therefore f$ is **smooth**.
> *Notes in margin:* Using other chart, then $f \circ \phi_2^{-1} = f \circ \phi_1^{-1} \circ (\phi_1 \circ \phi_2^{-1})$ which is smooth since we have shown $f \circ \phi_1^{-1}$ is smooth & $\phi_1 \circ \phi_2^{-1}$ is smooth by definition of a manifold.

2. If $(O, \phi)$ is a coordinate chart on M, write $\phi(p) = (x^1(p), x^2(p), ..., x^n(p))$, $p \in O$.
Then $x^i(p)$ defines a map from $O$ to $\mathbb{R}$. This is a **smooth** for each $i=1, ..., n$. If $(O', \phi')$ is another overlapping coordinate chart then $x^i \circ (\phi')^{-1}$ is the i$^{th}$ component of $\phi \circ (\phi')^{-1}$, hence **smooth**.
> *Annotation:* $f \circ \phi'^{-1}$ is smooth because it is the $i^{th}$ component of a smooth function.

3. We can define a smooth function chart-by-chart.
For simplicity $N=\mathbb{R}$, let $\{(O_\alpha, \phi_\alpha)\}$ be an atlas on M. Define smooth functions (coordinate chart representations) $F_\alpha: U_\alpha \to \mathbb{R}$ and suppose $F_\alpha \circ \phi_\alpha = F_\beta \circ \phi_\beta$ on $O_\alpha \cap O_\beta$ for all $\alpha, \beta$.
Then for $p \in M$ we can define $f(p) = F_\alpha(\phi_\alpha(p))$ where $(O_\alpha, \phi_\alpha)$ is any chart with $p \in O_\alpha$. $f$ is **smooth** as $f \circ \phi_\beta^{-1} = F_\alpha \circ \phi_\alpha \circ \phi_\beta^{-1}$ if $O_\alpha \cap O_\beta \neq \emptyset$.
> *Annotation:* smooth since $\phi_\alpha \circ \phi_\beta^{-1}$ smooth

In practice, often don't distinguish between $f$ and its coordinate chart representations $F_\alpha$.

### CURVES AND VECTORS

For a surface in $\mathbb{R}^3$ we have a notion of **tangent space** at a point consisting of all vectors tangent to the surface.
*Diagram of a surface in 3D with two tangent vectors shown at a point.*
*General picture diagram showing a curve λ: I -> M and a function f: M -> R. Also shown are the chart mappings.*
> *Annotations on diagram:*
> - *λ is a map λ: I -> M. λ(t) is a curve in M.*
> - *φ is a map φ: M -> Rⁿ. φ(p) is a point in Rⁿ.*
> - *we see $F \circ \phi \circ \lambda(t) = f \circ \phi^{-1} \circ \phi \circ \lambda(t) = f(\lambda(t))$.*
> - *So we move from the manifold to Rⁿ where we can do differentiation & and eval. $\frac{d}{dt}(f(\lambda(t)))$*

---
### Page 6

==Start of Transcription for page 6==

The tangent spaces are vector spaces (copies of $\mathbb{R}^n$).
Different points have different tangent spaces.

In order to define the tangent space for a manifold we first consider tangent vectors of a curve.

**Recall** $\lambda: I \to M$ is a smooth map, is a **smooth curve** in M.
> *Annotation:* a smooth curve in a manifold M is a smooth function $\lambda: I \to M$ where I is an open interval on the reals.

If $\lambda(t)$ is a smooth curve in $\mathbb{R}^n$ and $f: \mathbb{R}^n \to \mathbb{R}$ is a smooth function. The chain rule gives
$$ \frac{d}{dt}[f(\lambda(t))] = \dot{\lambda}(t) \cdot \nabla f(\lambda(t)) $$
where $\dot{\lambda}(t) = \frac{d\lambda}{dt}(t)$ is the tangent vector to $\lambda$ at $t$.

**IDEA:** identify $\dot{\lambda}(t)$ with $\dot{\lambda}(t) \cdot \nabla$
> *Annotation:* this map takes $f \to \dot{\lambda} \cdot \nabla f$

**DEF** Let $\lambda: I \to M$ be a smooth curve with (wlog) $\lambda(0)=p$.
The tangent vector to $\lambda$ at $p$ is the linear map $X_p$ from the space of smooth functions $f: M \to \mathbb{R}$ given by
$$ X_p(f) := \left.\frac{d}{dt}f(\lambda(t))\right|_{t=0} $$
> *Annotation:* $X_p$ is the vector, $X_p(f)$ specifies how the vector acts on a function, it is a number. map that takes $f$ to $\frac{d}{dt}f = X\cdot \nabla f$

**We observe:**
i) $X_p$ is linear: $X_p(f+ag) = X_p f + a X_p g$ for $g$ smooth, $a \in \mathbb{R}$.
ii) $X_p$ satisfies Leibniz rule: $X_p(fg) = (X_p f)g(p) + f(p)X_p g$.

If $(O, \phi)$ is a chart $p \in O$, write $\phi(p) = (x^1(p), ..., x^n(p))$.
Let $F = f \circ \phi^{-1}$ and $x(t) = \phi(\lambda(t))$.
Then $f(\lambda(t)) = f \circ \phi^{-1} \circ \phi(\lambda(t)) = F \circ x(t)$ and
$$ \left.\frac{d}{dt}(f(\lambda(t)))\right|_{t=0} = \left.\frac{\partial F}{\partial x^\mu}(\phi(p))\frac{dx^\mu}{dt}\right|_{t=0} $$
> *Annotation:* Einstein summation convention: sum over repeated $\mu=1, ..., n$.

---
### Page 7

==Start of Transcription for page 7==

### Lecture 3
<div style="text-align:right">16.10.24</div>

$ (*) \quad X_p(f) = \frac{\partial F}{\partial x^\mu}(\phi(p)) \left.\frac{dx^\mu}{dt}\right|_{t=0} $

**Prop** The set of tangent vectors to curves at P forms a vector space $T_p M$ of dimension $n=\dim M$. We call $T_p M$ the **tangent space** to M at P.
> *Annotation:* dim of manifold

**Pf/** Given $X_p, Y_p$ tangent vectors, we need to show $\alpha X_p + \beta Y_p$ is a tangent vector for $\alpha, \beta \in \mathbb{R}$.
Let $\lambda, \kappa$ be smooth curves with $\lambda(0)=\kappa(0)=p$ and whose tangent vectors at $p$ are $X_p, Y_p$ resp.
Let $(O, \phi)$ be a chart with $p \in O$, $\phi(p)=0$ (chart centered at p).

Let $V(t) = \phi^{-1}[\alpha\phi(\lambda(t)) + \beta\phi(\kappa(t))]$
$V(0) = \phi^{-1}(0) = p$.

From (*) we have that if $Z_p$ is the tangent to $V$ at $p$:
$$ Z_p(f) = \left.\frac{d}{dt}(f(V(t)))\right|_{t=0} = \left.\frac{\partial F}{\partial x^\mu}\right|_0 \left.\frac{d}{dt}[\alpha x^\mu(\lambda(t)) + \beta x^\mu(\kappa(t))]\right|_{t=0} $$
$$ = \alpha \left.\frac{\partial F}{\partial x^\mu}\right|_0 \left.\frac{dx^\mu}{dt}(\lambda(t))\right|_{t=0} + \beta \left.\frac{\partial F}{\partial x^\mu}\right|_0 \left.\frac{dx^\mu}{dt}(\kappa(t))\right|_{t=0} $$
$$ = \alpha X_p(f) + \beta Y_p(f) $$
Thus $T_p M$ is a vector space.

To see $T_p M$ is n-dimensional consider the curves
$$ \lambda_\mu(t) = \phi^{-1}(0,...,0,t,0,...,0) \quad (\mu^{th} \text{ component}) $$
We denote the tangent vector to $\lambda_\mu$ at p by $(\frac{\partial}{\partial x^\mu})_p$.
To see why, note that (*):
$$ (\frac{\partial}{\partial x^\mu})_p f = \frac{\partial F}{\partial x^\nu}(\phi(p))\delta^\nu_\mu = \frac{\partial F}{\partial x^\mu}(\phi(p)=0) $$
> *Annotation:* think of it like "basis" vector in $\mathbb{R}^n$ getting moved to M (see pic below).

*Diagram showing a grid in Rⁿ mapping to a curved grid on the manifold M.*

---
### Page 8

==Start of Transcription for page 8==

The vectors $(\frac{\partial}{\partial x^\mu})_p$ are linearly independent. Otherwise, $\exists \alpha^\mu \in \mathbb{R}$ s.t. not all zero s.t. $\alpha^\mu(\frac{\partial}{\partial x^\mu})_p=0$.
$$ \Rightarrow \alpha^\mu \frac{\partial F}{\partial x^\mu}\big|_0 = 0 \quad \forall F. \quad \text{setting } F=x^\nu \text{ gives } \alpha^\nu=0. $$
Further, $(\frac{\partial}{\partial x^\mu})_p$ form a basis for $T_p M$, since if $\lambda$ is any curve with tangent $X_p$ at $p$, (*) gives
$$ X_p(f) = \left.\frac{\partial F}{\partial x^\mu}\right|_0 \left.\frac{d}{dt}x^\mu(\lambda(t))\right|_{t=0} = X^\mu (\frac{\partial}{\partial x^\mu})_p f $$
where $X^\mu = \left.\frac{d}{dt}x^\mu(\lambda(t))\right|_{t=0}$ are the **components** of $X_p$ w.r.t. the basis $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$ for $T_p M$.

Notice that $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$ depends on the coordinate chart $\phi$.
Suppose we choose another chart $(O', \phi')$, again centered at $p$. Write $\phi' = (x'^1, ..., x'^n)$. Then if $F = f \circ \phi^{-1}$ we have $F'(x') = f \circ (\phi')^{-1}(x') = f \circ \phi^{-1} \circ \phi \circ (\phi')^{-1}(x')$.
$F(x) = f \circ \phi^{-1}(x)$.
$F'(x') = F(x(x'))$.
so $(\frac{\partial}{\partial x'^\nu})_p f = \frac{\partial F'}{\partial x'^\nu}\big|_{\phi'(p)} = \frac{\partial F}{\partial x^\mu}\big|_{\phi(p)} \frac{\partial x^\mu}{\partial x'^\nu}\big|_{\phi'(p)} = \frac{\partial x^\mu}{\partial x'^\nu}\big|_{\phi'(p)} (\frac{\partial}{\partial x^\mu})_p f$
> *Annotation:* summation convention over $\mu$.

We deduce that
$$ (\frac{\partial}{\partial x'^\nu})_p = \frac{\partial x^\mu}{\partial x'^\nu}\big|_{\phi(p)} (\frac{\partial}{\partial x^\mu})_p $$
Let $X^\mu$ be components of $X_p$ w.r.t. $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$ and $X'^\mu$ be components of $X_p$ w.r.t. $\{(\frac{\partial}{\partial x'^\mu})_p\}_{\mu=1}^n$.
i.e. $X_p = X^\mu (\frac{\partial}{\partial x^\mu})_p = X'^\nu (\frac{\partial}{\partial x'^\nu})_p$
$$ = X'^\nu \frac{\partial x^\mu}{\partial x'^\nu}\big|_{\phi(p)} (\frac{\partial}{\partial x^\mu})_p $$

---
### Page 9

==Start of Transcription for page 9==

so $X^\mu = \frac{\partial x^\mu}{\partial x'^\nu}(\phi(p)) X'^\nu$.

We do not have to choose a coordinate basis such as $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$. With respect to a general basis $\{e_\mu\}_{\mu=1}^n$ for $T_p M$ we write $X_p = X^\mu e_\mu$ for $X^\mu \in \mathbb{R}$ are components w.r.t. $\{e_\mu\}_{\mu=1}^n$.
We always use summation convention: we always contract one upstairs and one downstairs index. The index on $\frac{\partial}{\partial x^\mu}$ counts as downstairs.

### COVECTORS

Recall that if V is a vector space over $\mathbb{R}$, the **dual space** $V^*$ is the space of linear maps from V to $\mathbb{R}$. If V is n-dimensional, so is $V^*$. Given a basis $\{e_\mu\}_{\mu=1}^n$ for V, we define the dual basis $\{\epsilon^\mu\}_{\mu=1}^n$ for $V^*$ by requiring $\epsilon^\mu(e_\nu) = \delta^\mu_\nu = \begin{cases} 1 & \mu=\nu \\ 0 & \mu\neq\nu \end{cases}$.
If V is finite dimensional then $V^{**} = (V^*)^*$ is isomorphic to V: to an element X of V we associate the linear map $\Lambda_X: V^* \to \mathbb{R}$, $\Lambda_X(\omega) = \omega(X)$, $\omega \in V^*$.

**Def**: The dual space of $T_p M$ is denoted $T_p^* M$ and called the **cotangent space** to M at p. An element of $T_p^* M$ is a **covector** at p. If $\{e_\mu\}_{\mu=1}^n$ is a basis for $T_p M$ and $\{\epsilon^\mu\}_{\mu=1}^n$ the dual basis for $T_p^* M$, we can expand a covector $\eta$ as $\eta = \eta_\mu \epsilon^\mu$ for $\eta_\mu \in \mathbb{R}$, the **components** of $\eta$.

---
### Page 10

==Start of Transcription for page 10==

### Lecture 4
<div style="text-align:right">18.10.24</div>

**Recap:**
- Defined $T_p M$: space of tangent vectors at p. Basis $\{e_\mu\}_{\mu=1}^n$.
- Coord basis $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$.
- Change of basis: $(\frac{\partial}{\partial x'^\nu})_p = (\frac{\partial x^\mu}{\partial x'^\nu})|_{\phi(p)} (\frac{\partial}{\partial x^\mu})_p$, $X'^\nu(x) = \frac{\partial x'^\nu}{\partial x^\mu}(\phi^{-1}(x)) X^\mu(\phi^{-1}(x))$
- Dual space $T_p^* M$: space of covectors - linear maps $\eta$ from $T_p M$ to $\mathbb{R}$.
- dual basis $\{\epsilon^\mu\}_{\mu=1}^n$ satisfying $\epsilon^\mu(e_\nu)=\delta^\mu_\nu$; $\eta=\eta_\mu \epsilon^\mu$
- **NOTE** * $\eta(e_\nu) = \eta_\mu \epsilon^\mu(e_\nu) = \eta_\mu \delta^\mu_\nu = \eta_\nu$.
* $\eta(X) = \eta(X^\mu e_\mu) = X^\mu \eta(e_\mu) = X^\mu \eta_\mu$

**DEF** If $f: M \to \mathbb{R}$ is a smooth function, define $(df)_p \in T_p^* M$, the **differential** of $f$ at p by
$$ (df)_p(X) = X(f) \quad \text{for any } X \in T_p M. $$
$(df)_p$ is sometimes also called the **gradient** of $f$ at p.

* If $f$ is constant $X(f)=0 \Rightarrow (df)_p=0$.
* If $(O, \phi)$ is a coord chart with $p \in O$ and $\phi=(x^1, ..., x^n)$ then we can set $f=x^\mu$ to find $(dx^\mu)_p$:
$$ (dx^\mu)_p(\frac{\partial}{\partial x^\nu})_p = (\frac{\partial}{\partial x^\nu})_p(x^\mu) = \frac{\partial x^\mu}{\partial x^\nu} = \delta^\mu_\nu $$
- Hence $\{(dx^\mu)_p\}_{\mu=1}^n$ is the dual basis to $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$.
- In this basis we can compute
$$ [(df)_p]_\mu = (df)_p(\frac{\partial}{\partial x^\mu})_p = (\frac{\partial}{\partial x^\mu})_p(f) = \frac{\partial f}{\partial x^\mu}\big|_{\phi(p)} \quad (F=f\circ\phi^{-1}) $$

Justifying the language 'GRADIENT'.

**Exercise:** show that if $(O', \phi')$ is another chart with $p \in O'$, then
$$ (dx^\mu)_p = (\frac{\partial x^\mu}{\partial x'^\nu})|_{\phi'(p)} (dx'^\nu)_p \quad x(x') = \phi\circ(\phi')^{-1} $$
and hence if $\eta = \eta_\mu dx^\mu = \eta'_\nu dx'^\nu$ then $\eta_\mu, \eta'_\mu$ are components w.r.t. these bases.
$$ \eta'_\nu = (\frac{\partial x^\mu}{\partial x'^\nu})|_{\phi'(p)} \eta_\mu $$
> *Annotation:* tangent space: vectors tangent to manifold.
> cotangent space: less clear, vectors annihilated by cotangent vector.
> glue these objects together as the point varies: co/tangent bundle

---
### Page 11

==Start of Transcription for page 11==

#### The (co)tangent bundle

We can glue together the tangent spaces $T_p M$ as p varies to get a new 2n dimensional manifold $TM$, the **tangent bundle**.
> *Annotation:* TM is 2n dimensional if M is dim n because an element of TM is specified by a point in the manifold along with a vector in the corresponding tangent space $T_p M$.
> TM is the collection of all the tangent spaces for all points on the manifold.

$TM = \bigcup_{p \in M} \{p\} \times T_p M$.
The set of ordered pairs $(p, X)$ with $p \in M$, $X \in T_p M$.
If $\{(O_\alpha, \phi_\alpha)\}$ is an atlas on M, we obtain an atlas for TM by setting $\tilde{O}_\alpha = \bigcup_{p \in O_\alpha} \{p\} \times T_p M$ and
$$ \tilde{\phi}_\alpha((p,X)) = (\phi(p), X^\mu) \in U_\alpha \times \mathbb{R}^n = \tilde{U}_\alpha $$
where $X^\mu$ are the components of X w.r.t. the coord bases of $O_\alpha$.

**Exercise:** If $(O, \phi)$ and $(O', \phi')$ are two charts on M, show that on $\tilde{U} \cap \tilde{U}'$, if we write $\phi' \circ \phi^{-1}(x) = x'(x)$ then $\tilde{\phi}' \circ \tilde{\phi}^{-1}(x, X^\mu) = (x'(x), (\frac{\partial x'^\nu}{\partial x^\mu})X^\mu)$. Deduce TM is a manifold.

A similar construction permits us to define the **cotangent bundle**
$T^* M = \bigcup_{p \in M} \{p\} \times T_p^* M$.

**Exercise:** show that the map $\pi: TM \to M$ which takes $(p,X) \mapsto p$ is smooth.

> *Note:* I'm mostly forget the last 10 mins of what I was saying "bundle construction" is not going to play much of a role in the rest of the course

#### ABSTRACT INDEX NOTATION
(capturing the value of index notation without having to assume we've chosen a coord. chart or basis.)

We've used greek letters $\mu, \nu$, etc. to label components of vectors (or covectors) w.r.t. the basis $\{e_\mu\}_{\mu=1}^n$ (resp $\{\epsilon^\mu\}_{\mu=1}^n$).
Equations involving these quantities refer to the specific basis. e.g. if we write $X^\mu = \delta^\mu_1$ (no longer true if change to diff. basis). This says X only has one non-zero component in current basis. This won't be true in other bases.
We know some equations hold in **all bases**, e.g. $\eta(X) = X^\mu \eta_\mu$.
> *Annotation:* abstract index promotes this type of statement to latin indices.

To capture this, we can use **abstract index notation (AIN)**. We denote a vector by $X^a$ where the latin index a does not denote a component, rather it tells us $X^a$ is a vector.

---
### Page 12

==Start of Transcription for page 12==

Similarly we denote a covector by $\eta_a$.
> *Annotation:* downstairs
If an equation is true in all bases, we can replace greek indices by latin indices.
> *Annotation:* If an eqn in latin indices is true in all bases, we're allowed to write it by replacing greek ones.
i.e. $\eta(X) = X^\mu \eta_\mu = X^a \eta_a$
or $X(f) = X^\mu(df)_\mu = X^a (df)_a$

An equation in AIN can always be turned into an equation for components by picking a basis and changing $a \to \mu$, $b \to \nu$ etc.

#### TENSORS
> *Annotation:* some quantities not described by either a scalar or a vector, even in Newtonian physics e.g. moment of inertia - need higher rank object.

In Newtonian physics, we know some quantities are described by higher rank objects (e.g. inertia tensor of a body).

**DEF:** A tensor of type (r,s) is a multilinear map
$$ T: \underbrace{T_p^*(M) \times ... \times T_p^*(M)}_{r \text{ factors}} \times \underbrace{T_p M \times ... \times T_p M}_{s \text{ factors}} \to \mathbb{R} $$
**Multilinear** means linear in each argument.

**Examples**
1. A tensor of type (0,1) is a linear map $T_p M \to \mathbb{R}$ i.e. a **covector**.
2. A tensor of type (1,0) is a linear map $T_p^* M \to \mathbb{R}$ i.e. an element $(T_p M)^{**} \cong T_p M$ a **vector**.
3. We can define a (1,1) tensor, $\delta$, by $\delta(\omega, X) = \omega(X)$, $\omega \in T_p^* M$, $X \in T_p M$.
> *Annotation:* defining it to be the map where the covector eats the vector.

If $\{e_\mu\}$ is a basis for $T_p M$ and $\{\epsilon^\mu\}$ the dual basis, the components of an (r,s) tensor T are
$$ T^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s} := T(\epsilon^{\mu_1}, ..., \epsilon^{\mu_r}, e_{\nu_1}, ..., e_{\nu_s}) $$
In AIN we denote T by $T^{a_1 ... a_r}_{b_1 ... b_s}$. Tensors at p form a vector space over $\mathbb{R}$ of DIM $n^{r+s}$.

**EXAMPLES**
1. consider $\delta$ above.
$\delta^\mu_\nu = \delta(\epsilon^\mu, e_\nu) = \epsilon^\mu(e_\nu) = \delta^\mu_\nu$
> *Annotation:* Kronecker delta defines a (1,1) tensor
we can write $\delta$ as $\delta^a_b$ in AIN.
2. consider a (2,1) tensor T, let $\omega, \eta \in T_p^* M$, $X \in T_p M$.
$T(\omega, \eta, X) = T(\omega_\mu \epsilon^\mu, \eta_\nu \epsilon^\nu, X^\sigma e_\sigma)$
$= \omega_\mu \eta_\nu X^\sigma T(\epsilon^\mu, \epsilon^\nu, e_\sigma)$
$= \omega_\mu \eta_\nu X^\sigma T^{\mu\nu}_\sigma$
in AIN $T(\omega, \eta, X) = \omega_a \eta_b X^c T^{ab}_c$.
generalisation to higher ranks.
> *Annotation:* basically given r covectors & s vectors, a tensor type (r,s) will give us a real number.

---
### Page 13

==Start of Transcription for page 13==

### Lecture 5
<div style="text-align:right">21.10.24</div>

#### CHANGE OF BASES

We've seen how components of X or $\eta$ w.r.t. a coordinate basis ($X^\mu$, $\eta_\mu$ resp.) change under a change of coordinates.
We don't have to only consider coordinate bases. Suppose $\{e_\mu\}_{\mu=1}^n$ and $\{e'_\mu\}_{\mu=1}^n$ are two bases for $T_p M$ with dual bases $\{\epsilon^\mu\}_{\mu=1}^n$ and $\{\epsilon'^\mu\}_{\mu=1}^n$.
We can expand $e'_\mu = A^\nu_\mu e_\nu$ and $e_\mu = B^\nu_\mu e'_\nu$ for some $A^\nu_\mu, B^\nu_\mu \in \mathbb{R}$.
But $\delta^\nu_\mu = \epsilon^\nu(e_\mu) = \epsilon^\nu(B^\sigma_\mu e'_\sigma) = B^\sigma_\mu \epsilon^\nu(e'_\sigma) = B^\sigma_\mu \delta^\nu_\sigma (A^\tau_\sigma \epsilon^\tau(e_\tau)) = A^\tau_\sigma B^\sigma_\mu (\delta^\nu_\tau) = A^\nu_\sigma B^\sigma_\mu$.
$$ e_\mu = B^\nu_\mu e'_\nu = B^\nu_\mu (A^\sigma_\nu e_\sigma) $$
$$ \delta^\tau_\mu = \epsilon^\tau(e_\mu) = \epsilon^\tau(A^\rho_\mu e_\rho) = A^\rho_\mu \delta^\tau_\rho = A^\tau_\mu $$
$$ e'_\mu = A^\nu_\mu e_\nu $$
$$ \delta^\tau_\sigma = \epsilon^\tau(e_\sigma) = A^\nu_\sigma \epsilon^\tau(e'_\nu) \quad \text{& linear map} $$
$$ \delta^\sigma_\mu = \epsilon'^\sigma(e'_\mu) = \epsilon'^\sigma(A^\nu_\mu e_\nu) = A^\nu_\mu \epsilon'^\sigma(e_\nu) = A^\nu_\mu B^\sigma_\nu $$
Thus $B^\sigma_\mu = (A^{-1})^\sigma_\mu$.

If $e_\mu = (\frac{\partial}{\partial x^\mu})_p$ and $e'_\mu = (\frac{\partial}{\partial x'^\mu})_p$ we've already seen $A^\nu_\mu = \frac{\partial x^\nu}{\partial x'^\mu}\big|_{\phi(p)}$, $B^\nu_\mu = \frac{\partial x'^\nu}{\partial x^\mu}\big|_{\phi(p)}$ which indeed satisfy $A^\nu_\sigma B^\sigma_\mu = \delta^\nu_\mu$ by the chain rule.

A change of bases induces a transformation of tensor components.
E.g. If T is a (1,1)-tensor
$T'^\mu_\nu = T(\epsilon'^\mu, e'_\nu) = T(B^\sigma_\mu \epsilon^\sigma, A^\tau_\nu e_\tau) = B^\sigma_\mu A^\tau_\nu T(\epsilon^\sigma, e_\tau) = B^\sigma_\mu A^\tau_\nu T^\sigma_\tau$
$T'^\mu_\nu = (A^{-1})^\mu_\sigma A^\tau_\nu T^\sigma_\tau$.

#### TENSOR OPERATIONS

Given an (r,s)-tensor, we can form an (r-1,s-1)-tensor by **contraction**.
For simplicity, assume T is a (2,2)-tensor. Define a (1,1)-tensor S by
$$ S(\omega, X) = T(\omega, \epsilon^\mu, X, e_\mu) \quad (*) $$
To see this is **independent** of the choice of basis:
$T(\omega, \epsilon'^\mu, X, e'_\mu) = T(\omega, (A^{-1})^\mu_\sigma \epsilon^\sigma, X, A^\tau_\mu e_\tau)$
$= (A^{-1})^\mu_\sigma A^\tau_\mu T(\omega, \epsilon^\sigma, X, e_\tau)$
$= \delta^\tau_\sigma T(\omega, \epsilon^\sigma, X, e_\tau) = T(\omega, \epsilon^\sigma, X, e_\sigma) = S(\omega, X)$

---
### Page 14

==Start of Transcription for page 14==

So (*) does not depend on the choice of basis. S and T have components related by $S^\mu_\nu = T^{\mu\sigma}_{\nu\sigma}$.
In any basis in AIN we write $S^a_b = T^{ac}_{bc}$.

Generalise to contract any upstairs index with any downstairs index in a general (r,s)-tensor.
Another way to make new tensors from old is to form the **tensor product**.

If S is a (p,q) tensor and T is an (r,s) tensor then $S \otimes T$ is a (p+r, q+s) tensor:
$$ (S \otimes T)(\omega^1, ..., \omega^p, \eta^1, ..., \eta^r, X_1, ..., X_q, Y_1, ..., Y_s) = S(\omega^1, ..., \omega^p, X_1, ..., X_q) T(\eta^1, ..., \eta^r, Y_1, ..., Y_s) $$
In AIN $(S \otimes T)^{a_1...a_p, b_1...b_r}_{c_1...c_q, d_1...d_s} = S^{a_1...a_p}_{c_1...c_q} T^{b_1...b_r}_{d_1...d_s}$.
**Exercise:** for any (1,1) tensor T, in a basis we have $T = T^\mu_\nu e_\mu \otimes \epsilon^\nu$.

The final tensor operations we require are **(anti-)symmetrisation**.
If T is a (0,2)-tensor, we can define two new tensors.
$S(X,Y) := \frac{1}{2}(T(X,Y)+T(Y,X))$
$A(X,Y) := \frac{1}{2}(T(X,Y)-T(Y,X))$
In AIN we write
$S_{ab} = \frac{1}{2}(T_{ab}+T_{ba}) = T_{(ab)}$
$A_{ab} = \frac{1}{2}(T_{ab}-T_{ba}) = T_{[ab]}$

These operations can be applied to any pair of matching indices in a more general tensor.
e.g. $T^{(abc)}_{de} := \frac{1}{2}(T^{abc}_{de} + T^{acb}_{de})$, etc.

We can (anti-)symmetrise over more than two indices.
* To **symmetrise** over n indices, sum over all permutations of the indices and divide by n!
* To **anti-symmetrise** over n indices sum over all permutations weighted by sign (even=+) and divide by n!.

---
### Page 15

==Start of Transcription for page 15==

e.g. $T_{(abc)} := \frac{1}{3!}(T_{abc} + T_{bca} + T_{cab} + T_{acb} + T_{cba} + T_{bac})$
$T_{[abc]} := \frac{1}{3!}(T_{abc} + T_{bca} + T_{cab} - T_{acb} - T_{cba} - T_{bac})$

To exclude indices from (anti-)symmetrisation, use vertical lines.
e.g. $T_{a|b|c} = \frac{1}{2}(T_{abc}+T_{cba})$

#### TENSOR BUNDLES (not super relevant)

The space of (r,s)-tensors at a point p is the vector space $(T^r_s)_p M$. These can be glued together to form the bundle of (r,s)-tensors.
$$ T^r_s M = \bigcup_{p \in M} \{p\} \times (T^r_s)_p M $$
If $(O, \phi)$ is a coordinate chart on M, set
$\tilde{O} = \bigcup_{p \in O} \{p\} \times (T^r_s)_p M \subset T^r_s M$
$\tilde{\phi}(p, S_p) = (\phi(p), S^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s})$ components of S w.r.t. coordinate basis

$T^r_s M$ is a manifold, with a natural smooth map $\pi: T^r_s M \to M$ such that $\pi(p, S_p) = p$.
An (r,s)-**TENSOR FIELD** is a smooth map $T: M \to T^r_s M$ such that $\pi \circ T = id$.
If $(O, \phi)$ is a coordinate chart on M then $\tilde{\phi} \circ T \circ \phi^{-1}(x) = (x, T^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s}(x))$ which is smooth provided the components $T^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s}(x)$ are smooth functions of x.

**SPECIAL CASE**
If $T^1_0 M = TM$
The tensor field is called a **vector field**. In a local coord. patch, if X is a vector field, we can write
$$ X(p) = (p, X_p) \text{ with } X_p = X^\mu(x)(\frac{\partial}{\partial x^\mu})_p $$
In particular $(\frac{\partial}{\partial x^\mu})$ are always **smooth** (but only **defined locally**).

---
### Page 16

==Start of Transcription for page 16==

### Lecture 6
<div style="text-align:right">23.10.24</div>

**RECAP**
If $T^1_0 = (1,0)$ we get a vector field $X(p)=(p,X_p)$. $X=X^\mu(x)(\frac{\partial}{\partial x^\mu})$.

A **vector field** can act on a function $f: M \to \mathbb{R}$ by to give a new function $Xf$ by
$$ (Xf)(p) = X_p(f) $$
In coordinates
$$ (Xf)(p) = X^\mu(\phi(p)) \frac{\partial f}{\partial x^\mu}\big|_{\phi(p)} $$

#### INTEGRAL CURVES

Given a vector field X on M, we say a curve $\lambda: I \to M$ is an **integral curve** of X if its tangent at every point is X. i.e. denote the tangent vector to $\lambda$ at t by $(\frac{d}{dt})_t$.
Then, $(\frac{d}{dt})_t\lambda(t) = X_{\lambda(t)} \quad \forall t \in I$.
Through each point p, an integral curve passes, unique up to extension/shift of the parameter.

To see this, pick a chart $\phi$ with $\phi=(x^1, ..., x^n)$ and $\phi(p)=0$. In this chart (+) becomes
$$ (*) \quad \frac{dx^\mu}{dt}(t) = X^\mu(x(t)) \quad x(t) = \phi(\lambda(t)) $$
Assuming wlog that $\lambda(0)=p$, we see that get an initial condition $(**) x^\mu(0)=0$.
Standard ODE theory gives that (*) with (**) has a solution unique up to extension.

#### COMMUTATORS

Suppose X and Y are two vector fields and $f: M \to \mathbb{R}$ is smooth. Then $X(Y(f))$ is a smooth function. Is it of the form $K(f)$ for some vector field K? **No**, because
$X(Y(fg)) = X(fY(g) + gY(f)) = X(f)Y(g) + X(g)Y(f)$
$= fX(Y(g)) + gX(Y(f)) + (Xf)Y(g) + (Xg)Y(f)$
Leibniz requires $X(Y(fg)) = fX(Y(g)) + (X(Y(f)))g$.
So Leibniz doesn't hold. But if we consider
$$ [X,Y](f) := X(Y(f)) - Y(X(f)) $$
then Leibniz does hold.
In fact $[X,Y]$ defines a **vector field**.
> *Annotation:* the commutator or the Lie bracket

---
### Page 17

==Start of Transcription for page 17==

To see this use coordinates
$$ [X,Y](f) = X(Y^\nu \frac{\partial f}{\partial x^\nu}) - Y(X^\mu \frac{\partial f}{\partial x^\mu}) $$
$$ = X^\mu \frac{\partial}{\partial x^\mu}(Y^\nu \frac{\partial f}{\partial x^\nu}) - Y^\nu \frac{\partial}{\partial x^\nu}(X^\mu \frac{\partial f}{\partial x^\mu}) $$
$$ = X^\mu \frac{\partial Y^\nu}{\partial x^\mu} \frac{\partial f}{\partial x^\nu} + X^\mu Y^\nu \frac{\partial^2 f}{\partial x^\mu \partial x^\nu} - Y^\nu \frac{\partial X^\mu}{\partial x^\nu} \frac{\partial f}{\partial x^\mu} - Y^\nu X^\mu \frac{\partial^2 f}{\partial x^\nu \partial x^\mu} $$
$$ = (X^\mu \frac{\partial Y^\nu}{\partial x^\mu} - Y^\mu \frac{\partial X^\nu}{\partial x^\mu}) \frac{\partial f}{\partial x^\nu} = [X,Y]^\nu \frac{\partial f}{\partial x^\nu} $$
Where $[X,Y]^\nu = X^\mu \frac{\partial Y^\nu}{\partial x^\mu} - Y^\mu \frac{\partial X^\nu}{\partial x^\mu}$ are the components of the commutator.
Since $f$ arbitrary, $[X,Y] = [X,Y]^\nu \frac{\partial}{\partial x^\nu}$, valid only in a coordinate basis.

#### METRIC TENSOR
> *Annotation:* talked about manifolds so far but not geometry stuff e.g. normals & length. so we'll do this now.

We're familiar from Euclidean geometry (and special relativity) with the fact that the fundamental object when talking about distance and angles (time intervals/rapidity) is an **inner product** between vectors.
E.g.
* $\vec{x} \cdot \vec{y} = x_1 y_1 + x_2 y_2 + x_3 y_3$ $\mathbb{R}^3$ w/ Euclidean geometry
* $X \cdot Y = -x^0 y^0 + x^1 y^1 + x^2 y^2 + x^3 y^3$ $\mathbb{R}^{3+1}$ w/ Minkowski geometry (c=1)

**DEF:** A **metric tensor** at $p \in M$ is a (0,2)-tensor $g$ satisfying
i) g is **symmetric**: $g(X,Y) = g(Y,X) \quad \forall X,Y \in T_p M \quad (g_{ab}=g_{ba})$
ii) g is **non-degenerate**: $g(X,Y)=0$ for all $Y \in T_p M$ iff $X=0$.

**NOTATION:** sometimes write $g(X,Y) = \langle X, Y \rangle = \langle X, Y \rangle_g = X \cdot Y$.

By adapting the Gram-Schmidt algorithm we can always find a basis $\{e_\mu\}_{\mu=1}^n$ for $T_p M$ such that
$g(e_\mu, e_\nu) = \begin{cases} 0 & \mu \neq \nu \\ +1 \text{ or } -1 & \mu = \nu \end{cases}$ (orthonormal basis).
i.e. $g_{\mu\nu} = \begin{pmatrix} -1 & & & \\ & 1 & & \\ & & \ddots & \\ & & & 1 \end{pmatrix}$
The number of -1's and +1's appearing does not depend on choice of basis (Sylvestre's law of inertia) and is called the **signature**.

---
### Page 18

==Start of Transcription for page 18==

- If g has signature $++...+$, we say it is **RIEMANNIAN**.
- If g has signature $-++...+$, we say it is **LORENTZIAN**.

**DEF:** A **Riemannian** (resp. **Lorentzian**) manifold is a pair $(M, g)$ where M is a manifold and g is a Riemannian (resp. Lorentzian) **metric tensor field**.
> *Remarks:* On a Riemannian manifold, we use the metric to define lengths and angles.

* On a Riemannian manifold the **norm** of a vector $X \in T_p M$ is $|X| = \sqrt{g(X,X)}$.
* The **angle** between $X, Y \in T_p M$ is given by $\cos\theta = \frac{g(X,Y)}{|X||Y|}$.
* The **length** of a curve $\lambda: (a,b) \to M$ is given by $l(\lambda) = \int_a^b |\frac{d\lambda}{dt}(t)| dt$.

**EXERCISE!** If $\tau: (c,d) \to (a,b)$ with $\frac{d\tau}{du} > 0$, $\tau(c)=a$, $\tau(d)=b$, then $\tilde{\lambda} = \lambda \circ \tau: (c,d) \to M$ is a re-representation of $\lambda$. Show $l(\tilde{\lambda}) = l(\lambda)$.
> *Annotation:* reparametrise curve does get the same ans?

*Drawing of a heart and mountains with a skier saying "I'd rather be skiing"*

---
### Page 19

==Start of Transcription for page 19==

### Lecture 7
<div style="text-align:right">25.10.24</div>

In a coordinate basis, $g = g_{\mu\nu} dx^\mu \otimes dx^\nu$. We often write $dx^\mu dx^\nu := \frac{1}{2}(dx^\mu \otimes dx^\nu + dx^\nu \otimes dx^\mu)$ and by convention write $g = ds^2$ so that $g = ds^2 = g_{\mu\nu} dx^\mu dx^\nu$.

**Examples:**
i) $\mathbb{R}^n$ with $g=ds^2 = (dx^1)^2 + (dx^2)^2 + ... + (dx^n)^2 = \delta_{\mu\nu}dx^\mu dx^\nu$.
is called **Euclidean space**. Any chart covering $\mathbb{R}^n$ in which the metric takes this form is called **CARTESIAN**.
> *Annotation:* flat (Riemannian example)

ii) $\mathbb{R}^{1+3} = \{(x^0, x^1, x^2, x^3)\}$ with $g = ds^2 = -(dx^0)^2 + (dx^1)^2 + (dx^2)^2 + (dx^3)^2 = \eta_{\mu\nu}dx^\mu dx^\nu$, $\eta_{\mu\nu} = \begin{pmatrix} -1 & & & \\ & 1 & & \\ & & 1 & \\ & & & 1 \end{pmatrix} = \begin{cases} -1 & \mu=\nu=0 \\ 1 & \mu=\nu\neq 0 \\ 0 & \text{otherwise} \end{cases}$.
Is **Minkowski space**. A coordinate chart covering $\mathbb{R}^{1+3}$ in which the metric takes this form is an **inertial frame**.
> *Annotation:* flat (Lorentz example)

iii) On $S^2 = \{x \in \mathbb{R}^3 \mid \|x\|=1\}$ define a chart by
$\Phi: (0, \pi) \times (-\pi, \pi) \to S^2$
$(\theta, \psi) \mapsto (\sin\theta\cos\psi, \sin\theta\sin\psi, \cos\theta)$
In this chart the round metric is $g = ds^2 = d\theta^2 + \sin^2\theta d\psi^2$.
This covers $S^2 \setminus \{\|x\|=1, x^2=0, x^1 \le 0\}$
To cover the rest let $\Phi': (0, \pi) \times (-\pi, \pi) \to S^2$
$(\theta', \psi') \mapsto (-\sin\theta'\cos\psi', \cos\theta', \sin\theta'\sin\psi')$
which covers $S^2 \setminus \{\|x\|=1, x^3=0, x^1 > 0\}$.
setting $g = d\theta'^2 + \sin^2\theta' d\psi'^2$ defines a metric on all of $S^2$. (check this by finding coordinate transform from unprimed to primed then on region of overlap the metric defined in each set of coords is the same tensor).
> *Annotation:* "painful to do this"

*Diagrams of a sphere showing the coordinate systems.*

---
### Page 20

==Start of Transcription for page 20==

Since $g_{ab}$ is non-degenerate, it is invertible as a matrix in any basis. We can check the inverse defines a symmetric (2,0)-tensor $g^{ab}$ satisfying $g^{ab} g_{bc} = \delta^a_c$.

**Example:** In the $\theta, \psi$ coordinates of the $S^2$ example
$g_{\mu\nu} = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}$
$g^{\mu\nu} = \begin{pmatrix} 1 & 0 \\ 0 & 1/\sin^2\theta \end{pmatrix}$

An important property of the metric is that it induces a canonical identification of $T_p M$ and $T_p^* M$.
* (vector) Given $X^a \in T_p M$ we define a covector $X_b = g_{ab}X^a$.
* (covector) given $\eta_a \in T_p^* M$ we define a vector $\eta^b = g^{ab}\eta_a$.

In $(\mathbb{R}^3, \delta)$ Euclidean space we often do this without realising (metric & its inverse are the identity in these coordinates: don't really distinguish between covectors & vectors).

More generally, this allows us to **raise** tensor indices with $g^{ab}$ and **lower** with $g_{ab}$.

**Example:** If $T^{ab}$ is a (2,1)-tensor then $T_a{}^b{}_c$ is the $(1,2)$-tensor given by $T_a{}^b{}_c = g_{ad} g^{ce} T^{d}{}_{e}{}^{b}$. etc.

#### LORENTZIAN SIGNATURE
> *Annotation:* irrelevant b/c this is what we assume spacetime to have.

In Lorentzian signature indices $0, 1, ..., n$, at any point P in a Lorentzian manifold we can find a basis $\{e_\mu\}_{\mu=0}^n$ st. $g(e_\mu, e_\nu) = \eta_{\mu\nu} = \text{diag}(-1, 1, ..., 1)$.
This basis is not unique, if $e'_\mu = (\Lambda^{-1})^\nu_\mu e_\nu$ is another such basis then
$\eta_{\lambda\rho} = g(e'_\lambda, e'_\rho) = (\Lambda^{-1})^\mu_\lambda (\Lambda^{-1})^\nu_\rho g(e_\mu, e_\nu) = (\Lambda^{-1})^\mu_\lambda (\Lambda^{-1})^\nu_\rho \eta_{\mu\nu}$
$\Rightarrow \Lambda^\sigma_\lambda \Lambda^\tau_\rho \eta_{\sigma\tau} = \eta_{\lambda\rho}$
which is the condition that $\Lambda^\mu_\nu$ is a **LORENTZ TRANSFORMATION** (cf. special relativity).
The tangent space at p has $\eta_{\mu\nu}$ as metric tensor (in this basis).

---
### Page 21

==Start of Transcription for page 21==

so has the structure of Minkowski space in particular.

**DEF:** on a lorentzian manifold $(M,g)$
$X \in T_p M$ is
**SPACELIKE** if $g(X,X) > 0$
**NULL/LIGHTLIKE** if $g(X,X) = 0$
**TIMELIKE** if $g(X,X) < 0$

*Diagram of a light cone, showing timelike (inside), null (on the cone), and spacelike (outside) regions.*
> *Annotation:* every whilst a vector has to be one of these a curve can be now one of one & become another.

A curve $\lambda: I \to M$ in a Lorentzian manifold is **spacelike**, **timelike** or **null** if the tangent vector is everywhere spacelike, timelike or null resp.

A **spacelike** curve has a well-defined **LENGTH** given by the same formula as in Riemannian case.
For a **timelike** curve $\lambda: (a,b) \to M$, the relevant quantity is the **PROPER TIME**
$$ \tau(\lambda) = \int_a^b \sqrt{-g_{ab}\frac{d\lambda^a}{du}\frac{d\lambda^b}{du}} du $$
If $g_{ab}\frac{d\lambda^a}{d\tau}\frac{d\lambda^b}{d\tau}=-1$ for all $\tau$, then $\lambda$ is parametrised by **proper time**.
In this case we call the tangent vector $u^a := \frac{d\lambda^a}{d\tau}$ the **4-VELOCITY** of $\lambda$.

#### CURVES OF EXTREMAL PROPER TIME

Suppose $\lambda: (0,1) \to M$ is timelike, satisfies $\lambda(0)=p$, $\lambda(1)=q$ and extremises proper time among all such curves. This is a variational problem associated to (in a coordinate chart)
$$ \tau[\lambda] = \int_0^1 G(x^\mu(u), \dot{x}^\mu(u)) du \qquad (\dot{} = \frac{d}{du} \text{ here}) $$
with $G(x^\mu(u), \dot{x}^\mu(u)) = \sqrt{-g_{\mu\nu}(x(u))\dot{x}^\mu(u)\dot{x}^\nu(u)}$.
**EULER LAGRANGE EQN** is $\frac{d}{du}\frac{\partial G}{\partial \dot{x}^\mu} - \frac{\partial G}{\partial x^\mu} = 0$, we can compute:
$\frac{\partial G}{\partial \dot{x}^\mu} = -\frac{1}{2G} g_{\sigma\mu} \dot{x}^\sigma 2 = -\frac{1}{G}g_{\sigma\mu}\dot{x}^\sigma$
$\frac{\partial G}{\partial x^\mu} = -\frac{1}{2G}g_{\sigma\nu,\mu}\dot{x}^\sigma\dot{x}^\nu$.

---
### Page 22

==Start of Transcription for page 22==

### Lecture 8
<div style="text-align:right">28.10.24</div>

#### CURVES OF EXTREMAL PROPER TIME cont.
(we've had arbitrary parametrisation u)

* Now fix parametrisation so curve is parametrised by **proper time** $\tau$. Doing this $\frac{dx^\mu}{d\tau} = \dot{x}^\mu \frac{du}{d\tau}$ and $-1 = g_{\mu\nu}\frac{dx^\mu}{d\tau}\frac{dx^\nu}{d\tau}$.
chain rule
* Deduce $-1 = g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu (\frac{du}{d\tau})^2 \Rightarrow \frac{du}{d\tau} = \frac{1}{\sqrt{-g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu}} = \frac{1}{G}$. $\Rightarrow G \frac{du}{d\tau} = 1$.
* Returning to $(\dagger)$ we find
$$ \frac{d}{d\tau}(g_{\mu\nu}\frac{dx^\nu}{d\tau}) = \frac{1}{2}g_{\nu\sigma,\mu}\frac{dx^\nu}{d\tau}\frac{dx^\sigma}{d\tau} $$
$$ \Rightarrow g_{\mu\nu}\frac{d^2x^\nu}{d\tau^2} + g_{\mu\nu,\rho}\frac{dx^\rho}{d\tau}\frac{dx^\nu}{d\tau} = \frac{1}{2}g_{\nu\sigma,\mu}\frac{dx^\nu}{d\tau}\frac{dx^\sigma}{d\tau} $$
* Thus $\frac{d^2x^\mu}{d\tau^2} + \Gamma^\mu_{\nu\sigma}\frac{dx^\nu}{d\tau}\frac{dx^\sigma}{d\tau} = 0 \quad (*)$ the geodesic equation
* where $\Gamma^\mu_{\nu\sigma} := \frac{1}{2}g^{\mu\rho}(g_{\rho\nu,\sigma} + g_{\rho\sigma,\nu} - g_{\nu\sigma,\rho})$ are the **CHRISTOFFEL SYMBOLS** of g.

**NOTE:**
- $\Gamma^\mu_{\nu\sigma} = \Gamma^\mu_{\sigma\nu}$ (symmetric in downstairs indices)
- $\Gamma^\mu_{\nu\sigma}$ are not tensor components (cf. exercise 2.1)
- We can solve (*) with standard ODE theory, solutions are called **GEODESICS**.
- The same equation governs curves of extremal length in a Riemannian manifold (or spacelike curves in Lorentzian manifold) parametrised by arc length.

**EXERCISE!** Show that (*) can be obtained as the Euler-Lagrange equation for the Lagrangian
$$ L = -g_{\mu\nu}(x(\tau))\dot{x}^\mu(\tau)\dot{x}^\nu(\tau) $$
> *Annotation:* an easier way to derive the geodesic eqn or christoffel symbols.

---
### Page 23

==Start of Transcription for page 23==

**EXAMPLES:**
1) In Minkowski space in an inertial frame $g_{\mu\nu} = \eta_{\mu\nu}$ so $\Gamma^\rho_{\mu\nu}=0$ and geodesic equation is $\frac{d^2 x^\mu}{d\tau^2}=0$.
Geodesics are straight lines.

2) The Schwarzschild metric in Schwarzschild coordinates is on $M = \mathbb{R}_t \times (2m, \infty)_r \times S^2_{\theta,\phi}$.
$ds^2 = -f dt^2 + \frac{dr^2}{f} + r^2(d\theta^2 + \sin^2\theta d\phi^2)$, $f=1-\frac{2m}{r}$.
$L = f(\frac{dt}{d\tau})^2 - \frac{1}{f}(\frac{dr}{d\tau})^2 - r^2(\frac{d\theta}{d\tau})^2 - r^2\sin^2\theta(\frac{d\phi}{d\tau})^2$.
E-L equation for $t(\tau)$ is $\frac{\partial L}{\partial t} = \frac{d}{d\tau}(\frac{\partial L}{\partial \dot{t}}) \Rightarrow \frac{\partial L}{\partial t} = 0$.
$\Rightarrow \frac{d}{d\tau}(\frac{\partial L}{\partial \dot{t}}) = 0 \Rightarrow 2\frac{d}{d\tau}(f\frac{dt}{d\tau}) = 0 \Rightarrow f\frac{dt}{d\tau} = E \text{ (constant)}$.
Compare to (*) to see $\Gamma^0_{01} = \frac{1}{2f}\frac{df}{dr}$. $\Gamma^\mu_{\nu\sigma}=0$ otherwise.
Rest of symbols can be found from other EL equations.

#### COVARIANT DERIVATIVE

For a function $f: M \to \mathbb{R}$, we know that $\frac{df}{dx^\mu}$ are components of a covector $(df)_\mu$.
For a vector field, we can't just differentiate components.

**EXERCISE:** Show that if V is a vector field then $\partial_\mu V^\nu$ are not components of a (1,1)-tensor.
> *Note:* components of vector transform as $V'^\lambda(x') = \frac{\partial x'^\lambda}{\partial x^\nu}V^\nu(x)$.
> So $\partial'_\sigma V'^\lambda = \frac{\partial}{\partial x'^\sigma}(\frac{\partial x'^\lambda}{\partial x^\nu}V^\nu) = \frac{\partial^2 x'^\lambda}{\partial x'^\sigma \partial x^\nu}V^\nu + \frac{\partial x'^\lambda}{\partial x^\nu} \frac{\partial V^\nu}{\partial x^\mu}\frac{\partial x^\mu}{\partial x'^\sigma}$.
> ... 'extra term' does not transform as tensor.

---
### Page 24

==Start of Transcription for page 24==

**DEF:** A **covariant derivative** $\nabla$ on a manifold M is a map sending X,Y smooth vector fields to a vector field $\nabla_X Y$ satisfying (X,Y,Z smooth V.fields, f,g functions)
i) $\nabla_{fX+gY}Z = f\nabla_X Z + g\nabla_Y Z$
ii) $\nabla_X(Y+Z) = \nabla_X Y + \nabla_X Z$
iii) $\nabla_X(fY) = f\nabla_X Y + (Xf)Y$ where $Xf := X(f)$.

**Note** i) implies that $\nabla Y: X \mapsto \nabla_X Y$ is a linear map of $T_p M$ to itself, so defines a (1,1)-tensor, the **COVARIANT DERIVATIVE** of Y $\equiv$ **AFFINE CONNECTION**.
In AIN $(\nabla_X Y)^a_b = \nabla_b Y^a$ or $Y^a_{;b}$.

**DEF:** In a basis $\{e_\mu\}$ the **CONNECTION COMPONENTS** $\Gamma^\lambda_{\mu\nu}$ are defined by
$$ \nabla_{e_\mu}e_\nu = \Gamma^\lambda_{\mu\nu}e_\lambda $$
These determine $\nabla$.
$\nabla_X Y = \nabla_{X^\mu e_\mu}(Y^\nu e_\nu) = X^\mu \nabla_{e_\mu}(Y^\nu e_\nu) = X^\mu(e_\mu(Y^\nu)e_\nu + Y^\nu \nabla_{e_\mu}e_\nu)$
$= (X^\mu e_\mu(Y^\nu) + \Gamma^\nu_{\mu\sigma} Y^\sigma X^\mu)e_\nu$
Hence $(\nabla_X Y)^\nu = X^\mu(e_\mu(Y^\nu) + \Gamma^\nu_{\mu\sigma}Y^\sigma)$.
In a coord basis $e_\mu = \frac{\partial}{\partial x^\mu}$ then $e_\mu(Y^\nu) = \frac{\partial Y^\nu}{\partial x^\mu}$.
$Y^\nu_{;\mu} := (\nabla_{e_\mu} Y)^\nu = \partial_\mu Y^\nu + \Gamma^\nu_{\mu\sigma}Y^\sigma$.

$\Gamma^\lambda_{\mu\nu}$ are not components of a tensor.
**E.g.** for $\eta$ a tensor field we define $(\nabla_X \eta)(Y) := X(\eta(Y)) - \eta(\nabla_X Y)$.
In components $(\nabla_X \eta)_Y = X^\mu e_\mu(\eta_\sigma Y^\sigma) - \eta_\sigma(\nabla_X Y)^\sigma$
$= X^\mu(e_\mu(\eta_\sigma)Y^\sigma + \eta_\sigma e_\mu(Y^\sigma)) - \eta_\sigma(X^\mu \partial_\mu Y^\sigma + X^\mu\Gamma^\sigma_{\mu\nu}Y^\nu)$
$= (e_\mu(\eta_\sigma) - \Gamma^\nu_{\sigma\mu}\eta_\nu)X^\mu Y^\sigma$. $\nabla \eta$ is a tensor (0,2)
$\nabla_\mu \eta_\sigma = e_\mu(\eta_\sigma) - \Gamma^\nu_{\sigma\mu}\eta_\nu =: \eta_{\sigma;\mu}$
in coord basis $\eta_{\sigma;\mu} = \partial_\mu \eta_\sigma - \Gamma^\nu_{\sigma\mu}\eta_\nu$.

---
### Page 25

==Start of Transcription for page 25==

$X^\mu_{;\nu} = \partial_\nu X^\mu + \Gamma^\mu_{\sigma\nu} X^\sigma$
$\eta_{\mu;\nu} = \partial_\nu \eta_\mu - \Gamma^\sigma_{\mu\nu} \eta_\sigma$

### Lecture 9
<div style="text-align:right">30.10.24</div>

**Exercise:** in a coordinate basis $T^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s; \rho} = \partial_\rho T^{\mu_1 ... \mu_r}_{\nu_1 ... \nu_s} + \sum_i \Gamma^{\mu_i}_{\sigma\rho}T^{\mu_1 .. \sigma .. \mu_r}_{\nu_1 ... \nu_s} - \sum_j \Gamma^{\sigma}_{\nu_j \rho} T^{\mu_1 ... \mu_r}_{\nu_1 .. \sigma .. \nu_s}$

**Remark** If $T^a_b$ is a (1,1) tensor, then $T^a_{b;c}$ is a (1,2) tensor and we can take further covariant derivatives.
$(T^a_{b;c})_{;d} =: T^a_{b;cd} = \nabla_d \nabla_c T^a_b$
In general $T^a_{b;cd} \neq T^a_{b;dc}$

If $f$ is a function $f_{;a} = (df)_a$ is a covector. In a coordinate basis $f_{;\mu} = \partial_\mu f$.
$f_{;[\mu\nu]} = \frac{1}{2}(\partial_\nu\partial_\mu f - \Gamma^\sigma_{\mu\nu}\partial_\sigma f - (\partial_\mu\partial_\nu f - \Gamma^\sigma_{\nu\mu}\partial_\sigma f)) = -\Gamma^\sigma_{[\mu\nu]}\partial_\sigma f$.

**DEF:** A connection (= covariant derivative) is **torsion free** or **symmetric** if $\nabla_X Y - \nabla_Y X = [X,Y]$.
For any function f in a coordinate basis this is equivalent to $\Gamma^\rho_{[\mu\nu]} = 0 \Leftrightarrow \Gamma^\rho_{\mu\nu} = \Gamma^\rho_{\nu\mu}$.

**LEMMA:** If $\nabla$ is torsion free, then for X,Y vector fields
$$ \nabla_X Y - \nabla_Y X = [X,Y] $$

**PROOF:** In a coordinate basis
$(\nabla_X Y - \nabla_Y X)^\mu = X^\sigma Y^\mu_{;\sigma} - Y^\sigma X^\mu_{;\sigma}$
$= X^\sigma(\partial_\sigma Y^\mu + \Gamma^\mu_{\sigma\nu}Y^\nu) - Y^\sigma(\partial_\sigma X^\mu + \Gamma^\mu_{\sigma\nu}X^\nu)$
$= [X,Y]^\mu + X^\sigma Y^\nu(\Gamma^\mu_{\sigma\nu} - \Gamma^\mu_{\nu\sigma}) = [X,Y]^\mu$
This is a tensor equation, so if true in one basis, true in all.

==End of Transcription==