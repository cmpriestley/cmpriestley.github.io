


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


so $X^\mu = \frac{\partial x^\mu}{\partial x'^\nu}(\phi(p)) X'^\nu$.

We do not have to choose a coordinate basis such as $\{(\frac{\partial}{\partial x^\mu})_p\}_{\mu=1}^n$. With respect to a general basis $\{e_\mu\}_{\mu=1}^n$ for $T_p M$ we write $X_p = X^\mu e_\mu$ for $X^\mu \in \mathbb{R}$ are components w.r.t. $\{e_\mu\}_{\mu=1}^n$.
We always use summation convention: we always contract one upstairs and one downstairs index. The index on $\frac{\partial}{\partial x^\mu}$ counts as downstairs.

### COVECTORS

Recall that if V is a vector space over $\mathbb{R}$, the **dual space** $V^*$ is the space of linear maps from V to $\mathbb{R}$. If V is n-dimensional, so is $V^*$. Given a basis $\{e_\mu\}_{\mu=1}^n$ for V, we define the dual basis $\{\epsilon^\mu\}_{\mu=1}^n$ for $V^*$ by requiring $\epsilon^\mu(e_\nu) = \delta^\mu_\nu = \begin{cases} 1 & \mu=\nu \\ 0 & \mu\neq\nu \end{cases}$.
If V is finite dimensional then $V^{**} = (V^*)^*$ is isomorphic to V: to an element X of V we associate the linear map $\Lambda_X: V^* \to \mathbb{R}$, $\Lambda_X(\omega) = \omega(X)$, $\omega \in V^*$.

**Def**: The dual space of $T_p M$ is denoted $T_p^* M$ and called the **cotangent space** to M at p. An element of $T_p^* M$ is a **covector** at p. If $\{e_\mu\}_{\mu=1}^n$ is a basis for $T_p M$ and $\{\epsilon^\mu\}_{\mu=1}^n$ the dual basis for $T_p^* M$, we can expand a covector $\eta$ as $\eta = \eta_\mu \epsilon^\mu$ for $\eta_\mu \in \mathbb{R}$, the **components** of $\eta$.

---


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


---


---


note: Even if $\nabla$ is torsion free, $\nabla_a \nabla_b X^c \neq \nabla_b \nabla_a X^c$ in general.

# THE LEVI-CIVITA CONNECTION

For a manifold with metric there is a preferred connection (T.M. Fundamental theorem of Riemannian geometry).

If $(M,g)$ is a manifold with a metric, there is a **unique** torsion free connection $\nabla$ satisfying $\nabla g=0$. This is called the **levi-civita connection**.
> *(in brief $\nabla_X g = 0$)*

**PROOF:** Suppose such a connection exists. By Leibniz rule, if $X,Y,Z$ are smooth vector fields
* $X(g(Y,Z)) = \nabla_X(g(Y,Z)) = (\nabla_X g)(Y,Z) + g(\nabla_X Y,Z) + g(Y, \nabla_X Z)$
$X(g(Y,Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$ a)
$Y(g(Z,X)) = g(\nabla_Y Z, X) + g(Z, \nabla_Y X)$ b)
$Z(g(X,Y)) = g(\nabla_Z X, Y) + g(X, \nabla_Z Y)$ c)

> *since metric symmetric*

* a) + b) - c):
$X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) = g(\nabla_X Y + \nabla_Y X, Z) + g(\nabla_X Z - \nabla_Z X, Y)$
$+ g(\nabla_Y Z - \nabla_Z Y, X)$

* Use $\nabla_X Y - \nabla_Y X = [X,Y]$
> *this step helps be no dependence on $\nabla$ on RHS. is that correct? an alternative to torsion free.*

$X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) = 2g(\nabla_X Y, Z) - g([X,Y],Z) - g([Z,X],Y) + g([Y,Z],X)$

*
> *deters uniqueness. Z is arbitrary & g is non degenerate*
=> $g(\nabla_X Y, Z) = \frac{1}{2}\{X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) + g([X,Y],Z) + g([Z,X],Y) - g([Y,Z],X)\}$ (+)

> *In general this means for any $V \in T_p M$ s.t. $g(V,W)=0 \forall W$, $V=0$. In general not a non-degenerate matrix.*
> *This RHS can be used to show that $\nabla_X Y$ is unique since everything on the RHS are already known quantities. Then w/ completeness, there is no ambiguity.*

This determines $\nabla_X Y$ uniquely since $g$ is **non-degenerate**.
Conversely we can use (+) to define $\nabla_X Y$. Then need to check properties of a symmetric connection hold.

* E.g. $g(\nabla_Y X, Z) = \frac{1}{2}\{Y(g(X,Z)) + X(g(Z,Y)) - Z(g(Y,X)) + g([Y,X],Z) + g([Z,Y],X) - g([X,Z],Y)\}$

> *show $\nabla_X Y - \nabla_Y X = [X,Y]$*
> *use similar arguments but not in general coords, and $[X,Y]$ is a vector.*

$= \frac{1}{2}\{X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) + Y(g(X,Z)) - Z(g(X,Y)) + g(g[X,Y]-Yg_SX, Z) + g(Y[Z,X]+Zg_X,Y) - g(g[Y,Z],X)\}$

=> $g(\nabla_X Y, Y) = g(\nabla_Y X, Z) \implies g(\nabla_X Y - \nabla_Y X, Z) = 0$. $\forall Z$

* so $\nabla_X Y = \nabla_Y X$ as $g$ non-degenerate.

> *"non-degenerate" means "for any $X \neq 0$ s.t. $g(X,Y)=0 \forall Y$ "*
> *I think apparently the others are easier.*

***
**Exercise:** check other properties
> *Use the expression (+) we found.*

---


In a coord. basis we can compute
$g(\nabla_{e_\mu} e_\nu, e_\sigma) = \frac{1}{2} \{e_\mu(g(e_\nu, e_\sigma)) + e_\nu(g(e_\sigma, e_\mu)) - e_\sigma(g(e_\mu, e_\nu))\}$
$g(\Gamma^\tau_{\mu\nu} e_\tau, e_\sigma) = \Gamma^\tau_{\mu\nu} g_{\tau\sigma} = \frac{1}{2} (g_{\nu\sigma,\mu} + g_{\sigma\mu,\nu} - g_{\mu\nu,\sigma})$
=> $\Gamma^\tau_{\mu\nu} = \frac{1}{2}g^{\tau\sigma} (g_{\sigma\nu,\mu} + g_{\sigma\mu,\nu} - g_{\mu\nu,\sigma})$
> *now on lhs take inner product w/ $g^{\tau\sigma}$*
> *lhs becomes $\Gamma^\tau_{\mu\nu}$ since the difference between two connections is a tensor field, so can define the components of the connection as the levi-civita connection components (only for LC?)*
> *minding my $g$'s.*

↑ called **CHRISTOFFEL SYMBOLS**
> *important: learn!*
> *notes, we have shown the Christoffel symbols are the components of the Levi-Civita connection in a coordinate basis.*

If $\nabla$ is levi-civita can raise/lower indices and this **commutes** with covariant differentiation.
If $\nabla$ is levi-civita then $g_{ab}\nabla_c X^b = \nabla_c (g_{ab} X^a) = \nabla_c X_b)$
> *now think a bit about curves that satisfy this eqn. (where is $g$ from?)*

### GEODESICS

We found that a curve extremizing proper time satisfies
> *integral curve? the geodesic equation?*
> *is this because we derived this using proper time as the parameter, so proper time is the only parameterisation so that this eqn holds?*

$$(\dagger) \quad \frac{d^2x^\mu}{dt^2} + \Gamma^\mu_{\nu\sigma}(x(\tau)) \frac{dx^\nu}{dt}\frac{dx^\sigma}{dt} = 0$$
> $\tau$ proper time along curve.

The tangent vector $X^\mu$ to the curve has components $X^\mu = \frac{dx^\mu}{dt}$. Extending this off the curve we get a vector field, of which the geodesic is an integral curve. We note
> *I don't think this works is it a vector field?*

$$\frac{d^2x^\mu}{dt^2} = \frac{d}{dt} \left( \frac{dx^\mu}{dt} \right) = \frac{\partial X^\mu}{\partial x^\nu} \frac{dx^\nu}{dt} = X^\nu \partial_\nu X^\mu = X^\nu \nabla_\nu X^\mu \quad (\text{chain rule})$$
$(\dagger)$ becomes $X^\nu\nabla_\nu X^\mu + \Gamma^\mu_{\nu\sigma}X^\nu X^\sigma = 0 \implies X^\nu\nabla_\nu X^\mu = 0 \implies \nabla_X X = 0$.
> *where we are using the Levi Civita connection. now extend.*
> *no, in fact isn't this what we are going to define to be an affine parameter for any connection?*

Extend to any connection
**DEF:** Let M be a manifold with connection $\nabla$. An **AFFINELY PARAMETRIZED GEODESIC** satisfies
> *notes, an integral curve of a vector grid X satisfying...*

$$\nabla_X X = 0$$
where $X$ is the tangent vector.
> *tangent vect. to curve defined only on curve itself.*

**(Lecture 10)** note: if we reparametrise $t \to t(u)$ then
$$\frac{dx^\mu}{du} = \frac{dx^\mu}{dt} \frac{dt}{du}$$
so $X \to Y = hX$ with $h > 0$.
$\nabla_Y Y = \nabla_{hX}(hX) = h\nabla_X(hX) = h^2 \nabla_X X + hX(h) = gX$.
with $g = X(h) = \frac{d}{dt}(h) = \frac{dh}{du}\frac{du}{dt} = \frac{1}{h}\frac{dh}{du^2}$. so $\nabla_Y Y = 0 \iff \ddot{t} = a\dot{t} + \beta, \alpha, \beta \in \mathbb{R}$
> *conclude:*
> *... the form of the affine parameters.*
> *From any two parameters, we can link them with affine parameterisation for any geodesic.*
> *$t = au + \beta$*
> *what is meant by affine reparametrisation: "$\nabla_Y Y$ describes the same curve but in general not affinely parametrised but it is also possible to find a parameter s.t. it is. in this case $\nabla_Y Y=0$. $u=at+b$ wlog precision calways restrict to APGs*
> *affine reparametrisation, line composites $\alpha > 0$*
> *$X(h) = \frac{d}{dt}(\frac{dt}{du}) = \frac{d^2 t}{du^2}$*
> *this must be 0 for $\nabla_Y Y = 0$*
> *$\frac{d^2 t}{du^2} = 0 \implies \frac{dt}{du} = \text{const}$*
> *$t = \alpha u + \beta$*

---


*Lecture 10*

> *Exercise: let X be tangent to an APG of the Levi-Civita connection. show that $\nabla_X(g(X,X))=0$.*
> *attempt: $\nabla_X(g(X,X)) = X(g(X,X)) = (\nabla_X g)(X,X) + g(\nabla_X X, X) + g(X, \nabla_X X) = 2g(0,X)=0$.*
> *so this implies the tangent vector does not change its magnitude (e.g. 1.11.24 for timelike or null) along the geodesic, a geodesic must remain timelike, spacelike or null.*

**Theorem:** given $p \in M$, $X_p \in T_p M$, there exists a unique **A.P.G.** $\lambda: I \to M$ satisfying $\lambda(0)=p \quad \dot{\lambda}(0) = X_p$. $\lambda(t)$

**PROOF:** Choose coordinates with $\phi(p)=0$. $\lambda^\mu(t) = \phi(\lambda(t))$.
Satisfies $\nabla_X X=0$ with $X = X^\mu \frac{\partial}{\partial x^\mu}$. This becomes
> *$X^\mu = \frac{d \lambda^\mu}{dt}$*

$$ \frac{d^2 x^\mu}{dt^2} + \Gamma^\mu_{\nu\sigma} \frac{dx^\nu}{dt} \frac{dx^\sigma}{dt} = 0 \quad (GE) $$

and $\lambda^\mu(0)=0 \quad \frac{dx^\mu}{dt}(0) = X_p^\mu$.
> *2nd order ODE with two BCs $\implies$ uniqueness follows from standard ODE theory.*

This has a **unique** solution $x^\mu: (-\epsilon, \epsilon) \to \mathbb{R}^n$ for $\epsilon$ sufficiently small by standard ODE theory.

### GEODESIC POSTULATE

> *not acted on by any force except gravity.*

In general relativity free particles move along **geodesics** of the Levi-Civita connection.
These are **TIMELIKE** for massive particles and **NULL/LIGHTLIKE** for massless particles.

### Normal Coordinates
> *"exponential map"*
> *local "rectifying" points on manifold, by specifying a direction and a distance to get there. breaks down long distances. "how to walk in a straight line, but to specify there are many paths"*

If we fix $p \in M$ we can map $T_p M$ into $M$ by setting $\psi(X_p) = \lambda_{X_p}(1)$ where $\lambda_{X_p}$ is the unique affinely parametrised geodesic with $\lambda_{X_p}(0)=p, \dot{\lambda}_{X_p}(0)=X_p$.
> *uniqueness problem*
> *define a map through p that preserves the tangent vector at that point, unique to the point and tangent vector*

Notice that
> *rescale vector equivalence argument*
$\lambda_{\alpha X_p}(t) = \lambda_{X_p}(\alpha t)$ for $\alpha \in \mathbb{R}$.
since if $\tilde{\lambda}(t) = \lambda_{X_p}(\alpha t)$ affine reparametrisation so still geodesic, and $\tilde{\lambda}(0) = \lambda_{X_p}(0) = \alpha X_p$, $\dot{\tilde{\lambda}}(0)=p$.
> *since if $\lambda$ is affine param with $u=at+b$, $t$ is also affine param for $a,b \in \mathbb{R}$ I think.*
> *i.e. the map that sends $X_p$ to 1 also sends $\alpha X_p$ to $\alpha$.*

Moreover, $\alpha \mapsto \psi(\alpha X_p)$ is an affinely parametrised geodesic $= \lambda_{X_p}(\alpha)$.
> *define map from tangent space to M. claim as origin is "sufficiently small", neighborhood of ig origin then bijective map.*
> *noteworthy: like throwing a ball and seeing where it lands. if i throw it w/ initial velocity $X_p$ guess will take 1 time unit to land. if $2X_p$ gives twice the distance.*
> *The exponential map sends $X_p$ to the point unit distance along the geodesic through p tangent to $X_p$ at p. But it sends $tX_p$ distance $t$ along that same geodesic.*

---


**CLAIM:** If $U \subset T_p M$ is a sufficiently small neighbourhood of the origin, then $\psi|_U : T_p M \to M$ is one-to-one and onto.
> *(real analysis part where onto)*
> *can prove this easily, check jacobian, is invertible (not part of this course)*

**DEF:** Construct **normal coordinates** at $p$. Suppose $\{e_\mu\}$ is a basis for $T_p M$, as follows. For $q \in \psi(U) \subset M$, we define $\phi(q) = (x^1, ..., x^n)$ where $x^\mu$ are components of the unique $X_p \in U$ with $\psi(X_p)=q$. (write $q=x^\mu e_\mu$)
> *geodesic $\lambda(t)$ has coords $tx^\mu$. the straight lines through origin are geodesics.*
> *see "where it lands". he relates land velocty needed to reach that point in say 1 second. $q=X_p(1) \implies$ assigning coordinates $x^\mu$ for $X_p=x^\mu e_\mu$.*
> *if normal coord $x^\mu(t)=x^\mu$ then $x^\mu(t) = tx^\mu$?*
> *I don't just want to assign vector coords, I want to say $\phi(X_p(t)) = t x^\mu$.*
> *so an unique pt in U corresponds to a unique tangent vector at that point which will get me to q in one appropriately scaled unit of time.*

By our previous observation, the curve given in normal coordinates by $x^\mu(t)=ty^\mu$ for $y^\mu$ constant is an affinely parametrised geodesic so from (GE)
> *we know this is a geodesic so can just plug it in?*
$\Gamma^\mu_{\nu\sigma}(ty)y^\nu y^\sigma = 0$
> *the geodesic eqn. say $\psi(X_p) = \lambda_{X_p}(1)$, then $\psi(tX_p) = \lambda_{X_p}(t)$ so normal coords are my speed if i throw at speed $X_p$. then to the point that i reach in 1s. but if at $tX_p$ then it's the point i reach in ts, ie my coords are assigned.*

Set $t=0$ deduce (since $y^\mu$ arbitrary) that $\Gamma^\mu_{\nu\sigma}|_p = 0$.
> *reminder: antisymmetric part is for torsion so if we contract with symmetric $y^\nu y^\sigma$ then since $y^\mu$ is arbitrary, the symmetric part of $\Gamma$ must vanish. p vanishes*
> *components vanish at p but not in general.*

So if $\nabla$ is torsion free, $\Gamma^\mu_{[\nu\sigma]}|_p = 0$ in normal coordinates.
> *(with this means need to be careful with symmetric contraction vanishing)*
> *everywhere*

If $\nabla$ is the Levi-Civita connection of a metric, then otherwise this would be a stronger result
use $g_{\mu\nu,\rho} = \frac{1}{2}(g_{\mu\rho,\nu} + g_{\rho\nu,\mu} - g_{\mu\nu,\rho})$.
(NB normal coords are coords where the spacetime looks like minkowski space (at a point) to 1st order)
Since $g_{\mu\nu,\rho} = \frac{1}{2}(g_{\mu\nu,\rho} + g_{\rho\nu,\mu} - g_{\mu\rho,\nu}) + \frac{1}{2}(g_{\mu\rho,\nu} + g_{\mu\rho,\nu} - g_{\rho\nu,\mu})$
$= \Gamma^\sigma_{\mu\rho}g_{\sigma\nu} + \Gamma^\sigma_{\nu\rho}g_{\sigma\mu} = 0$ at p.
> *for any nice $p$ of a metric, if we choose normal coords at p then the first partial derivs of the metric vanishes.*
> *if $\Gamma_{\mu\nu\sigma}$ vanishes then the deriv of the metric vanishes.*

We can always choose the basis $\{e_\mu\}$ for $T_p M$ on which base the normal coordinates to be orthonormal. We have
**(Euclidean??)!!**

**LEMMA:** On a Riemannian/Lorentzian manifold we can choose normal coordinates at p s.t. $g_{\mu\nu}|_p = 0$ and $g_{\mu\nu}|_p = \{\delta_{\mu\nu}, \eta_{\mu\nu}\}$
**RIEMANNIAN**
**LORENTZIAN**
> *1st derivative vanishes at p. i.e. locally flat (or minkowski)*
> *(ie locally looks like flat space (euclidean or minkowski))*
> *components of metric not covariant differentiation*

**PROOF:** The curve given in normal coordinates by $t \mapsto (t,0,...,0)$ is the APG with $\lambda(0)=p, \dot{\lambda}(0)=e_1$ by previous argument. But by defn. of coord basis this vector is $(\frac{\partial}{\partial x^1})|_p$. So $\{e_\mu\}$ is ON at p (e.g.) form an ON basis.
> *orthonormal?*
> *thing is if we pick the initial basis $\{e_\mu\}$ to be orthonormal then the geodesics will point in orthogonal directions which means the metric looks like some $g_{\mu\nu}|_p \approx \delta_{\mu\nu}$.*

---


### CURVATURE

> *look for defn intrinsic to manifold that tells us it's not flat. (do this by considering parallel transport)*
> *Sphere, transporting a vector around changes the angle, tells us it is curved*

#### Parallel transport

> *noteworthly, if $\lambda$ is a curve with tangent vector $X^a$, then a tensor field $T$ is parallelly transported along $\lambda$ then $\nabla_X T = 0$*

Suppose $\lambda: I \to M$ is a curve with tangent vector $\dot{\lambda}(t)$.
If we say a tensor field $T$ is parallely transported/propagated along $\lambda$.
$$\nabla_{\dot{\lambda}} T = 0 \quad \text{on} \quad \lambda \quad (PP)$$
> *looks a bit like geodesic eqn (GE)*

* If $\dot{\lambda}$ is an APG then $\dot{\lambda}$ is parallely propagated along $\lambda$.
> *A tangent vector along itself is a geodesic is a parallelly transported along the curve.*

* A parallely propagated tensor is determined everywhere on $\lambda$ by its value at one point.

**(E.g.)** If $T$ is a (1,1) tensor then in coordinates (PP) becomes
$$0 = \frac{dx^\mu}{dt}\nabla_\mu T^\nu_\sigma = \frac{dx^\mu}{dt} (\partial_\mu T^\nu_\sigma + \Gamma^\nu_{\rho\mu}T^\rho_\sigma - \Gamma^\rho_{\sigma\mu}T^\nu_\rho)$$
> *use $\nabla_X T^\nu_\sigma = X^\mu\partial_\mu T^\nu_\sigma + ...$ then $X^\mu = \frac{dx^\mu}{dt}$*

but $\frac{d}{dt} T^\nu_\sigma = \frac{dx^\mu}{dt}\partial_\mu T^\nu_\sigma$ so
> *1st partial deriv of T in $\mu$ direction is $\frac{d}{dt}(T^\nu_\sigma)$*
> *... since the components are scalar*

$$0 = \frac{d}{dt} T^\nu_\sigma + \Gamma^\nu_{\rho\mu}T^\rho_\sigma \frac{dx^\mu}{dt} - \Gamma^\rho_{\sigma\mu}T^\nu_\rho \frac{dx^\mu}{dt}$$
> *linear in T*
> *1st order eqn $\implies$ solution determined entirely by values at a point.*

This is a 1st order linear ODE for $T^\nu_\sigma(\lambda(t))$, so ODE theory gives a unique solution once $T^\nu_\sigma(\lambda(0))$ specified.
> *T uniquely determined if know its value at one point*

* Parallel transport along a curve from p to q gives an **isomorphism** between tensors at p and q. This **depends** on the choice of curve in general.
> *important*
> *is order 1 ... is $T^\nu_\sigma$ (if we're considering the components of T)*
> *the isomorphism depends on the choice of path. On a curved manifold, parallel transporting around a loop may not return you to the same tensor. Isomorphism means the map is invertible & preserves the tensor structure i.e. maps Tensors to Tensors. wiki: structure preserving map that can be reversed by an inverse mapping.*

---


*Lecture 11*

*4.11.24*

### THE RIEMANN TENSOR

The Riemann tensor captures the extent to which parallel transport depends on the curve.

**LEMMA:** Given $X,Y,Z$ vector fields, $\nabla$ a connection, define
$$ R(X,Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z $$
Then $(R(X,Y)Z)^a = R^a{}_{bcd} X^b c^Y d^Z b$ for a (1,3)-tensor $R^a{}_{bcd}$, the **Riemann tensor**.

**PROOF:** Suppose $f$ is smooth function, then
> *prove it is a tensor by showing it is linear in X, Y, Z*

$R(fX,Y)Z = \nabla_{fX}\nabla_Y Z - \nabla_Y\nabla_{fX}Z - \nabla_{[fX,Y]}Z$
$= f\nabla_X\nabla_Y Z - \nabla_Y(f\nabla_X Z) - \nabla_{f[X,Y]-Y(f)X}Z$
$= f\nabla_X\nabla_Y Z - f\nabla_Y\nabla_X Z - Y(f)\nabla_X Z - f\nabla_{[X,Y]}Z + Y(f)\nabla_X Z$
$= fR(X,Y)Z$.
> *linear in X*

Since $R(X,Y)Z = -R(Y,X)Z$, we have $R(X,fY)Z = fR(X,Y)Z$
> *(antisymmetry check linear in Y,Y means we don't need to)*
> *linear in Y*

**Exercise** check $R(X,Y)(fZ) = fR(X,Y)Z^{(2)}$
> *since this is linear in its arguments for X,Y,Z above, we have 3 results.*
> *this is fine (remember $\nabla_X(fY) = f\nabla_X Y + X(f)Y$, etc)*
> *role mostly as summons for tensors.*

Now suppose we pick a basis $\{e_\mu\}$ with dual basis $\{e^\nu\}$.
$R(X,Y)Z = R(X^\rho e_\rho, Y^\sigma e_\sigma)(Z^\nu e_\nu) = X^\rho Y^\sigma Z^\nu R(e_\rho, e_\sigma)e_\nu = R^\mu{}_{\nu\rho\sigma}X^\rho Y^\sigma Z^\nu e_\mu$
> *holds in one basis so holds in any basis*

where $R^\mu{}_{\nu\rho\sigma} = e^\mu(R(e_\rho, e_\sigma)e_\nu)$ are components of $R^\mu_{\nu\rho\sigma}$ in this basis. Since result holds in one basis, it holds in all bases.
> *interlude: why partial derivatives commute?*

In a coordinate basis $e_\mu = \frac{\partial}{\partial x^\mu}$ and $[e_\mu, e_\nu]=0$. so
$R(e_\rho,e_\sigma)e_\nu = \nabla_{e_\rho}(\nabla_{e_\sigma}e_\nu) - \nabla_{e_\sigma}(\nabla_{e_\rho}e_\nu) = \nabla_{e_\rho}(\Gamma^\tau_{\nu\sigma} e_\tau) - \nabla_{e_\sigma}(\Gamma^\tau_{\nu\rho}e_\tau)$
$= \partial_\rho(\Gamma^\tau_{\nu\sigma})e_\tau + \Gamma^\mu_{\tau\rho}\Gamma^\tau_{\nu\sigma} e_\mu - \partial_\sigma(\Gamma^\tau_{\nu\rho})e_\tau - \Gamma^\mu_{\tau\sigma}\Gamma^\tau_{\nu\rho}e_\mu$
Hence $R^\mu{}_{\nu\rho\sigma} = \partial_\rho(\Gamma^\mu_{\nu\sigma}) - \partial_\sigma(\Gamma^\mu_{\nu\rho}) + \Gamma^\tau_{\nu\sigma}\Gamma^\mu_{\tau\rho} - \Gamma^\tau_{\nu\rho}\Gamma^\mu_{\tau\sigma}$

> *note, $R^\mu_{\nu\rho\sigma}$ are components of $R$.*
> *$R^\mu_{\nu\rho\sigma} e_\mu = ( ... ) e_\mu$*
> *since these are not tensor components, then we can treat them as scalars (typo?). since components are scalars*
> *(especially 1st term)*

---


In normal coordinates we can drop the last two terms.
> *(if we also have $\partial g=0$ then we can have $\partial\Gamma=0$)*
> *local flatness at a stop-gap means the deriv of connection must vanish at p, so $\partial\Gamma=0$?*

**Example:** For the Levi-Civita connection of **Minkowski space** in an inertial frame, $\Gamma^\mu_{\nu\sigma}=0$, so $R^\mu_{\nu\sigma\tau}=0$.
> *inertial frame $\implies$ $\Gamma=0$ true in that basis, so true in all*

hence $R^a{}_{bcd}=0$. Such, conversely, for a Lorentzian spacetime with flat L-C connection, we can locally find coordinates such that $g_{\mu\nu} = \text{diag}(-1,1,1,1)$.
> *everywhere is called flat*
> *the spacetime is "locally isometric" to Minkowski spacetime.*

**A note of caution:**
$(\nabla_X \nabla_Y Z)^c = X^a \nabla_a (Y^b \nabla_b Z^c) \neq X^a Y^b \nabla_a \nabla_b Z^c$
hence $(R(X,Y)Z)^c = X^a \nabla_a (Y^b \nabla_b Z^c) - Y^a \nabla_a (X^b \nabla_b Z^c) - [X,Y]^b \nabla_b Z^c$
$= X^a Y^b \nabla_a \nabla_b Z^c - Y^a X^b \nabla_a \nabla_b Z^c + (\nabla_X Y - \nabla_Y X - [X,Y])^b\nabla_b Z^c$
So if $\nabla$ is torsion free,
$$\nabla_a \nabla_b Z^c - \nabla_b \nabla_a Z^c = R^c{}_{dab} Z^d \quad \text{RICCI IDENTITY}$$

on ex. sheet 2 there's a question to generalise for an expression for $\nabla_{e_a} \nabla_{e_b} T^{c_1...c_m}_{d_1...d_n}$.

We can construct a new tensor from $R^a{}_{bcd}$ by **contraction**: *(noteworthy: $R^a_{acd} = R^c_{acd} \implies R(c,b;c,d)=0$)*
**Definition:** The **RICCI TENSOR** is the (0,2)-tensor
> *(...surely not middle?)*

$$R_{ab} = R^c{}_{acb}$$

> *why not contract diff 2 indices, $R^c_{adc}=0$; others are the same up to a minus sign so this is the only one of interest.*

Suppose $X,Y$ are vector fields satisfying $[X,Y]=0$.

*(Diagram shows a small parallelogram A-B-C-D with sides generated by flowing along X and Y. A vector Z is parallel transported along the path. The initial vector is at A, and the final vector, also at A after the loop, is different, denoted Z'. A separate diagram shows two commuting vector fields X, Y.)*
> *Note: if Z is parallel transported around the curves, it's not necessarily the same vector*
> *flow along the integral curves of X and Y, and point them. Flow along integral "curve of X" through a point, then back etc. from back to start, I'd get to the same point.*

---


Go from A to B by flowing parameter distance $\epsilon$ along int. curve of X.
B to C " " $\epsilon$ " " " " Y.
C to D " " $-\epsilon$ " " " " X.
D to A " " $-\epsilon$ " " " " Y.
Since $[X,Y]=0$, we indeed return to start.

**CLAIM:** (See H. Reall notes)
> *remark: R-tensor measures path-dependence of parallel transport. can be interpreted as the expected parallel transport error for transporting a vector around the closed loop approx.*
> *(assuming levi-civita connection?)*

If Z is parallely transported around ABCD to a vector $Z'$, then
$(Z'-Z)^\mu = \epsilon^2 R^\mu{}_{\nu\rho\sigma}Z^\nu X^\rho Y^\sigma + O(\epsilon^3)$
> *parallel transport up, around, down, back. $(Z')_B$ is just $(Z)_A$.*
> *$(\delta Z^\mu_A = Z^\mu_{A \to B \to C \to D \to A} - Z^\mu_A)$*
> *$(\delta R \implies R^a_{bcd} Z^b X^c Y^d) = \lim \frac{\Delta Z^a}{\delta S \delta t}$*

### Geodesic Deviation
> *This is one way to visualise the R-tensor. Another way: geodesic deviation. R-tensor measures path dependence of parallel transport.*

Let $\nabla$ be a symmetric connection. Suppose $\lambda: I \to M$ is an APG through p. We can pick normal coordinates centred at p such that $\lambda$ is given by $t \mapsto (t,0,...,0)$.
> *$\lambda_s(t)$*
> *$s\gamma^\mu(t)$*

Suppose we start a geodesic with
> *start close to $\lambda$, in almost the same direction. find geo starting prop of ODE that curve $\lambda_s(t)$ looks like curve $\lambda_0(t)$ for a with connection + H. orders?*

$\dot{\lambda}_s(0) = sX^\mu_0$ $|s| \ll 1$
$\dot{x}^\mu_s(0) = s\dot{x}^\mu_0 + (1,0,...,0)$

Then we find $x^\mu_s(t) = x^\mu(s,t) = (t,0,0,...,0) + sY^\mu(t) + O(s\epsilon)$
$Y^\mu(t) = \frac{\partial x^\mu}{\partial s}|_{s=0}$ are components of a vector field along $\lambda$.
> *$x^\mu(s,t)|_s=x^\mu(0,t) + s \frac{\partial x^\mu}{\partial s}(s,t)|_{s=0}$*

Measuring the (infinitesimal) deviation of the geodesics, we have
$\frac{\partial^2 x^\mu}{\partial t^2} + \Gamma^\mu_{\nu\sigma}(x^\alpha(s,t))\frac{\partial x^\nu}{\partial t}\frac{\partial x^\sigma}{\partial t} = 0$. take $\frac{\partial}{\partial s}|_{s=0}$
=> $\frac{\partial^2 Y^\mu}{\partial t^2} + \frac{\partial\Gamma^\mu_{\nu\sigma}}{\partial x^\rho}|_{s=0} Y^\rho T^\nu T^\sigma + 2\Gamma^\mu_{\rho\sigma} \frac{\partial Y^\rho}{\partial t} T^\sigma = 0$
> *$T^\mu = \frac{\partial x^\mu}{\partial t}|_{s=0}$*

=> $T^\nu(\partial_\nu Y^\mu)_{,s} + 2\Gamma^\mu_{\rho\sigma}(\Gamma^\rho_{\tau\nu})_{,s=0}T^\tau T^\nu + 2\Gamma^\mu_{\sigma\rho}\frac{\partial Y^\sigma}{\partial t}T^\rho = 0$
At $p=0, \Gamma=0$, so $T^\nu T^\sigma (\partial_\nu \Gamma^\mu_{\sigma\rho} - \partial_\rho \Gamma^\mu_{\sigma\nu})_{,s} + (\partial_\rho \Gamma^\mu_{\nu\sigma}) T^\nu T^\sigma Y^\rho = 0$.
> *t.b.c... only thing left over is when partial deriv hits $\Gamma$*

(lect.12) => $T^\nu T^\sigma (\partial_\sigma Y^\mu)_{,\nu} + (\partial_\rho\Gamma^\mu_{\sigma\nu} - \partial_\nu\Gamma^\mu_{\sigma\rho})T^\nu T^\sigma Y^\rho = 0$
> *vanishes at p=0!*
=> $(\nabla_T \nabla_T Y)^\mu + R^\mu{}_{\rho\sigma\nu} T^\rho T^\sigma Y^\nu = 0$
=> $$\nabla_T \nabla_T Y + R(Y,T)T = 0 \quad \text{GEODESIC DEVATION. JACOBI EQN.}$$

> *note: $\nabla_T \nabla_S = R(T,S)T$*
> *main pt: $T^a \nabla_a (T^b \nabla_b S^d) = T^a T^b S^c R^d_{cab}$*
> *result => curvature results in relative acceleration of geodesics.*
> *if geodesics are initially parallel, in flat space they remain parallel forever. In curved space they will deviate.*

---


*Lecture 12*

*6.11.24*
> *SUMMARY*
> - *$R^a{}_{b(cd)}=0$*
> - *$R^a{}_{[bcd]}=0$*
> - *$R_{[ab]cd} = 0$; $R_{ab[cd]}=0$ (TORSION FREE)*
> - *$R_{abcd}=R_{cdab}$ (D LEVI CIVITA)*

### SYMMETRIES OF THE RIEMANN TENSOR

From the definition it's clear that $R^a{}_{bcd} = -R^a{}_{bdc} \implies R^a{}_{b(cd)}=0$.

**PROPOSITION:** If $\nabla$ is torsion free then $R^a{}_{[bcd]}=0$.
> *antisymmetric over 3 downstairs indices*

**PROOF:** Fix $p \in M$, choose normal coordinates at p and work in coordinate basis, then
> *use normal coords to simplify*
> *torsion free*

$\Gamma^\mu_{\nu\sigma}|_p = 0$ and $\Gamma^\mu_{[\nu\sigma]}=0$ everywhere
$R^\mu{}_{\nu[\rho\sigma]}|_p = (\partial_\rho \Gamma^\mu_{\sigma]\nu})|_p - \partial_{[\sigma}\Gamma^\mu_{\rho]\nu}|_p$.
> *antisymmetric over identical indices then both these terms vanish*

$\implies R^\mu{}_{\nu[\rho\sigma]}|_p = 0$ as $\partial_\rho \Gamma^\mu_{\sigma\nu}|_p = \partial_\sigma\Gamma^\mu_{\rho\nu}|_p$
> *use the trick again to prove stuff: establish it holds at arbitrary point $p$ so holds everywhere!*
> *(normal coords)*

p arbitrary and so $R^\mu{}_{\nu[\rho\sigma]} = 0$ everywhere.
> *this is the long way or maybe not. note one form of $\Gamma$ is symmetric in $\nu\sigma$, and then $\Gamma$ is antisymmetric in $\nu\rho$ so antisymmetrizing both vanish. since we have antisymmetry part, $\nabla_{e_a} e_b = \nabla_{e_b} e_a$*

**PROPOSITION:** If $\nabla$ is torsion free then the **Bianci Identity** holds:
> *There is a 1st and 2nd bianchi identity. if we say just bianchi identity we usually mean this one.*
> *and $\nabla g=0$*

$$R^a{}_{b[cd;e]} = 0$$

**PROOF:** Choose coordinates as above then $R^\mu{}_{\nu\rho\sigma;\tau}|_p = R^\mu{}_{\nu\rho\sigma,\tau}|_p$.
> *... (contracted into $R,T$ vanishes at p)*

schematically, $R \sim \partial\Gamma + \Gamma\Gamma$ so $\partial R \sim \partial^2\Gamma + \partial\Gamma\Gamma$.
and since $\Gamma|_p=0$ we deduce
$R^\mu{}_{\nu\rho\sigma;\tau}|_p = \partial_\tau\partial_\rho\Gamma^\mu_{\sigma\nu}|_p - \partial_\tau\partial_\sigma\Gamma^\mu_{\rho\nu}|_p$
> *only terms that are going to contribute are linear and 2 order derivatives of $\Gamma$*

By symmetry of the mixed partial derivatives, we see $R^\mu{}_{\nu[\rho\sigma;\tau]}|_p = 0$.
> *same logic as above: 1st term symmetric in $(\rho,\tau)$ so when we antisymmetrise over $(\rho,\sigma,\tau)$ it all vanishes*

since p arbitrary result follows.
> *These identities hold for any torsion free connection.*

* Suppose $\nabla$ is the Levi-Civita connection of a manifold with metric g. We can lower an index with $g_{ab}$ and consider $R_{abcd}$.
> *Claim $R_{abcd}$ has additional symmetries. lots of symmetries! reminder since $g_{ab}\nabla_c X^a = \nabla_c(g_{ab} X^a) = (\nabla_c g_{ab})X^a + g_{ab}\nabla_c X^a$*

**PROPOSITION:** $R_{abcd}$ satisfies $R_{abcd}=R_{cdab}$ ($\implies R_{(ab)cd}=0$)

**PROOF:** Pick normal coordinates at p so that $\partial_\mu g_{\nu\rho}=0$. We notice that
> *connection components vanish at p but also partial derivs. vanish*

$0 = \partial_\nu\partial_\tau g_{\sigma\rho}|_p = \partial_\nu(g^{\alpha\beta}g_{\sigma\rho})|_{p} = (\partial_\nu\partial_\tau g^{\alpha\beta})g_{\sigma\rho}|_p$
because $\partial_\mu g_{\nu\rho}|_p=0$.
> *exercise is non degenerate*

---


hence $\partial_\rho(\Gamma^\kappa_{\tau\sigma})|_p = \frac{1}{2}\partial_\rho(g^{\kappa\mu}(g_{\mu\sigma,\tau} + g_{\mu\tau,\sigma} - g_{\sigma\tau,\mu}))|_p$
$= \frac{1}{2} g^{\kappa\mu} (g_{\mu\sigma,\tau\rho} + g_{\mu\tau,\sigma\rho} - g_{\sigma\tau,\mu\rho})|_p$
we have $R_{\mu\nu\rho\sigma}|_p = g_{\mu\kappa}(\partial_\rho \Gamma^\kappa_{\sigma\nu} - \partial_\sigma \Gamma^\kappa_{\rho\nu})|_p$
> *let's have a look at this not using normal coords if you don't want to sleep tonight!*

$= \frac{1}{2}(g_{\mu\sigma,\nu\rho} + g_{\nu\rho,\mu\sigma} - g_{\rho\sigma,\mu\nu} - g_{\mu\rho,\nu\sigma} - g_{\nu\sigma,\mu\rho} + g_{\rho\sigma,\mu\nu})|_p$
This satisfies $R_{\mu\nu\rho\sigma}|_p = R_{\rho\sigma\mu\nu}|_p$ hence true everywhere.

**COROLLARY:** The Ricci tensor is symmetric $R_{ab}=R_{ba}$.
> *(metric we can go further, use the contraction. in any basis $R_{ab}=g^{cd}R_{cadb}=g^{cd}R_{dbca}=R_{da}=R_{ba}$)*

**DEFINITION:**
* The **Ricci scalar** (scalar curvature) is $R = R^a_a = g^{ab}R_{ab}$.
* The **Einstein tensor** is $G_{ab} = R_{ab} - \frac{1}{2}g_{ab}R$.

**Exercise:** The Bianchi identity implies $\nabla^a G_{ab}=0$. (contracted Bianchi Identity)
> *this gives the gauge group of einstein eqns.*
> *important for when we construct einstein equations*

### DIFFEOMORPHISMS AND THE LIE DERIVATIVE

Suppose $\psi: M \to N$ is a smooth map, then $\psi$ induces maps between corresponding vector/covector bundles.

**DEFINITION:** Given $f: N \to \mathbb{R}$, the **PULL BACK** of $f$ by $\psi$ is the map $\psi^*f: M \to \mathbb{R}$ given by $(\psi^* f)(p) = f(\psi(p))$.
> *"f acting on image of p"*

*(Diagram showing a map $\psi$ from manifold M to N, and a function f from N to R. The pullback $\psi^*f$ is a map from M to R.)*
> *notes: $d(\psi^*f)=f\circ d\psi$?*

**DEFINITION:** Given $X \in T_p M$, we define the **PUSH FORWARD** of $X$ by $\psi$, $\psi_* X \in T_{\psi(p)} N$ as follows,
Let $\lambda: I \to M$ be a curve with $\lambda(0)=p, \dot{\lambda}(0)=X$. Then set $\tilde{\lambda} = \psi \circ \lambda$, $\tilde{\lambda}: I \to N$ gives a curve in N with $\tilde{\lambda}(0)=\psi(p)$.
We set $\psi_* X = \dot{\tilde{\lambda}}(0)$.
> *notes: "push forward a curve in M to a curve $\psi \circ \lambda$ in N, hence we can push forward vectors from M to N"*
> *notes: if $\dot{\lambda}(0)=X$, then $\psi_*X$ is defined as $d(\psi \circ \lambda)/dt$. $\tilde{\lambda} = \psi \circ \lambda$ is a curve in N with $\dot{\tilde{\lambda}}(0)=\psi_* X$.*

*(Diagram shows a curve in M with a tangent vector being mapped by $\psi$ to a curve in N with a corresponding tangent vector.)*

**Note:** If $f: N \to \mathbb{R}$ then $\psi_* X (f) = \frac{d}{dt}(f \circ \tilde{\lambda}(t))|_{t=0} = \frac{d}{dt}(f \circ \psi \circ \lambda(t))|_{t=0} = X(\psi^*f)$.

---


**Exercise:** If $x^\mu$ are coords on M near p, $y^\alpha$ are coords on N near $\psi(p)$ then $\psi$ gives a map $y^\alpha(x^\mu)$. Show that in a coordinate basis
$(\psi_* X)^\alpha_p = X^\mu \frac{\partial y^\alpha}{\partial x^\mu}|_p$ or $(\psi_* \frac{\partial}{\partial x^\mu})_p = (\frac{\partial y^\alpha}{\partial x^\mu}|_p) \frac{\partial}{\partial y^\alpha}$

On cotangent bundle we go backwards.
**DEFINITION:** If $\eta \in T^*_{\psi(p)}N$, then the pullback of $\eta$, $\psi^*\eta \in T^*_p M$, is defined by
$$\psi^*\eta(X) = \eta(\psi_* X) \quad \forall X \in T_p M$$
> *worth check where these identities come from*

**Note:** If $f: N \to \mathbb{R}$, $\psi^*(df)[X] = df[\psi_* X] = \psi_* X(f) = X(\psi^*f)$
$= d(\psi^*f)[X]$.
$\implies \psi^*df = d(\psi^*f)$.
> *since X arbitrary*
> *use $\psi_*\eta(X) = \eta(\psi_* X)$. and $df(X)=X(f)$. so $\psi_*X(f) = X(\psi^*f)$.*
> *i.e. gradient commutes with pull back*

**Exercise:** With notation as before, show that
$(\psi^*\eta)_\mu = \frac{\partial y^\alpha}{\partial x^\mu}|_p \eta_\alpha$ or $\psi^*(dy^\alpha)_p = (\frac{\partial y^\alpha}{\partial x^\mu}|_p) (dx^\mu)_p$.

* We can extend the pullback to map a (0,s)-tensor $T$ at $\psi(p) \in N$ to a (0,s)-tensor $\psi^* T$ at $p \in M$ by
requiring $(\psi^*T)(X_1,...,X_s) = T(\psi_* X_1,...,\psi_* X_s) \quad \forall X_i \in T_p M$.

* Similarly we can push forward a (s,0)-tensor $S$ at $p \in M$ to a (s,0)-tensor $\psi_*S$ at $\psi(p) \in N$ by
$(\psi_*S)(\eta_1,...,\eta_s) = S(\psi^*\eta_1,...,\psi^*\eta_s) \quad \forall \eta_i \in T^*_{\psi(p)}N$.
> *note here in components pushforward maps things to upstairs indices, pullback maps things to downstairs indices*

* If $\psi:M \to N$ has the property that $\psi_*: T_p M \to T_{\psi(p)}N$ is injective (one-to-one), we say $\psi$ is an **immersion**. (dim M > dim N)
> *see luca notes for examples*

* If $N$ is a manifold with metric g, and $\psi:M \to N$ is an immersion, we can consider $\psi^*g$. pull back of g.

* If $g$ is Riemannian, then $\psi^*g$ is non-degenerate and positive definite, so defines a metric on M, the **induced metric**.
> *Note: if we pull back a lorentzian metric, there is no guarantee it ends up on the manifold. we will talk about lorentzian metric later.*
> *SUMMARY:*
> - *maps between manifolds induce maps between cotangent bundles*
> - *$\psi^* f = f \circ \psi$; $\psi_* X(f) = X(\psi^* f)$; $\psi^* \eta(X) = \eta(\psi_* X)$*
> - *$(\psi: M \to N, X \in T_p M, \eta \in T^*_{\psi(p)}N)$.*
> - *if $\psi_*: T_p M \to T_{\psi(p)}N$ injective, $\psi$ is an IMMERSION*
> - *if $\psi$ is an immersion, in riemannian manifold then $\psi^*g$ is a metric on M*

*(Diagram shows a cylinder being mapped to a flattened shape, illustrating an immersion where vectors might map to the same image vector.)*

---


*Lecture 13*

*8.11.24*
> *$(\psi^*g)_{\mu\nu} = (\frac{\partial y^\alpha}{\partial x^\mu})(\frac{\partial y^\beta}{\partial x^\nu})g_{\alpha\beta}$*

**Exercise:** Let $(N,g) = (\mathbb{R}^3, \delta)$, $M=S^2$. Let $\psi$ be the map taking a point on $S^2$ w/ spherical coordinates $(\theta, \phi)$ to $(x^1, x^2, x^3) = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ then
$\psi^*((dx^1)^2 + (dx^2)^2 + (dx^3)^2) = d\theta^2 + \sin^2\theta d\phi^2$?
> *i.e. take the standard metric for $\mathbb{R}^3$, pull it back via the embedding to the sphere. Find the induced metric.*

If $\psi$ immersion, $(N,g)$ is Lorentzian, then $\psi^*g$ is not in general a metric on M. There are 3 important cases:
* $\psi^*g$ is a **Riemannian** metric $\implies \psi(M)$ is **spacelike**
* $\psi^*g$ is a **Lorentzian** metric $\implies \psi(M)$ is **timelike**
* $\psi^*g$ is everywhere **degenerate** $\implies \psi(M)$ is **null**

*(Diagram shows Minkowski spacetime with axes $x^0, x'$, indicating spacelike (SL), timelike (TL), and null regions. An example is given for a map from $\mathbb{R}^3$ to Minkowski space.)*
> *$\psi: (y^1,y^2,y^3) \to (0, y^1,y^2,y^3)$, $\psi(\mathbb{R}^3)$ is spacelike. $(\psi^*g)_{ij} = (...) \to ds^2 = (dy^1)^2 + (dy^2)^2 + (dy^3)^2$ riemannian spacetime*

Recall that $\psi:M \to N$ is a **diffeomorphism** if it is bijective with a smooth inverse. If we have a diffeomorphism we can push forward a general (r,s)-tensor at p to an (r,s)-tensor at $\psi(p)$ by
> *if we have a diffeomorphism we can extend our definitions of pushforward & pullback to apply to a general tensor.*

$\psi_*T(\eta_1,...,\eta_r, X_{r+1},...,X_{r+s}) = T(\psi^*\eta_1,...,\psi^*\eta_r, \psi^{-1}_* X_{r+1},...,\psi^{-1}_* X_{r+s})$
> *(r,s) lower indices*
> *(1,1) tensor components $(\psi_*T)^\alpha_\beta = (\frac{\partial y^\alpha}{\partial x^\mu})(\frac{\partial x^\nu}{\partial y^\beta})T^\mu_\nu$*

Define a pull-back by $\psi^* = \psi^{-1}_*$.
If M,N are diffeomorphisms, we often don't distinguish between them we can think of $\psi: M \to M$.
> *if $\psi$ an isometry, means $\psi^*g=g$*

We say a diffeomorphism $\psi:M \to M$ is a **symmetry** of T if $\psi_*T = T$. If T is the metric, we say it is an **isometry**. e.g. in Minkowski space w/ an inertial frame
$\psi(x^0, x^1,...,x^n) = (x^0+h, x^1,...,x^n)$ is a symmetry of g.
> *note: T is a tensor field*

An important class of diffeomorphisms are those generated by a vector field.
If X is a smooth vector field, we associate to each point $p \in M$ the point $\psi_t^X(p) \in M$ given by flowing a parameter distance t along the integral curve of X starting at p.

---


Suppose $\psi_t(p)$ is well defined for all $t \in I \subset \mathbb{R}$ for each $p \in M$ then $\psi_t: M \to M$ is well defined a diffeomorphism for all $t \in I$.
> *flows distance t along the integral curve of X defined for small enough t. $\psi_t: M \to M$ is a transformation of M. is a diffeomorphism.*

Further
* If $t,s, t+s \in I$ then $\psi_t \circ \psi_s = \psi_{t+s}$ and $\psi_0 = \text{id}$ (*)
* If $I = \mathbb{R}$ this gives $\{\psi_t\}_{t \in \mathbb{R}}$ the structure of a **one parameter abelian group**. *(don't worry what this is if you don't know)*
* If $\psi_t$ is any smooth family of diffeomorphisms satisfying (*), we can define a vector field by
$X(p) = \frac{d}{dt}(\psi_t(p))|_{t=0}$. then $\dot{\psi}_t = \psi_t \circ X$

We can use $\psi_t^*$ to compare tensors at different points as $t \to 0$. This gives a new which is derivative.
> *note to self (gpt). integral curve... only defined one curve at a time, tells you one point p. not a diffeomorphism. can't pull back tensors. How $\psi_t^*$: if we flow all points p we can recover entire manifold from the $\psi_t$'s (as a diffeomorphism).*

### THE LIE DERIVATIVE
Suppose $\psi_t: M \to M$ is the smooth one-parameter family of diffeomorphisms generated by a vector field X.

**DEFINITION:** For a tensor field T, the **lie derivative** of T with respect to X is
$(\mathcal{L}_X T)_p = \lim_{t \to 0} \frac{((\psi_t)^*T)_p - T_p}{t}$

*(Diagram shows a point p being moved by the flow to $\psi_t(p)$. The tensor T at $\psi_t(p)$ is pulled back to p to be compared with T at p.)*

It's easy to see that for constant $\alpha, \beta$ and (r,s) tensors S, T
$\mathcal{L}_X(\alpha S + \beta T) = \alpha \mathcal{L}_X S + \beta \mathcal{L}_X T$.

To see how $\mathcal{L}_X$ acts in components, it's helpful to construct coordinates adapted to X.
Near p we can construct an (n-1)-surface $\Sigma$ which is transverse to X (line nowhere tangent). Pick coords $x^i$ on $\Sigma$ and assign the coordinate $(t,x^i)$ to the point at parameter distance t along the integral curve of X satisfying starting at $x^i$ on $\Sigma$.
In these coords, $X = \frac{\partial}{\partial t}$ and $\psi_\tau(t, x^i) = (t+\tau, x^i)$.
So if $y^\mu = (\psi_t^* x^\mu)$ then $\frac{\partial y^\mu}{\partial x^\nu} = \delta^\mu_\nu$ and
$[(\psi_t)^*T]^{\mu_1...\mu_r}_{\nu_1...\nu_s} (t,x^i) = T^{\mu_1...\mu_r}_{\nu_1...\nu_s}(t+\tau, x^i)$
> *flow takes me from point p to point $\psi_t(p)$, then pullback takes from $\psi_t(p)$ back to p. $\psi_t^*(x^\mu)$ are coords of p.*

---


thus $(\mathcal{L}_X T)^{\mu_1...\mu_r}_{\nu_1...\nu_s}|_p = \frac{\partial T^{\mu_1...\mu_r}_{\nu_1...\nu_s}}{\partial t}|_{v_1...v_s}|_p$.
So in these coords, $\mathcal{L}_X$ acts on components by $\frac{\partial}{\partial t}$.
In particular we immediately see
* $\mathcal{L}_X$ obeys leibniz:
$\mathcal{L}_X(S \otimes T) = (\mathcal{L}_X S) \otimes T + S \otimes (\mathcal{L}_X T)$. $\mathcal{L}_X$ commutes w/ contraction

To write $\mathcal{L}_X$ in a coordinate-free fashion, we can simply seek a basis independent expression that agrees with $\mathcal{L}_X$ in these coords.
* **E.g.** For a function $f$, $\mathcal{L}_X f = \frac{\partial f}{\partial t} = X(f)$ in these coords.
> *$\mathcal{L}_X f = X(f)$, but this is for scalars so this must be valid in any basis.*

* For a vector field Y we observe that
$(\mathcal{L}_X Y)^\mu = \frac{\partial Y^\mu}{\partial t} = X^\sigma\partial_\sigma Y^\mu - Y^\sigma\partial_\sigma X^\mu = [X,Y]^\mu$
$\mathcal{L}_X Y = [X,Y]$
> *since in these coords $X=\frac{\partial}{\partial t}$ components are const so this vanishes.*

**Exercise:** In any coord basis show if $\omega_\alpha$ is a covector field,
$(\mathcal{L}_X \omega)_\mu = X^\sigma\partial_\sigma \omega_\mu + \omega_\sigma\partial_\mu X^\sigma$
If $\nabla$ is torsion free, $(\mathcal{L}_X \omega)_a = X^b\nabla_b \omega_a + \omega_b\nabla_a X^b$.
If $g_{ab}$ is a metric tensor, $\nabla$ Levi-Civita, then
$(\mathcal{L}_X g)_{ab} = \nabla_a X_b + \nabla_b X_a$
> *in normal coords $\partial_a g_{bc}=0$ so promote to $\nabla_a$ true in general*

If $\psi_t$ is a one-parameter family of isometries for a manifold with metric g, then $\mathcal{L}_X g = 0$.
Conversely, if $\mathcal{L}_X g=0$ then X generates a one-parameter family of isometries.
> *isometry $\psi^* g = g$, this means $\psi$ is a symmetry of the metric.*

**DEFINITION:** A vector field K satisfying $\mathcal{L}_K g=0$ is called a **KILLING VECTOR**. It satisfies **Killing's equation**.
$\nabla_a K_b + \nabla_b K_a = 0$ ($\nabla$ Levi-Civita)

**LEMMA:** Suppose K is killing and $\lambda: I \to M$ is a geodesic of the Levi-Civita connection. Then $g_{ab}\dot{\lambda}^a K^b$ is const. along $\lambda$.
> *tangent to curve is $\dot{\lambda}$*

**PROOF:** $\frac{d}{dt}(K_b \dot{\lambda}^b) = \dot{\lambda}^a \nabla_a(K_b \dot{\lambda}^b) = (\nabla_a K_b)\dot{\lambda}^a \dot{\lambda}^b + K_b(\dot{\lambda}^a\nabla_a \dot{\lambda}^b)$
$= \frac{1}{2}(\nabla_a K_b + \nabla_b K_a)\dot{\lambda}^a \dot{\lambda}^b = 0$.
> *(by killing)*
> *vanishes for geodesic*

---


*Lecture 14*

*11.11.24*

### Physics in Curved Spacetime

> *For any spacetime $(M,g)$ for any $p \in M$ we can introduce normal coordinates s.t. in a neighborhood of p, $g_{\mu\nu} = \eta_{\mu\nu}$ + higher order terms so locally looks like minkowski (to 1st order)*
> *notes: beyond this, in an inertial frame, levi-civita $\nabla_\mu$ is the same as $\partial_\mu$, so to promote an eqn from spec. rel. system to any arbitrary basis, we replace $\partial_\mu$ with $\nabla_\mu$.*

#### Minkowski space (special relativity)
We review physical theorems in Minkowski $\mathbb{R}^{1+3}$ equipped with $\eta_{\mu\nu} = \text{diag}(-1,1,1,1)$, we set $c=1$.
> *various ways to validate equations:
> 1) $\partial \to \nabla$
> 2) $\eta \to g$*

**1. Klein-Gordon equation**
$\partial_\mu \partial^\mu \phi - m^2 \phi = 0$ (a)
Note that in inertial coords $\partial_\mu = \nabla_\mu$ so we can write this in a covariant manner as
> *promote partial to nabla*
$\nabla_a \nabla^a \phi - m^2 \phi = 0$
> *3) also replace $g_{ab}$ with arbitrary metric $g_{ab}$ (take $\nabla$ to be levi-civita connection of that metric) when we get an eq. valid in an arbitrary spacetime.*

Associate to (a) is the energy-momentum tensor
$T_{\mu\nu} = \partial_\mu\phi\partial_\nu\phi - \frac{1}{2}\eta_{\mu\nu}(\partial^\sigma\phi\partial_\sigma\phi + m^2\phi^2)$
or covariantly
$T_{ab} = \nabla_a\phi\nabla_b\phi - \frac{1}{2}\eta_{ab}(\nabla^c\phi\nabla_c\phi + m^2\phi^2)$
This satisfies $T_{ab} = T_{ba}$, (check) $\nabla^a T_{ab} = 0$.

**2. Maxwell's eqns**
The Maxwell field is an anti-symmetric (0,2)-tensor
$F_{\mu\nu} = -F_{\nu\mu}$ where $F_{0i}=E_i, F_{ij}=\epsilon_{ijk}B_k$ (i,j=1,2,3).
If $j^\mu$ is the charge current density, then Maxwell's eqns are
$\partial_\mu F^{\mu\nu} = 4\pi j^\nu$ $\implies$ $\nabla_a F^{ac} = 4\pi j^c$
$\partial_{[\mu} F_{\nu\sigma]} = 0$ $\implies$ $\nabla_{[a}F_{bc]} = 0$
(b)

Associated to (b) is the energy-momentum tensor
$T_{\mu\nu} = F_{\mu\sigma}F_\nu{}^\sigma - \frac{1}{4}\eta_{\mu\nu}F_{\sigma\tau}F^{\sigma\tau}$
$\to T_{ab} = F_a{}^c F_{bc} - \frac{1}{4}\eta_{ab}F_{cd}F^{cd}$
Check $T_{ab}=T_{ba}$, $\nabla^a T_{ab}=0$ for source-free Maxwell.

**3. Perfect Fluid**
> *$\nabla_a F_{a[\mu\nu]}$ and $F_{ab}=-F_{ba}$ (add this on paper 2024?)*

A perfect fluid is described by a local velocity field $u^\mu$.

---


normalise (idealised fluid (no viscosity), perfectly described by P, $\rho$ and $u^a$-velocity)
satisfying $u^\mu u_\mu = -1$, together with a pressure P and density $\rho$. They satisfy the first law of thermodynamics
$u^\mu\partial_\mu \rho + (p+\rho)\partial_\mu u^\mu = 0$
covariant eq $\to U^a \nabla_a \rho + (p+\rho)\nabla_a U^a = 0$
and Euler's equation (read as cov. form momentum law)
$(p+\rho)u^\nu\partial_\nu u^\mu + \partial^\mu p + u^\mu u^\nu\partial_\nu p = 0$
> *see cosmology course we deal w/ this*
> *project relay. moves along with fluid*
$\to (p+\rho)u^b\nabla_b u^a + \nabla^a p + u^a u^b\nabla_b p = 0$
> *because of $\nabla_a u^a=0$ you get another equation needed, eq of state? to close the system (3 deg.f.)*

Associated is the energy-momentum tensor
$T_{\mu\nu} = (p+\rho)u_\mu u_\nu + p\eta_{\mu\nu}$
$\to T_{ab} = (p+\rho)U_a U_b + p\eta_{ab}$
> *again $T_{ab}=T_{ba}, \nabla^a T_{ab}=0$*

Notice that in all these cases we can, if we wish, **promote** the Minkowski $\eta$ to a general Lorentzian metric g, and take $\nabla$ to be the Levi-Civita connection. Consider normal coordinates, we see that coords exist near any point $p \in M$, such that the physics described is approximately Minkowskian, with corrections of the order of curvature.
> *minimal coupling approach: we could add eq. relating to Tab that just happens to vanish in minkowski. we don't know if they exist but simplest approach is that they don't.*

### GENERAL RELATIVITY

In Einstein's theory of general relativity we postulate that spacetime is a 4-dimensional Lorentzian manifold $(M,g)$. We also require any matter model to consist of
* some matter fields $\Phi^\alpha$
* eqns of motion for $\Phi^\alpha$ which are expressed geometrically in terms of $g(+\nabla, R, ...)$
* an energy-momentum tensor $T_{ab}$ depending on $\Phi^\alpha$ satisfies $T_{ab}=T_{ba}, \nabla^a T_{ab}=0$.

The matter should reduce to a non-grav theory when $(M,g)$ is fixed to be Minkowski.
The metric $g$ should satisfy the **Einstein equation**
> *GRs geometrise grav.*

$R_{ab} - \frac{1}{2} g_{ab} + \Lambda g_{ab} = 8\pi G T_{ab}$
$\Lambda$ cosmological constant ($\Lambda > 0$ but small). G Newton's constant.
> *Einstein knew of this. He wanted divergence free, needed to add term $\Lambda g_{ab}$ to get there.*
> *LHS is purely metric, contains all geometry of matter. responsible for generating gravitation.*

The Einstein eqns together with the EoMs for $\Phi^\alpha$, constitute a coupled system which must be solved simultaneously.

---


### GEODESIC POSTULATE

Free test particles move along timelike/null **geodesics** if they have non zero/zero rest mass.
> *The principle of equivalence motivates the geodesic postulate!*

### Gauge Freedom
> *view einstein eqns as evolution equations*
> *we have a gauge freedom/gauge symmetry if there exist distinct configurations that are physically equivalent. A gauge transformation maps a field configuration to a physically equivalent configuration. Diffeomorphisms are gauge transformations in GR.*

Consider Maxwell with no sources
$\partial_\mu F^{\mu\nu}=0$ (M1)
$\partial_{[\mu}F_{\nu\sigma]}=0$ (M2)

a standard approach to solve M2 is to introduce the **gauge potential** $A_\mu$ s.t. $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$, then M1 becomes
$\partial_\mu\partial^\mu A_\nu - \partial_\nu\partial^\mu A_\mu = 0 \quad (*)$
We'd like to solve (*) given data at $\{x^0=0\}= \Sigma$. However this eq doesn't give a good evolution problem. (initial value at surface $\Sigma$)
If $\chi \in C^\infty_0(\mathbb{R}^{1+3})$ which vanishes near $\text{Supp}(F) \cap \Sigma$, then $A_\mu = \tilde{A}_\mu + \partial_\mu \chi$ will also solve (*) and $\partial_{[\mu}\tilde{A}_{\nu]} = \partial_{[\mu}A_{\nu]}$.
> *If gauge trans. does not change the observable quantity $F_{\mu\nu}$*

To resolve this, we can fix a gauge e.g. if we assume $\partial^\mu A_\mu = 0$ then (*) becomes $\Box A_\nu = 0$.
> *lorenz gauge condition*

a wave eq for each cpt of $A_\nu$.
> *ie: wave eq for each component of $A_\nu$ with unique solutions given by specifying $A_\mu$ and its first time derivative on $\Sigma$. (unique sol. to initial value problem).*

**L15:**
*(This section contains handwritten notes, likely scratchpad work for an exercise or proof derivation related to the Riemann tensor and Ricci tensor in terms of Christoffel symbols and their derivatives.)*

1.  Start w/ expression for $R^\mu{}_{\nu\rho\sigma}$ in terms of connection components.
    $R^\mu{}_{\nu\rho\sigma} = \partial_\rho\Gamma^\mu_{\sigma\nu} - \partial_\sigma\Gamma^\mu_{\rho\nu} + \Gamma^\tau_{\rho\nu}\Gamma^\mu_{\sigma\tau} - \Gamma^\tau_{\sigma\nu}\Gamma^\mu_{\rho\tau}$
2.  lower $\mu$, remember g commutes w/ $\partial$ but not $\nabla$! so last 2 terms become $\Gamma \cdot \Gamma$
3.  but bit mad so let's look like $g_{\mu\kappa}\partial_\rho \Gamma^\kappa_{\sigma\nu}$ need to find this.
    $\partial_\rho(g_{\mu\kappa}\Gamma^\kappa_{\sigma\nu}) = g_{\mu\kappa}\partial_\rho\Gamma^\kappa_{\sigma\nu} + \Gamma^\kappa_{\sigma\nu}\partial_\rho g_{\mu\kappa}$. $\implies g_{\mu\kappa}\partial_\rho\Gamma^\kappa_{\sigma\nu} = \partial_\rho(\Gamma_{\mu\sigma\nu}) - \Gamma^\kappa_{\sigma\nu}\partial_\rho g_{\mu\kappa}$. I want to isolate $g_{\mu\kappa}\partial_\rho$ ... at this inclusion I use exper for $\Gamma$ in terms of g and low indices.
    $\Gamma_{\mu\sigma\nu} = \frac{1}{2}(g_{\mu\nu,\sigma} + g_{\sigma\mu,\nu} - g_{\sigma\nu,\mu})$.
4.  Plug this into our main eqn. Plug in all terms and simplifying gives $R_{\nu\rho\sigma}$
5.  get $R_{\rho\nu}$ from our expression see "Poke" above & plug that in
6.  notice that $\partial^\mu \Gamma_{\mu\nu\rho} = 0$?
7.  $\partial_\sigma g^{\mu\nu} = -g^{\mu\alpha}g^{\nu\beta}\partial_\sigma g_{\alpha\beta}$ to find $\partial_\sigma g^{\mu\kappa} = g^{\mu\nu}\partial_\sigma g_{\nu\kappa}$
8.  put everything together.

---


*Lecture 15*

*13.11.24*

**Recap:** Maxwell $\partial_\mu F^{\mu\nu}=0$, $\partial_{[\mu}F_{\nu\sigma]}=0$. Let $F_{\mu\nu}=\partial_{[\mu}A_{\nu]}$ then M1 $\implies \Box A_\nu - \partial_\nu\partial^\mu A_\mu = 0 \quad (*)$. But cannot solve this as good evolution problem. For example, $\tilde{A}_\mu = A_\mu + \partial_\mu \chi$ also solves (*) with some F and some initial data $A_\nu|_{t=0}, \partial_0 A_\nu|_{t=0}$ (provided $\text{supp}(\chi)\cap\{t=0\}=\emptyset$). To resolve, we fix a gauge via $\partial^\mu A_\mu=0$. Then $(*) \implies \Box A_\nu=0$ ($\dagger$) which has unique solution given $A_\nu|_{t=0}, \partial_0 A_\nu|_{t=0}$.

**CLAIM:** There is no non-trivial $\chi$ s.t. $\tilde{A}_\nu$ also satisfies $\partial^\mu \tilde{A}_\mu=0$ and the same initial condition. If we solve ($\dagger$) for data s.t. $\partial_\mu A^\mu|_{t=0}=0, \partial_0(\partial_\mu A^\mu)|_{t=0}$ then we find $\partial_\mu A^\mu=0$, and so our solution solves (*) and hence Maxwell's equations.

### Gauge Freedom for Einstein

If $(M,g)$ solves Einstein equations with F/M tensor T and $\psi:M \to M$ a diffeomorphism then $\psi^*g$ solves the Einstein equations with Fc/M tensor $\psi^*T$. At a local level, this arises as the coordinate indices of the Einstein equations.
> *energy momentum tensor*
> *coordinate independence invariance*

In order to solve EEs, we need to find a way to fix coordinates. There are several approaches, we consider **waver harmonic gauge**.
> *fixing harmonic coords turns EE into a well-posed evolution problem*

**LEMMA:** In any local coordinate system,
$R_{\rho\sigma\nu\mu} = \frac{1}{2}(g_{\rho\nu,\sigma\mu} + g_{\sigma\mu,\rho\nu} - g_{\rho\mu,\sigma\nu} - g_{\sigma\nu,\rho\mu}) - (\Gamma^\lambda_{\mu[\rho}\Gamma_{\sigma]\nu\lambda} + \Gamma^\lambda_{\nu[\rho}\Gamma_{\sigma]\mu\lambda})$
and $R_{\sigma\nu} = -\frac{1}{2}g^{\mu\rho}\partial_\mu\partial_\rho g_{\sigma\nu} + \frac{1}{2}\partial_\sigma\Gamma_\nu + \frac{1}{2}\partial_\nu\Gamma_\sigma + \frac{1}{2}(\Gamma^\lambda_{\sigma\mu}\Gamma^\mu

---


***

### Page 44 (PDF page 26)

> note: Even if $\nabla$ is torsion free, $\nabla_a \nabla_b X^c \neq \nabla_b \nabla_a X^c$ in general.

### THE LEVI-CIVITA CONNECTION

For a manifold with metric there is a preferred connection
> T.A.M. (Fundamental theorem of Riemannian geometry).

If $(M, g)$ is a manifold with a metric, there is a **unique** torsion free connection $\nabla$ satisfying $\nabla g = 0$. This is called the **levi-civita connection**.
> (e.g. $\nabla_X g = 0$)

**PROOF:** Suppose such a connection exists. By Leibniz rule, if $X, Y, Z$ are smooth vector fields.

$X(g(Y,Z)) = (\nabla_X g)(Y,Z) + g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$

* $X(g(Y,Z)) = g(\nabla_X Y, Z) + g(Y, \nabla_X Z)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a)
* $Y(g(Z,X)) = g(\nabla_Y Z, X) + g(Z, \nabla_Y X)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b)
* $Z(g(X,Y)) = g(\nabla_Z X, Y) + g(X, \nabla_Z Y)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c)
> since metric symmetric

* a) + b) - c):
$X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) = g(\nabla_X Y + \nabla_Y X, Z) + g(\nabla_X Z - \nabla_Z X, Y) + g(\nabla_Y Z - \nabla_Z Y, X)$

* Use $\nabla_X Y - \nabla_Y X = [X,Y]$
> this step helps be no dependence on $\nabla..$ on RHS
$X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) = 2g(\nabla_X Y, Z) - g([X,Y], Z) - g([Z,X], Y) + g([Y,Z], X)$

* > deters uniqueness Z is arbitrary & g is non degenerate
$\implies g(\nabla_X Y, Z) = \frac{1}{2}\{X(g(Y,Z)) + Y(g(Z,X)) - Z(g(X,Y)) + g([X,Y],Z) + g([Z,X],Y) - g([Y,Z],X)\}$
> i.e. non-degenerate means if g(v,z)=0 for all z, then v=0. In our case our 'v' is $(\nabla_X Y - \text{something})$. Since everything on the RHS are already known quantities, then our v is completely determined, there is no non-uniqueness. => $\nabla_X Y$ is unique.

This determines $\nabla_X Y$ uniquely since g is non-degenerate.
Conversely we can use $(\dagger)$ to define $\nabla_X Y$. Then need to check properties of a symmetric connection hold.

* E.g.
> show $\nabla_{gX}Y = g\nabla_X Y$
$g(\nabla_{gX}Y, Z) = \frac{1}{2}\{gX(g(Y,Z)) + Y(g(Z,gX)) - Z(g(gX,Y)) + g([gX,Y],Z) + g([Z,gX],Y) - g([Y,Z],gX)\}$
> use three linear arguments but not just replace X with gX ...
$= \frac{1}{2}\{gX(g(Y,Z)) + gY(g(Z,X)) - gZ(g(X,Y)) + (Yg)g(Z,X) - (Zg)g(X,Y) + g(g[X,Y]-Yg_X, Z) + g(Y[Z,X]+Zg_X,Y) - g(g[Y,Z],X)\}$
$\implies g(\nabla_{gX}Y) = g(g\nabla_X Y,Z) \implies g(\nabla_{gX}Y - g\nabla_X Y, Z) = 0. \quad \forall Z$
> "non-degenerate" means for any $V\neq 0$ there exists $Y$ s.t. $g(V,Y) \neq 0$.

* so $\nabla_{gX}Y = g\nabla_X Y$ as g non-degenerate.

**Exercise**: check other properties
> Use the expression ($\dagger$) we found

***

### Page 45 (PDF page 27)

In a coord. basis we can compute
$g(\nabla_{e_\mu} e_\nu, e_\sigma) = \frac{1}{2}\{e_\mu(g(e_\nu, e_\sigma)) + e_\nu(g(e_\sigma, e_\mu)) - e_\sigma(g(e_\mu, e_\nu))\}$
$g(\Gamma^\tau_{\mu\nu}e_\tau, e_\sigma) = \Gamma^\tau_{\mu\nu} g_{\tau\sigma} = \frac{1}{2}(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$
$\implies \Gamma^\tau_{\mu\nu} = \frac{1}{2}g^{\tau\sigma}(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu})$
> notes, we have shown the Christoffel symbols are the components of the Levi-Civita connection in a coordinate basis.

$\uparrow$ These are **CHRISTOFFEL SYMBOLS**
> important: learn!

If $\nabla$ is Levi-Civita can raise/lower indices and this commutes with covariant differentiation.
> If $\nabla$ is Levi-Civita then $g^{ab}\nabla_c X^b = \nabla_c(g^{ab}X^b) = \nabla_c X_b)$

#### GEODESICS

We found that a curve extremizing proper time satisfies
> if proper time along curve

($\dagger$) $\frac{d^2x^\mu}{dt^2} + \Gamma^\mu_{\nu\sigma}(x(t))\frac{dx^\nu}{dt}\frac{dx^\sigma}{dt} = 0$

The tangent vector $X^a$ to the curve has components $X^\mu = \frac{dx^\mu}{dt}$.
Extending this off the curve we get a vector field, of which the geodesic is an integral curve. We note
$\frac{d X^\mu}{dt} = \frac{d}{dt}\left(\frac{dx^\mu}{dt}\right) = \frac{\partial X^\mu}{\partial x^\nu}\frac{dx^\nu}{dt} = X^\nu \partial_\nu X^\mu \quad (\text{chain rule})$

($\dagger$) becomes $X^\nu \partial_\nu X^\mu + \Gamma^\mu_{\nu\sigma} X^\nu X^\sigma = 0 \iff X^\nu \nabla_\nu X^\mu = 0 \iff \nabla_X X = 0$.

Extend to any connection.
> where we are using the Levi-Civita connection. now extend.

**DEF:** Let M be a manifold with connection $\nabla$. An **AFFINELY PARAMETRIZED GEODESIC** satisfies $\nabla_X X = 0$ where $X$ is the tangent vector.
> tangent vector to curve defined only on curve itself

(Lecture 10) **note:** if we reparametrise $t \to t(u)$ then
$\frac{dx^\mu}{dt} = \frac{dx^\mu}{du}\frac{du}{dt}$
so $X = Y h$ with $h > 0$
$\nabla_Y Y = \nabla_{hX}(hX) = h\nabla_X(hX) = h^2\nabla_X X + hX(h) = \nabla_Y X(h)$
with $g=X(h) = \frac{d}{dt}(h) = \frac{dh}{du}\frac{du}{dt} = \frac{1}{h}\frac{dh}{du^2}$ , so $\nabla_Y Y=0 \iff t=au+\beta, \quad \alpha,\beta \in \mathbb{R}, \alpha>0$
> what is meant by affine reparametrize: $\nabla_Y Y=X(h)Y=0$. but in general not affinely parametrized. but it is always possible to find a parameter s.t. it is. in this case $\nabla_Y Y=0$. $u=at+b$. wlog we can always restrict to APGs.

***

### Page 46 (PDF page 28)

> Exercise: let X be tangent to an APG of the Levi-Civita connection. show that $\nabla_X(g(X,X)) = 0$.
> attempt: $\nabla_X(g(X,X)) = (\nabla_X g)(X,X) + g(\nabla_X X, X) + g(X, \nabla_X X) = 2g(0,X)=0$
> So the tangent vector has constant change along a geodesic if it is timelike or null along the geodesic, a geodesic is timelike, spacelike or null.

#### Lecture 10

**Theorem:** given $p \in M, X_p \in T_pM$, there exists a unique A.P.G. $\lambda: I \to M$ satisfying $\lambda(0)=p, \dot{\lambda}(0)=X_p$.

**PROOF:** Choose coordinates with $\phi(p)=0$.
satisfies $\nabla_X X=0$ with $X=X^\mu\frac{\partial}{\partial x^\mu}$.
this becomes
$\frac{d^2x^\mu}{dt^2} + \Gamma^\mu_{\nu\sigma}\frac{dx^\nu}{dt}\frac{dx^\sigma}{dt} = 0 \quad (\text{GE})$
and $x^\mu(0)=0, \frac{dx^\mu}{dt}(0) = X_p^\mu$
> 2nd order ODE with two BCs -> uniqueness follows from standard ODE theory.

This has a unique solution $x^\mu: (-\epsilon, \epsilon) \to \mathbb{R}^n$ for $\epsilon$ sufficiently small by standard ODE theory. $\square$

#### GEODESIC POSTULATE

> not acted on by any force except gravity

In general relativity free particles move along geodesics of the Levi-Civita connection.
These are **TIMELIKE** for massive particles and **NULL/LIGHTLIKE** for massless particles.

#### Normal Coordinates
> "exponential map".

If we fix $p \in M$ we can map $T_pM$ into $M$ by setting $\psi(X_p) = \lambda_{X_p}(1)$ where $\lambda_{X_p}$ is the unique affinely parametrised geodesic with $\lambda_{X_p}(0)=p, \dot{\lambda}_{X_p}(0)=X_p$.
> "breaks down long distances"
> "local 'rectifying' points on my nfold"
> "if we give a direction & distance to get there"
> "back to waitress, but to reach there are many paths" -> uniqueness problem.

Notice that
$\lambda_{aX_p}(t) = \lambda_{X_p}(at)$ for $a \in \mathbb{R}$.
since if $\tilde{\lambda}(t) = \lambda_{aX_p}(t)$ affine reparametrisation. so still geodesic, and $\dot{\tilde{\lambda}}(0) = a\dot{\lambda}_{X_p}(0) = aX_p, \tilde{\lambda}(0)=p$.
> since if $\lambda$ is affine param with u=at+b $\tilde{u}$ is also affine param for a,b in R i think.
> Luru: the map that sends $X_p$ to 1 also sends $aX_p$ to $a$.

Moreover, $t \mapsto \psi(tX_p)$ is an affinely parametrised geodesic $= \lambda_{X_p}(t)$.
> notetosely:
> * like throwing a ball and seeing where it lands. if i throw it w/ initial velocity $2X_p$ it goes twice as far in unit time.
> * the exponential map sends $X_p$ to the point unit distance along the geodesic through p tangent to $X_p$ at p. But it sends $tX_p$ distance t along that same geodesic.

***

### Page 47 (PDF page 29)

**CLAIM:** If $U \subset T_pM$ is a sufficiently small neighbourhood of the origin, then $\psi|_U: T_pM \to M$ is one-to-one and onto.

**DEF:** Construct **normal coordinates** at p.
suppose $\{e_\mu\}$ is a basis for $T_pM$, as follows. For $q \in U(0) \subset M$, we define $\phi(q) = (x^1, ..., x^m)$ where $x^\mu$ are components of the unique $X_p \in U$ with $\psi(X_p)=q$. ($q = \sum x^\mu e_\mu$)
> q = $\lambda_{X_p}(1)$ (discrete)
> now re-use geodesic $\lambda_{X_p}(t)$
> Its coords are
> $x^\mu(t) = tx^\mu$ (continuous map)
> i.e. we don't just want to argue given in normal coords by $x^\mu(t)=ty^\mu$ for $y^\mu$ constant.

By our previous observation, the curve given in normal coordinates by $x^\mu(t) = ty^\mu$ for $y^\mu$ constant is an affinely parametrised geodesic. so from the **geodesic eqn.**
$\Gamma^\sigma_{\nu\sigma}(ty)y^\nu y^\sigma = 0$
Set $t=0$ deduce (since $Y$ arbitrary) that $\Gamma^\sigma_{\nu\sigma}|_p = 0$.
> components vanish at p in normal coordinates.

So if $\nabla$ is torsion free, $T^\sigma_{\mu\rho}|_p=0$.
If $\nabla$ is the Levi-Civita connection of a metric, then
> use $g_{\mu\nu,\rho} = g_{\sigma\nu}\Gamma^\sigma_{\mu\rho} + g_{\mu\sigma}\Gamma^\sigma_{\nu\rho}$
> (At p, normal coords are coords where the spacetime looks like Minkowski space (is flat at a point) to 1st order)
Since $g_{\mu\nu,\rho} = \frac{1}{2}(g_{\mu\nu,\rho} + g_{\rho\nu,\mu} - g_{\mu\rho,\nu}) + \frac{1}{2}(g_{\mu\rho,\nu} + g_{\nu\rho,\mu} - g_{\mu\nu,\rho})$
$= \Gamma_{\mu\nu\rho} + \Gamma_{\rho\mu\nu} = g_{\sigma\rho}(\Gamma^\sigma_{\mu\nu} + \Gamma^\sigma_{\nu\mu})$
$\partial_\rho g_{\mu\nu}|_p = 0$
> at p.

We can always choose the basis $\{e_\mu\}$ for $T_pM$ on which base the normal coordinates to be orthonormal. We have

**LEMMA:** On a Riemannian/Lorentzian manifold we can choose normal coordinates at p s.t. $g_{\mu\nu,\rho}|_p=0$ and $g_{\mu\nu}|_p = \begin{cases} \delta_{\mu\nu} & \text{RIEMANNIAN} \\ \eta_{\mu\nu} & \text{LORENTZIAN} \end{cases}$
> 1st derivative vanish at p.
> (i.e. locally looks flat?)
> (Euclidian/flat spacetime equivalence principle, differeomorphism)

**PROOF:** The curve given in normal coordinates by $t \mapsto (t,0,...,0)$ is the APG with $\dot{\lambda}(0)=p, \dot{\lambda}(0)=e_1$ by previous argument. But by defn. of coord basis this vector is $(\frac{\partial}{\partial x^1})|_p$. So $\{e_\mu\}$ is ON at p ($\Leftrightarrow g_{\mu\nu}$) form an ON basis. $\square$
> idea: if we pick the initial basis $\{e_\mu\}$ to be orthonormal then the geodesics will point in orthogonal directions which means the metric looks like $g_{\mu\nu}|_p = \delta_{\mu\nu}$

***

### Page 48 (PDF page 30)

### CURVATURE

> look for degree intrinsic to manifold that tells us its not flat. (do this by considering parallel transport).
> Sphere: transporting a vector around changes the angle, tells us it is curved.

#### Parallel transport

> notetosely if $\lambda$ is a curve with tangent vector $X^a$, then a tensor field T is parallely transported along $\lambda$ then $\nabla_X T = 0$

Suppose $\lambda:I\to M$ is a curve with tangent vector $\dot{\lambda}(t)$.
If we say a tensor field T is parallely transported/propagated along $\lambda$.
$\nabla_{\dot{\lambda}}T = 0$ on $\lambda \quad (\text{PP})$
> looks a bit like geodesic eqn (GE)

* If $\lambda$ is an APG then $\dot{\lambda}$ is parallely propagated along $\lambda$.
> a geodesic is a tangent vector that is parallely transported along the curve

* A parallely propagated tensor is determined everywhere on $\lambda$ by its value at one point.

* **Eg.** If $T$ is a (1,1) tensor then in coordinates (PP) becomes
$0 = \frac{dx^\mu}{dt}\nabla_\mu T^\sigma_\rho = \frac{dx^\mu}{dt}(\partial_\mu T^\sigma_\rho + \Gamma^\sigma_{\mu\nu}T^\nu_\rho - \Gamma^\nu_{\mu\rho}T^\sigma_\nu)$
but $T^\sigma_\rho \frac{dx^\mu}{dt} = \frac{d}{dt}(T^\sigma_\rho)$ so
$0 = \frac{d}{dt}T^\sigma_\rho + (\Gamma^\sigma_{\mu\nu}T^\nu_\rho - \Gamma^\nu_{\mu\rho}T^\sigma_\nu)\frac{dx^\mu}{dt}$
> linear in T
> 1st order eqn -> solution determined entirely by values at a point

This is a 1st order linear ODE for $T^\sigma_\rho(\lambda(t))$, so ODE theory gives a unique soln. once $T^\sigma_\rho(\lambda(0))$ specified.
> T uniquely determined if know its value at one point

* Parallel transport along a curve from $p$ to $q$ gives an **isomorphism** between tensors at $p$ and $q$. This depends on the choice of curve in general.
> important
> * the isomorphism depends on the choice of path
> * on a curved manifold, parallel transporting around a loop may not return you to the same tensor
> * isomorphism means the map is invertible & preserves the tensor structure i.e. maps Tensors to Tensors.
> * iso: a structure preserving map that can be reversed by an inverse mapping

***

### Page 49 (PDF page 31)

#### Lecture 11
### THE RIEMANN TENSOR
4.11.24

The Riemann tensor captures the extent to which parallel transport depends on the curve.

**LEMMA:** Given $X,Y,Z$ vector fields, $\nabla$ a connection, define
$R(X,Y)Z = \nabla_X\nabla_Y Z - \nabla_Y\nabla_X Z - \nabla_{[X,Y]}Z$
Then $(R(X,Y)Z)^a = R^a_{bcd} X^c Y^d Z^b$ for a (1,3)-tensor $R^a_{bcd}$, the **Riemann tensor**.

**PROOF:** Suppose $f$ is smooth function, then
> prove it is a tensor by showing it is linear in X,Y,Z
$R(X,Y)(fZ) = \nabla_X\nabla_Y fZ - \nabla_Y\nabla_X fZ - \nabla_{[X,Y]}fZ$
$= \nabla_X(f\nabla_Y Z) - \nabla_Y(f\nabla_X Z) - f\nabla_{[X,Y]}Z$
$= f\nabla_X\nabla_Y Z - f\nabla_Y\nabla_X Z - Y(f)\nabla_X Z - f\nabla_{[X,Y]}Z + Y(f)\nabla_{[X,Y]}Z$
$= fR(X,Y)Z$
> linear in Z

Since $R(X,Y)Z = -R(Y,X)Z$, we have $R(X,fY)Z = fR(X,Y)Z$
> linear in Z

**Exerase** check $R(X,Y)(fZ) = fR(X,Y)Z$
> since components are linear in Z above, we have 3 results

Now suppose we pick a basis $\{e_\mu\}$ with dual basis $\{\xi^\mu\}$
$R(X,Y)Z = R(X^\rho e_\rho, Y^\sigma e_\sigma)(Z^\nu e_\nu) = X^\rho Y^\sigma Z^\nu R(e_\rho, e_\sigma)e_\nu$
$= R^\mu_{\nu\rho\sigma} X^\rho Y^\sigma Z^\nu e_\mu$
> holds in one basis => holds in any basis

where $R^\mu_{\nu\rho\sigma} = \xi^\mu(R(e_\rho, e_\sigma)e_\nu)$ are components.
$R^\mu_{\nu\rho\sigma}$ in this basis. Since result holds in one basis, it holds in all bases. $\square$

In a coordinate basis $e_\mu = \frac{\partial}{\partial x^\mu}$ and $[e_\mu, e_\nu]=0$ so
$R(e_\rho, e_\sigma)e_\nu = \nabla_\rho(\nabla_\sigma e_\nu) - \nabla_\sigma(\nabla_\rho e_\nu) = \nabla_\rho(\Gamma^\tau_{\nu\sigma} e_\tau) - \nabla_\sigma(\Gamma^\tau_{\nu\rho} e_\tau)$
$= \partial_\rho(\Gamma^\tau_{\nu\sigma})e_\tau + \Gamma^\tau_{\nu\sigma}\Gamma^\mu_{\tau\rho}e_\mu - \partial_\sigma(\Gamma^\tau_{\nu\rho})e_\tau - \Gamma^\tau_{\nu\rho}\Gamma^\mu_{\tau\sigma}e_\mu$
Hence $R^\mu_{\nu\rho\sigma} = \partial_\rho(\Gamma^\mu_{\nu\sigma}) - \partial_\sigma(\Gamma^\mu_{\nu\rho}) + \Gamma^\tau_{\nu\sigma}\Gamma^\mu_{\tau\rho} - \Gamma^\tau_{\nu\rho}\Gamma^\mu_{\tau\sigma}$

***

### Page 50 (PDF page 32)

In **normal coordinates** we can drop the last two terms.
> (if we also have $\partial_c\Gamma=0$ not just $\Gamma=0$, then $R=0$)
> at a point p in normal coords, $\Gamma=0$, but $\partial\Gamma \neq 0$. The last two terms vanish at p.

**Example:** For the **Levi-Civita connection** of **Minkowski space** in an **inertial frame**, $\Gamma^\mu_{\nu\sigma}=0$, so $R^\mu_{\nu\sigma\tau}=0$.
hence $R^a_{bcd}=0$. Such spacetime is called flat. Conversely, for a **flat** L-C connection, we can locally find coordinates such that $g_{\mu\nu} = \text{diag}(-1,1,1,1)$.
> the spacetime is "locally isometric" to Minkowski.

**A note of caution:**
$(\nabla_X \nabla_Y Z)^c = X^a\nabla_a(Y^b\nabla_b Z^c) \neq X^a Y^b \nabla_a \nabla_b Z^c$

hence $(R(X,Y)Z)^c = X^a \nabla_a (Y^b \nabla_b Z^c) - Y^a \nabla_a(X^b \nabla_b Z^c) - [X,Y]^b \nabla_b Z^c$
$= X^a Y^b \nabla_a \nabla_b Z^c - Y^a X^b \nabla_b \nabla_a Z^c + (\nabla_X Y - \nabla_Y X - [X,Y])^b\nabla_b Z^c$

So if $\nabla$ is torsion free,
$\nabla_a \nabla_b Z^c - \nabla_b \nabla_a Z^c = R^c_{dab} Z^d \quad$ **RICCI IDENTITY**

on ex. sheet 2 there's a question to generalise for an expression for $\nabla_{c_a}...\nabla_{c_n} T^{d_1...d_s}_{b_1...b_r}$.

We can construct a new tensor from $R^a_{bcd}$ by **contraction**:
> (notetosely: $R^a_{acd} = R^c_{acd}$ => $R^a_{[acd]}=0$)

**Definition:** The **RICCI TENSOR** is the (0,2)-tensor
$R_{ab} = R^c_{acb}$
> why not contract diff 2 indices?
> $R^a_{acd}=0$; others are the same up to a minus sign. so this is the only one of interest.

Suppose $X,Y$ are vector fields satisfying $[X,Y]=0$.

[Diagram shows a vector $Z$ at point A. It's parallel transported along a path A->B (along X), then B->C (along Y), then C->D (along -X), then D->A (along -Y). The final vector at A is different from the initial vector Z. The diagram also shows an infinitesimal version with vectors $\epsilon X, \epsilon Y$ forming a parallelogram.]

---


***


$h_{\mu\nu} = \text{Re}(H_{\mu\nu} e^{ik_\alpha x^\alpha}) = |H_+| \begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix} \cos(\omega(t-\xi)) - \xi_0$
> [Side note top left] modulus / means / can be complex
>
> [Side note top right] since sin complex
>
> [Side note top right] phase that comes from it being complex

Along $\lambda$, $\xi=0$ equations are:

$\frac{d^2y^0}{dt^2} = \frac{d^2y^3}{dt^2} = 0$ (no relative acceleration in direction of the wave)
> [Side note left, on `d^2y^0/dt^2=0`] (since $h_{0\nu}=0$)
>
> [Side note right] (wave moves in this direction)

$\frac{d^2y^1}{dt^2} = -\frac{1}{2}\omega^2|H_+|\cos(\omega(t-t_0))Y^1$
> [Side note left] cannot solve explicitly linear eqn with periodic coefficient (Mathieu eqn)

$\frac{d^2y^2}{dt^2} = \frac{1}{2}\omega^2|H_+|\cos(\omega(t-t_0))Y^2$
> [Side note left] $(t-t_0)$

$H_+$ is small, so solve perturbatively with $\frac{dy^1}{dt} = \frac{dy^2}{dt} = 0 \implies Y^1=Y_0^1+O(\epsilon^2) \quad Y^2=Y_0^2+O(\epsilon^2)$
> [Side note right] put back into

$Y^1 = Y_0^1(1+\frac{1}{2}|H_+|\cos(\omega(t-t_0)))$
> [Side note] initially consists of test masses on a circle / we've collection of test masses on a circle / as time passes this is going to squash in $Y^2$ direct. & stretch in $Y^1$ direct. then

$Y^2 = Y_0^2(1-\frac{1}{2}|H_+|\cos(\omega(t-t_0)))$
> [Side note] stretching along one / oscillating

> [Side note right of equations] note that any displacement is $\propto Y_0^1, Y_0^2$ respectively. If we arrange points in 1-2 plane with $(Y_0^1, Y_0^2) = R(\cos\phi, \sin\phi)$ i.e. when $\omega(t-t_0)=\pi/2$ they will form a circle. At $\omega(t-t_0)=\pi$ this stretch in 1-direction & squash in 2-direction etc.

[Diagram showing a circle of test particles being distorted by a gravitational wave.
1. A circle at $\omega(t-t_0) = \pi/2$.
2. An ellipse stretched horizontally and compressed vertically at $\omega(t-t_0) = \pi$.
3. A circle at $\omega(t-t_0) = 3\pi/2$.
4. An ellipse stretched vertically and compressed horizontally at $\omega(t-t_0) = 2\pi$.]

**Exercise:** Find solution for $\times$ polarisation.

> [Side note left] note that solving perturbatively
>
> zero order solution (no wave) $d^2Y/dt^2=0, Y=Y_0$
>
> get 1st order correction: $d^2 Y^1/dt^2 = \frac{1}{2}\omega^2|H_x|\cos(\omega(t-t_0))Y_0^2$
> so put higher order $Y_0^1$ into RHS
> $Y^1 = Y_0^1(1+\frac{1}{2}|H_x|\cos(\omega(t-t_0)))Y_0^2$ (Mistake in side note, correction is additive)

*   attempt: solve $d^2Y^j/dt^2 = ...$ perturbatively
    $Y^1 = Y_0^1 + \frac{1}{2}|H_\times|\cos(\omega(t-t_0))Y_0^2$
    $Y^2 = Y_0^2 + \frac{1}{2}|H_\times|\cos(\omega(t-t_0))Y_0^1$

[Diagram showing a circle of test particles being distorted by x-polarization.
1. A circle at $\omega(t-t_0)=\pi/2$.
2. An ellipse stretched along the y=x axis and compressed along y=-x at $\omega(t-t_0)=\pi$.
3. A circle at $\omega(t-t_0)=3\pi/2$.
4. An ellipse stretched along y=-x axis and compressed along y=x at $\omega(t-t_0)=2\pi$.]

same sol. rotated by 45°

---
51

***


## Lecture 18
20.11.24

**Last lecture:** monochromatic plane gravitational waves.
*   effect on test particles
    > [Side note] particles freely falling, wave travels into the page
    [Diagram of `+` polarisation effect on a circle of particles, followed by `x` polarisation effect.] `(+ POLARISATION)`
    > [Side note] Found using geodesic deviation.

### THE FIELD FAR FROM A SOURCE

Return to linearised Einstein equations with matter.
> [Side note left] think of this like E&M equations in Lorenz gauge. $g_{\mu\nu}$ potential satisfies wave equation. $\partial_\mu \bar{h}^{\mu\nu}=0$ is 4-divergence free. (Lorenz-Fischer potential) useful reg.

$(*) \ \partial_\rho\partial^\rho \bar{h}_{\mu\nu} = -16\pi T_{\mu\nu}$
$\partial_\mu \bar{h}^{\mu\nu}=0$
> [Side note right] wave, gauge condition

As is the case for electromagnetism, we can solve (*) explicitly using a retarded Green's function:
> [Side note left] as is standard in Jackson, use t, x, z components
>
> [Side note right] modulus computed in euclidean space is practically as we normally think of it

$\bar{h}_{\mu\nu}(t, \mathbf{x}) = 4 \int d^3x' \frac{T_{\mu\nu}(t - |\mathbf{x}-\mathbf{x'}|, \mathbf{x'})}{|\mathbf{x}-\mathbf{x'}|} \quad (\#)$

Where $|\mathbf{x}-\mathbf{x'}|$ is computed in the Euclidean metric.
> [Side note left] (from source far away)

If matter is concentrated within a distance $d$ of the origin (so $T_{\mu\nu}=0$ for $|\mathbf{x'}|>d$), we can expand in the far field region where $r=|\mathbf{x}| \gg |\mathbf{x'}| \approx d$. Then
> [Side note in diagram] interested in field at point. look over past lightcone of pt. (that is where info comes from). integrate over past lightcone with some jacobian factor. tells me how much source from one side to go from another.

[Diagram showing a point `(t, x)` and its past lightcone intersecting with the source region at earlier times.]

$|\mathbf{x}-\mathbf{x'}|^2 = r^2 - 2\mathbf{x}\cdot\mathbf{x'} + |\mathbf{x'}|^2 = r^2(1 - 2\frac{\mathbf{\hat{x}}\cdot\mathbf{x'}}{r}) + O(\frac{d^2}{r^2})$

with $\mathbf{\hat{x}}=\mathbf{x}/r$. Hence, (use binomial expansion):
> [Side note] unit vector
>
> [Side note right] correction is inner product (i.e. vector pointing from origin to source & pos'n vector) so this is where $\hat{x}$ gets into formula

$|\mathbf{x}-\mathbf{x'}| = r(1-\frac{2\mathbf{\hat{x}}\cdot\mathbf{x'}}{r} + O(\frac{d^2}{r^2}))^{1/2} = r - \mathbf{\hat{x}}\cdot\mathbf{x'} + O(\frac{d^2}{r})$
> [Side note] next we Taylor expand so distance over large time we neglect

and $T_{\mu\nu}(t-|\mathbf{x}-\mathbf{x'}|, \mathbf{x'}) = T_{\mu\nu}(t', \mathbf{x'}) + \mathbf{\hat{x}}\cdot\mathbf{x'} \partial_0 T_{\mu\nu}(t', \mathbf{x'}) + \dots$
where $t'=t-r$.
> [Side note] relativistic velocity?
>
> [Side note right] what do I mean by this? above timescale tells me which matter contributes to field at some time.

If $T_{\mu\nu}$ varies on a timescale of $\tau$ so that $\partial_0 T_{\mu\nu} \approx \frac{1}{\tau} T_{\mu\nu}$, then second term above is $O(d/\tau)$ which we can neglect if matter moves non-relativistically. We thus have,
> [Side note left] want to hear from first term in series, in integral. on $T_{ij}$ at the retarded time. and add up contributions to get $\bar{h}_{ij}$

$\bar{h}_{ij} = \frac{4}{r} \int d^3x' T_{ij}(t', \mathbf{x'}) \quad t'=t-r \quad (**)$

This gives the spatial components of $\bar{h}_{ij}$. To find remaining components we use gauge condition: $\partial_0 \bar{h}_{0i} = \partial_j \bar{h}_{ji}, \quad \partial_0 \bar{h}_{00} = \partial_i \bar{h}_{0i}$.
> [Side note bottom] from $\partial_\mu h^{\mu\nu}=0$, the $\nu=0$ component

---
52

***


First solve for $\bar{h}_{0i}$ then $\bar{h}_{00}$.
> [Side note] note $\partial_\mu T^{\mu\nu}=0$

We can simplify the integral in ($**$) by recalling that $\partial_\mu T^{\mu\nu}=0$ (EM conservation) and that $T_{\mu\nu}(t', \mathbf{x'})$ vanishes for $|\mathbf{x'}|>d$.

Dropping primes in the integral
> [Side note top right] can't have this but student has written it so does not mean anything
> [Side note top right] should be clear about what this means. write by parts using strict divergence vanishes. Integral of $\partial_k T_{ik}$...
>
> [Side note below] think $T^{ij} \approx \rho v^i v^j$. $T^{0j} \approx \rho v^j$.
>
> [Side note below] $\partial_i T^{ij} \approx \rho (\partial_i v^j) v^i$

$\int d^3x T_{ij}(t, \mathbf{x}) = \int d^3x \partial_k(T_{ik}x_j) - (\partial_k T_{ik})x_j$
> [Side note right] subtract what happens when deriv hits the other

= $\int_{|\mathbf{x}|=d} T_{ik}x_j n^k dS - \int d^3x (\partial_k T_{ik})x_j$
> [Side note left] divergence theorem. write as int. over a surface.
>
> [Side note right] surface int. is on sphere, but vanishes by assumption $T_{ij}=0, |\mathbf{x}|>d$.
>
> [Side note below] relate spatial divergence of $T$ to a time derivative.

LHS symmetric in $ij$ so replace RHS by symmetric part.
> [Side note middle] $0$ as $T_{ij}=0$ for $|\mathbf{x}|>d$

= $\int d^3x \partial_0 T_{0i} x_j = \frac{d}{dt} \int d^3x T_{0i} x_j$
> [Side note right] $(\partial_0 T^{0j} + \partial_i T^{ij} = 0)$

Symmetrising on $ij$

$\int d^3x T_{ij}(t, \mathbf{x}) = \frac{1}{2} \int d^3x[\partial_k(T_{0k}x_i)x_j + \frac{1}{2} T_{0k}\partial_k(x_i x_j)]$
> [Side note right] realise again this looks like something linear in a momentum.
>
> [Side note below] $=-\partial_0 T_{00}$

= $2 \int d^3x (\frac{1}{2} \partial_k(T^{0k}x_ix_j) - \frac{1}{2}T^{0k}\partial_k(x_ix_j))$
> [Side note] Again notice connection to EM momentum. as spatial divergence of energy-momentum, I can replace w/ time derivative of $T_{00}$ since $(\partial_0 T^{00} + \partial_k T^{0k}=0)$.
> [Side note] find time deriv of $I_{ij}$ at home.

> [Side note left] again pull out time derivatives this is integral over spatial derivs.

= $2 \int_V d^3x (\frac{1}{2} \partial_0 T^{00} x_i x_j)$ as above by divergence theorem
= $\frac{1}{2}\partial_0 \partial_0 \int d^3x T^{00}x^ix^j = \frac{1}{2}\ddot{I}_{ij}(t)$

Where $I_{ij}(t) = \int d^3x T^{00}(t, \mathbf{x})x^ix^j$.
> [Side note] 2nd moment of 00 component of EM tensor (mass energy) evaluated at time t.

Noting $T_{00} = T^{00}$ and $T_{ij}=T^{ij}$ we deduce,
$\bar{h}_{ij} = \frac{2}{r} \ddot{I}_{ij}(t-r) \quad \begin{matrix} r \gg d \\ \lambda \gg d \end{matrix}$
> [Side note right] (retarded time means light rays must have time to leave dist'n to arrive at me now. then dilute by $1/r$ to reflect spread over a sphere)

Now reconstruct remaining components using gauge condition.
> [Side note left] they are overdense indices so don't bring too much about indices vs coordinates

$\partial_0 \bar{h}_{0i} = \partial_j \bar{h}_{ji} = \partial_j (\frac{2}{r} \ddot{I}_{ij}(t-r))$
> [Side note right] Two free terms that contribute roughly
>
> [Side note middle] evaluated at t-r
>
> [Side note right] const of integration depend only on x

$\implies \bar{h}_{0i} = \frac{2}{r} \frac{x^j}{r} \ddot{I}_{ij}(t-r) - \frac{2}{r} \hat{x}^j \dddot{I}_{ij}(t-r) + k_i(\mathbf{x})$
> [Side note left] using chain rule and this fact

> [Side note right] $\partial_i(1/r) = - \hat{x}_i/r^2$. Hopefully, I've done this vector calculus right.

We now assume $r \gg \lambda$ so we are in radiation zone so can drop first term (it's $O(\lambda/r)$ relative to second) and get
> [Side note left] to term want to get rid of
>
> [Side note below] $\bar{h}_{00}$

$\bar{h}_{0i} = - \frac{2}{r} \hat{x}^j \dddot{I}_{ij}(t-r) + k_i(\mathbf{x})$

Now use,

---
53

***


$\partial_0 \bar{h}_{00} = \partial_i \bar{h}_{0i} = \partial_i(-\frac{2}{r} \hat{x}^j \dddot{I}_{ij}(t-r) + k_i(\mathbf{x}))$
> [Side note left] integrate wrt t
>
> [Side note right] ex. divergence of k

$\implies \bar{h}_{00} = -2\partial_i (\frac{x^j}{r^2}\ddot{I}_{ij}(t-r)) + t\partial_i k_i(\mathbf{x}) + g(\mathbf{x})$
> [Side note right] Const of integration f(x,t) or only g(x)

= $\frac{2\hat{x}^i \hat{x}^j}{r} \ddddot{I}_{ij}(t-r) + t\partial_i k_i + g(\mathbf{x}) +$ TERMS SUBLEADING IN $1/r$

To fix constants of integration, return to
$\bar{h}_{\mu\nu} = 4 \int d^3x' \frac{T_{\mu\nu}(t-|\mathbf{x}-\mathbf{x'}|, \mathbf{x'})}{|\mathbf{x}-\mathbf{x'}|}$
and observe that to leading order in $1/r$
> [Side note left] "ask me on the street I wouldn't remember this yourselves"

$\bar{h}_{00} = \frac{4E}{r}, \quad \bar{h}_{0i} = -\frac{4P_i}{r}$
> [Side note right] same expansion as we did before for $\bar{h}_{ij}$, but with $T_{00}/T_{0i}$? we can replace $\int d^3x T...$ with total E/P at retarded time

where $E=\int d^3x' T_{00}(t', \mathbf{x'}), \ P_i = \int d^3x' T_{0i}(t', \mathbf{x'})$
> [Side note] E, Pi actually const. in time although in principle they depend on retarded time

Observing that $\partial_0 \int d^3x' T_{0\mu}(t', \mathbf{x'}) = \int d^3x' \partial_0 T_{0\mu}(t', \mathbf{x'}) = \int d^3x' \partial_i T_{i\mu}(t', \mathbf{x'})$

So, $E, P_i$ constant in time.

**Exercise:** By a gauge transformation generated by a multiple of $\xi^\mu = (P \cdot x, -Pt)$ we can set $P=0$. This is the centre of momentum frame.
> [Side note left] we have chosen Minkowski gauge ... center of mass sits at origin

We've shown that:
> [Side note right] what do these terms tell us? perturbation depends only on 2nd moment of inertia

$* \ \bar{h}_{00}(t, \mathbf{x}) = \frac{4M}{r} + \frac{2\hat{x}^i\hat{x}^j}{r} \ddot{I}_{ij}(t-r)$
> [Side note right] formulae to copy correctly, they matter, e.g. binary pulsar system, bring around ... for long enough

$* \ \bar{h}_{0i}(t, \mathbf{x}) = -\frac{2\hat{x}^j}{r} \dddot{I}_{ij}(t-r)$
> [Side note right] time varying 2nd moment gives energy

$* \ \bar{h}_{ij} = \frac{2}{r} \ddot{I}_{ij}(t-r)$

where $r \gg \lambda \gg d$, in centre of momentum frame, where $E=M$.
> [Side note left] far away so in radiation zone, from non-rel. sources
>
> [Side note bottom] next time: assign energy to these perturbat so I can understand how waves that are sourced by time varying quadrupole can actually carry energy away from the system (today's computation)

---
54

***


## Lecture 19
22.11.24

**Last Lecture**
*   Linearised field for from a non-relativistic source:
    $\bar{h}_{00} = \frac{4M}{r} + \frac{2\hat{x}^i\hat{x}^j}{r}\ddot{I}_{ij}(t-r)$
    where $I_{ij}(t) = \int d^3x T_{00}(t;\mathbf{x})x^ix^j$
    $\bar{h}_{0i}(t, \mathbf{x}) = -\frac{2\hat{x}^j}{r}\dddot{I}_{ij}(t-r)$
    $r=|\mathbf{x}|$
    $\bar{h}_{ij}(t, \mathbf{x}) = \frac{2}{r}\ddot{I}_{ij}(t-r)$
    $M = \int d^3x T_{00}(t, \mathbf{x'})$
    and we are in centre-of-momentum frame
    $P_i = \int d^3x T_{0i}(t, \mathbf{x'}) = 0$
*   valid in radiation zone:
    distance from source $r \gg \lambda \gg d$ spatial extent of source
    > [Side note left] timescale of motion of source
    >
    > [Side note right] what source was doing when our past light cone intersected it?

[Diagram of a source emitting waves, with `(t, x)` observing them at a later time. The path of the wave is shown.]
> [Side note on diagram] These waves carry away energy from system which will get it e.g. rotating. How do waves carry energy away? what do we mean by energy? hard to assign energy to grav. field because it is geometry. And usually we define energy via Noether's theorem but we can choose coords s.t. metric is flat locally so energy vanishes. hard to define energy density in general relativity.

### ENERGY IN GRAVITATIONAL WAVES

Defining the local energy/local energy flux for a gravitational field is hard in general because we can always choose coords s.t. $\partial_\mu g_{\alpha\beta}|_p=0$.
> [Side note right] also often want to apply energy via Noether's theorem, involving usually quantity conserved due to time symmetry but in general background we don't have time translation symmetry

There is no hope of an energy density quadratic in first derivatives.
> [Side note] why is it hard to define energy?

> [Side note below] prob of def. energy is hard, but we can get round this by working in our perturbation theory.

In the context of perturbation theory there are various ways to define an energy. To do this we consider how to continue a perturbative solution beyond linear order.

We consider the ungauged vaccum Einstein equations and suppose $g_{\mu\nu} = \eta_{\mu\nu} + \epsilon h^{(1)}_{\mu\nu} + \epsilon^2 h^{(2)}_{\mu\nu}$.
> [Side note] (1) indicates this is linear piece, (2) correction
Work to $O(\epsilon^2)$.

We observe
> [Side note] linear Ricci tensor is function of the metric around minkowski space time

$R_{\mu\nu}[\eta_{\mu\nu} + \epsilon h^{(1)}_{\mu\nu}] = \epsilon R^{(1)}_{\mu\nu}[h] + \epsilon^2 R^{(2)}_{\mu\nu}[h]$
> [Side note] order O vanishes as Ric flat as $\eta_{\mu\nu} \to R_{\mu\nu}[\eta]=0$

where
> [Side note] ungauged ricci tensor to 1st order)

$R^{(1)}_{\mu\nu}[h] = \frac{1}{2}\partial_\rho\partial^\rho h_{\mu\nu} - \frac{1}{2}\partial_\rho\partial_{(\mu}h^\rho_{\nu)} - \frac{1}{2}\partial_\mu\partial_\nu h$ are linear terms.

and,
$R^{(2)}_{\mu\nu}[h] = \frac{1}{2}h^{\rho\sigma}\partial_\mu\partial_\nu h_{\rho\sigma} - h^{\rho\sigma}\partial_\rho\partial_{(\mu}h_{\nu)\sigma} + \frac{1}{4}\partial_\mu h^{\rho\sigma}\partial_\nu h_{\rho\sigma} - \frac{1}{2}\partial_\rho h^{\rho\sigma} \partial_{(\mu}h_{\nu)\sigma} + \frac{1}{2}h_{\rho\sigma}(\partial_\mu \partial^\rho h^\sigma_\nu + \dots)$ are quadratic terms.

---
55

***


This implies
> [Side note] (ricci tensor to quadratic order)
>
> [Side note] brackets evaluated on $h^{(1)}$

$R_{\mu\nu}[\eta_{\mu\nu} + \epsilon h^{(1)}_{\mu\nu} + \epsilon^2 h^{(2)}_{\mu\nu}] = \epsilon R^{(1)}_{\mu\nu}[h^{(1)}] + \epsilon^2 (R^{(1)}_{\mu\nu}[h^{(2)}] + R^{(2)}_{\mu\nu}[h^{(1)}])$
> [Side note right] call these terms this

Thus
> [Side note] what we really want is Einstein tensor

$G_{\mu\nu}[\eta + \epsilon h^{(1)} + \epsilon^2 h^{(2)}] = \epsilon G^{(1)}_{\mu\nu}[h^{(1)}] + \epsilon^2(G^{(1)}_{\mu\nu}[h^{(2)}] + G^{(2)}_{\mu\nu}[h^{(1)}])$
> [Side note] linear order in $\epsilon$

$(*)$ quadratic order in $\epsilon$:
$G^{(2)}_{\mu\nu}[h^{(1)}] = R^{(2)}_{\mu\nu}[h^{(1)}] - \frac{1}{2}\eta_{\mu\nu}\eta^{\sigma\tau}R^{(2)}_{\sigma\tau}[h^{(1)}] - \frac{1}{2}h^{(1)}_{\mu\nu}\eta^{\sigma\tau}R^{(1)}_{\sigma\tau}[h^{(1)}]$
> [Side note left] from metric + ($\epsilon x$ terms)
>
> [Side note right] remember when I take trace, I also get terms from order $\epsilon$ correction in metric
>
> [Side note far right] vaccum turns out these terms are actually 0 when we impose on-shell i.e. vacuum E.E.s

**note:** what is meant by $G^{(1)}$?
$G^{(1)}_{\mu\nu}[h] = R^{(1)}_{\mu\nu}[h] - \frac{1}{2}\eta_{\mu\nu}\eta^{\sigma\tau}R^{(1)}_{\sigma\tau}[h]$

We now consider the contracted Bianchi identity, $\nabla^\mu G_{\mu\nu} = 0$.
> [Side note right] to be explicit about what we contract over

which holds for any metric. Using $(*)$ and expanding gives
$(+) \ 0 = \epsilon \eta^{\sigma\mu}\partial_\sigma G_{\mu\nu}[h^{(1)}] + \epsilon^2(\eta^{\sigma\mu}\partial_\sigma G_{\mu\nu}^{(1)}[h^{(2)}] - 8\pi t_{\mu\nu}[h^{(1)}] + h^{(1)\sigma\mu} R^{(1)}_{\mu\nu}[h^{(1)}])$
> [Side note left] it's more terms but all prop. to linearised ricci tensor evaluated on linearised metric so of course 0.
>
> [Side note right] don't bother writing out all the terms, just this instead.

Considering $G_{\mu\nu}[\eta + \epsilon h^{(1)} + \epsilon^2 h^{(2)}] = 0$, using $(*)$ order by order, we deduce
> [Side note left] linearised EEs for $h^{(1)}$
>
> [Side note left] not really an 'energy momentum tensor'

$G^{(1)}_{\mu\nu}[h^{(1)}] = 0$.
> [Side note] use for Ricci tensor

$G^{(1)}_{\mu\nu}[h^{(2)}] = -G^{(2)}_{\mu\nu}[h^{(1)}] + \frac{1}{2}\eta_{\mu\nu}\eta^{\sigma\tau}R^{(2)}_{\sigma\tau}[h^{(1)}] = 8\pi t_{\mu\nu}[h^{(1)}]$
> [Side note right] same called $t_{\mu\nu}$ earlier.

Thus $h^{(2)}$ solves the linearised Einstein equations sourced by an 'energy momentum tensor'.
> [Side note right] want it to be divergence free: use bianchi identities

From $(+)$ we deduce that
> [Side note left] true for any metric so we have not imposed linearised E.E.s

$\eta^{\sigma\mu}\partial_\sigma G^{(1)}_{\mu\nu}[h] = 0$, which holds for ANY perturbation $h$.
And $\eta^{\sigma\mu}\partial_\sigma t_{\sigma\nu}[h^{(1)}] = 0$ when $h^{(1)}$ satisfies linearised E.E.S.
> [Side note left] total energy is conserved
>
> [Side note right] div free but not gauge invariant!

We can identify $t_{\mu\nu}$ with energy-momentum of the gravitational field, however, it is not gauge invariant. If $h^{(1)}$ decays sufficiently at $\infty$, then $\int t_{00} d^3x$ is invariant, (so gives total energy of field), but no gauge invariant local conservation.

We can get approximate gauge invariance by **averaging**.
Let $W$ be smooth, vanish for $|\mathbf{x}|^2+t^2>a$, and satisfy $\int_{\mathbb{R}^4} W(x,t)dxdt=1$.
> [Side note left] our spacetime use this to get a local average on region of scale a

---
56

***


We define the average of a tensor in almost inertial coordinates by
> [Side note] W supported on size a so non-zero outside ball of size a so deriv is order 1/a
>
> [Side note] spacetime x
>
> [Side note right] idea is we can avg over suitable region to recover gauge inv.

$<X_{\mu\nu}(x)> = \int_{\mathbb{R}^4} W(y-x) X_{\mu\nu}(y)d^4y$.

Suppose we're in far field regime, with radiation of wavelength $\lambda$ and we average over a region of size $a \gg \lambda$.
Since $\partial_\mu W \sim W/a$, we have
> [Side note] same as before, integrate by parts

$<\partial_\rho X_{\mu\nu}> = \int_{\mathbb{R}^4} \partial_\rho W(y-x) X_{\mu\nu}(y)d^4y$
roughly of order $\sim \frac{<X_{\mu\nu}>}{a}$
$\implies \frac{<\partial_\rho X_{\mu\nu}>}{<X_{\mu\nu}>} \sim \frac{1}{a} < \frac{1}{\lambda}$

We can ignore total derivatives inside averages, and thus
$<A\partial B> = <\partial(AB)> - <(\partial A)B> \approx -<(\partial A)B>$
> [Side note] integration by parts formula

With this we can show:
**EXERCISE:** 1) If $h$ solves vaccum linearised E.E,
> SHEETS 3

$<\eta^{\alpha\beta}R^{(2)}_{\alpha\beta}[h]> = 0$.
> [Side note] (raise & lower with $\eta$)

2) $<t_{\mu\nu}> = \frac{1}{32\pi} < \partial_\mu \bar{h}_{\rho\sigma}\partial_\nu\bar{h}^{\rho\sigma} - \frac{1}{2}\partial_\mu\bar{h}\partial_\nu\bar{h} - \frac{1}{2}\eta_{\mu\nu}(\dots) >$.
> [Side note left] this is kind of the result we were looking for all along

3) $<t_{\mu\nu}>$ is gauge invariant.

Using this formula and last lecture's results, we can find energy lost by a system producing gravitational waves.

The averaged spatial energy flux is $S_i = -<t_{0i}>$.
We calculate average energy flux across a sphere of radius $r$ centered on source
$<P> = -\int r^2 d\Omega <t_{0i}>\hat{x}_i$

In wave gauge $<t_{0i}> = \frac{1}{32\pi}<\partial_0\bar{h}_{\rho\sigma}\partial_i\bar{h}^{\rho\sigma} - \frac{1}{2}\partial_0\bar{h}\partial_i\bar{h}>$
> expand out
= $\frac{1}{32\pi}<\partial_0\bar{h}_{jk}\partial_i\bar{h}_{jk} - 2\partial_0\bar{h}_{0j}\partial_i\bar{h}_{0j} + \partial_0\bar{h}_{00}\partial_i\bar{h}_{00} - \frac{1}{2}\partial_0\bar{h}\partial_i\bar{h}>$

Using $\bar{h}_{ij} = \frac{2}{r}\ddot{I}_{ij}(t-r)$
> $0$ IN RADIATION GAUGE

$\partial_0 \bar{h}_{jk} = \frac{2}{r}\dddot{I}_{jk}(t-r) \quad \partial_i \bar{h}_{jk} = (-\frac{2}{r^2}\ddot{I}_{jk}(t-r)\hat{x}_i - \frac{2}{r}\dddot{I}_{jk}(t-r)\hat{x}_i)$

$-\frac{1}{32\pi} \int r^2d\Omega < \partial_0\bar{h}_{jk}\partial_i\bar{h}_{jk}> \hat{x}_i = \frac{1}{5}<\dddot{I}_{ij}\dddot{I}_{ij}>_{t-r}$
> [Side note left] integral over $d\Omega <\hat{x}_i\hat{x}_j>$
>
> [Side note left] average over period of time so will pick out $\cos^2$ terms so the other terms in middle in principle compute all terms to work out power
>
> [Side note right] AVERAGE OVER WINDOW CENTERED AT t-r

---
57

***


## Lecture 20
25.11.24

**Last lecture:**
*   EM tensor for linearised gravitational perturbations
    $t_{\mu\nu} = -\frac{1}{8\pi} (R^{(2)}_{\mu\nu}[h^{(1)}] - \frac{1}{2}\eta_{\mu\nu}\eta^{\sigma\rho}R^{(2)}_{\sigma\rho}[h^{(1)}])$
    where $R_{\mu\nu}[\eta_{\mu\nu}+\epsilon h_{\mu\nu}] = \epsilon R^{(1)}_{\mu\nu}[h] + \epsilon^2 R^{(2)}_{\mu\nu}[h] + O(\epsilon^2)$
*   After averaging over a window of size $a \gg \lambda$ ($\lambda$ typical wavelength of grav. radiation) have
    $<t_{\mu\nu}> = \frac{1}{32\pi} < \partial_\mu\bar{h}_{\rho\sigma}\partial_\nu\bar{h}^{\rho\sigma} - \frac{1}{2}\partial_\mu\bar{h}\partial_\nu\bar{h} - \frac{1}{2}\eta_{\mu\nu}\eta^{\alpha\beta}(\dots) >$
    > $=0$ IN WAVE GAUGE

**NEW CONTENT**
*   Insert expressions from radiation zone solution derived in previous lectures to compute flux through large sphere
    $<P>_z = -\int r^2 d\Omega <t_{0i}>\hat{x}_i = \frac{1}{5}<\dddot{Q}_{ij}\dddot{Q}_{ij}>_{t-r}$
    > [Side note left] integral over solid angle
    >
    > [Side note middle] clever way
    >
    > [Side note right] lose triple

---



If `{E_a^μ}_{a=1}^n$ is a dual basis (i.e. basis of covectors), then
$E_a^μ \dots ∧ E_p^μ$ for `μ₁ < μ₂ < ... < μ_p`

and we can write
$X = \frac{1}{p!} X_{μ_1 \dots μ_p} E^{μ_1} ∧ \dots ∧ E^{μ_p}$.

Another important feature of forms is that we can define a derivative
$d: Ω^p M \to Ω^{p+1} M$ by
EXTERIOR DERIVATIVE

**DEF:** If $X$ is a p-form, then in a coordinate basis
$(dX)_{μ_1 \dots μ_{p+1}} = (p+1) ∂_{[μ_1} X_{μ_2 \dots μ_{p+1}]}$ (*)
SYMMETRIC IN $μ_{p+1}, μ_1$

Suppose $∇$ is any symmetric connection, then
$∇_{[μ_1} X_{μ_2 \dots μ_{p+1}]} = ∂_{[μ_1} X_{μ_2 \dots μ_{p+1}]} - [Γ_{\dots}, X_{\dots}] - \dots - [Γ_{\dots}, X_{\dots}]$
$\implies ∇_{[μ_1} X_{μ_2 \dots μ_{p+1}]} = ∂_{[μ_1} X_{μ_2 \dots μ_{p+1}}]$

i.e. $(dX)_{μ_1 \dots μ_{p+1}} = (p+1) ∇_{[μ_1} X_{μ_2 \dots μ_{p+1}}]$

Thus $(dX)_{a_1 \dots a_{p+1}} = (p+1) ∇_{[a_1} X_{a_2 \dots a_{p+1}]}$ is well-defined independently of coordinates. However, (*) shows it does not depend on a metric or connection.

**EXERCISE (sheet 4) show:**
- $d(dX) = 0$
- $d(X∧Y) = dX∧Y + (-1)^p X∧dY$ (`X` p-form, `Y` q-form)
- $Φ^*dX = d(Φ^*X)$ if $Φ: N \to M$

The last property implies that $d$ commutes with lie derivatives
i.e. $L_v(dX) = d(L_vX)$.
This is called the EXTERIOR DERIVATIVE.

We say $X$ is CLOSED if $dX=0$, and $X$ is EXACT if $X=dY$ for some $Y$. EXACT $\implies$ CLOSED, but the converse is only true locally.

**POINCARÉ LEMMA:** If $X$ is a closed p-form ($p \ge 1$), then for any $r \in M$ there is an open neighbourhood $N \subset M$ with $r \in N$ and a (p-1)-form $Y$ defined on $N$ such that $X=dY$.

---

The extent to which CLOSED ≠ EXACT captures topological properties of $M$.

**E.g.** On $S^1$ the form $dθ$ (see ex. 1.3) is closed, but not exact (despite confusing notation).

### THE TETRAD FORMALISM

In GR it's often useful to work with an orthonormal basis of vector fields (TETRAD) `{e_a^μ}_{a=0}^3$, satisfying
$g_{ab} e_a^μ e_b^ν = g^{μν}$ (Sur is Riemannian)

Recall that the dual basis ${e^a_μ}_{a=0}^3$ is defined by
$δ_a^μ = g^μ(e_ν) = δ_a^c e_c^μ$.

We claim that
$g_a^μ = g^{μσ} g_{ab} e_b^σ$.

**PROOF:** $(g^{μσ} g_{ab} e_b^σ) e_ν^a = g^{μσ} g_{bσ} = δ_μ^ν \dots$

Recalling that $g_{ab}$ raises + lowers roman indices, and introducing the CONVENTION that $g_{μν}$ raises + lowers greek indices, we have
$g_a^μ = e_μ^a = g^{μσ} g_{ab} e_σ^b$.

we will thus denote basis vectors $e_μ$ and dual basis vectors $e^μ$.

Recall that two orthonormal bases are related by
$e_a^μ = (A^{-1})_a^b e_b'^μ$ where $g_{μν} A_ρ^μ A_σ^ν = g_{ρσ}$.
Unlike in special relativity, $A_b^a$ need not be constant. GR arises by gauging the Lorentz symmetry of SR.

**CLAIM:** $g_{μν} e_a^μ e_b^ν = g_{ab}$, $e_a^μ e_μ^b = δ_a^b$.

**PROOF:** Contract with $e_ρ^b$:
$g_{μν} e_a^μ e_b^ν e_ρ^b = g_{μν} e_a^μ δ_ρ^ν = g_{μρ} e_a^μ = (e_a)_ρ = g_{ab} e_ρ^b$.
Since equation holds for contracted with any basis vector, it holds in general.
Second equation follows from first by raising b.

---

### Lecture 21
27.11.24

**Last Lecture:**
- Differential forms, $Ω^p M$
    - $X ∧ Y$ 'wedge product'
    - $dX$ 'exterior derivative'
- Tetrad formalism: ${e_a^μ}$ on basis, ${e_μ^a}$ dual basis $e_a^μ = g_{ab} h e_b^μ$
    - raise + lower { LATIN indices w/ $g_{ab}$
    - { GREEK indices w/ $g_{μν}$
    - $g_{ab} = e_a^μ e_b^ν g_{μν}$, $g_{μν} = e_μ^a e_ν^b g_{ab}$, $e_μ^a e_a^ν = δ_μ^ν$, $e_a^μ e_μ^b = δ_a^b$.

### CONNECTION 1-FORMS

Let $∇$ be the Levi-Civita connection. The connection 1-forms are defined to be
$(ω^a_b)_μ = e_μ^a ∇_μ e_b^ν$

Recalling $e_b^μ ∇_μ e_a^ν = Γ_{μσ}^ν e_a^σ$
multiply by $e_c^ν$
$\implies ∇_μ e_a^ν = Γ_{μσ}^ν e_a^σ e_c^ν$
$\implies (∇_μ e_a)_c = Γ_{μσ}^c e_a^σ$.

so $(ω^b_a)_μ$ encodes the connection components.

**LEMMA:** $(ω_{μ})_a^b = -(ω_{μ})_b^a$.

**PROOF:** $(ω_{μ})_a^b = (e_b)_μ ∇_μ e_a^μ = ∇_μ((e_b)_μ e_a^μ) - e_a^μ ∇_μ (e_b)_μ$
$= -(e_b)_ν ∇_μ e_a^ν = -(ω_{μ})_b^a$

Now consider the exterior derivative of a basis 1-form.

**LEMMA:** The 1-form $e^a$ satisfies Cartan's first structure eqn.
$de^a + ω^a_b ∧ e^b = 0$.

**PROOF:** Note $(ω∧e)_μν^a b = (e_c^λ ∇_λ e_b^μ) e_ν^c = ∇_ν e_b^μ$.
Thus
$∇_μ e_b^ν = (ω_μ)^c_b e_c^ν = -(ω_μ)_c^b e_ν^c$
$\implies (de^b)_{ab} = 2 ∇_{[a} e_{b]}^μ = -2(ω)_{[a}^c e_{b]}^c = (ω∧e)_{ab}^c$.
what does this mean?
Given field components of ω by computing exterior derivative of e.
(efficient way of finding connection components.)

---

Note that in the orthonormal basis,
$(de^μ)_{νσ} = 2(ω_{[ν}^μ)_σ]$.
so computing $de^a$ leads to $(ω_{[ν})_σ]$ since $(ω_{μν}) = -(ω_{νμ})$.
We can check $(ω_{μν}) = (ω_{[μν)ρ]} + (ω_{[ρμ)ν]} - (ω_{[νρ)μ]}...$
So we can determine $ω_b^a$ and hence $Γ$ by computing $de^a$.

**Example:** The Schwarzchild Metric
$ds^2 = \dots$ has an obvious tetrad:
$e^0 = g dt \quad e^1 = \frac{1}{g} dr \quad e^2 = r dθ \quad e^3 = r \sinθ dφ$
where $g = \sqrt{1 - \frac{2M}{r}}$. gives the metric.

Then $de^0 = dg ∧ dt + g d(dt) = g' dr ∧ dt = g' e^1 ∧ e^0$
$de^1 = d(\frac{1}{g}) ∧ dr + \frac{1}{g} d(dr) = -\frac{g'}{g^2} dr ∧ dr = 0$
$de^2 = dr ∧ dθ$
$de^3 = \sinθ dr ∧ dφ + r \cosθ dθ ∧ dφ = \frac{g}{r} e^1 ∧ e^3 + \frac{\cotθ}{r} e^2 ∧ e^3$
$de^0 = -ω^0_μ ∧ e^μ \implies ω^0_1 = g' e^0, ω^0_2 \propto e^2, ω^0_3 \propto e^3$
$de^1 = -ω^1_μ ∧ e^μ \implies ω^1_0 \propto e^0, ω^1_2 \propto e^2, ω^1_3 \propto e^3$
$de^2 = -ω^2_μ ∧ e^μ \implies ω^2_1 = \frac{g}{r} e^2, ω^2_0 \propto e^0, ω^2_3 \propto e^3$
$de^3 = -ω^3_μ ∧ e^μ \implies ω^3_0 \propto e^0, ω^3_1 = \frac{g}{r} e^3, ω^3_2 = -\frac{\cotθ}{r} e^3$

We have:
$ω_{01} = -ω_{10} = -g' e^0$
$ω_{21} = -ω_{12} = \frac{g}{r} e^2$
$ω_{31} = -ω_{13} = \frac{g}{r} e^3$
$ω_{32} = -ω_{23} = \frac{\cotθ}{r} e^3$
all other components vanish

### CURVATURE 2-FORMS

We compute $dω^μ_ν$
$(dω^μ_ν)_{ab} = ∇_a(ω^μ_ν)_b - ∇_b(ω^μ_ν)_a$
$= ∇_a(e_c^μ ∇_b e_ν^c) - ∇_b(e_c^μ ∇_a e_ν^c)$
$= e_c^μ(∇_a ∇_b e_ν^c - ∇_b ∇_a e_ν^c) + (∇_a e_c^μ)∇_b e_ν^c - (∇_b e_c^μ)∇_a e_ν^c$
$= e_c^μ(R^c_{dab})e_ν^d + e_d^μ(∇_a e_c^d)e_f^c(∇_b e_ν^f) - e_d^μ(∇_b e_c^d)...$

---

$= (H^μ_ν)_{ab} + (ω_σ^μ ∧ ω_ν^σ)_{ab}$
where $H^μ_ν = \frac{1}{2} R^μ_{νστ} e^σ ∧ e^τ$ are the curvature 2-forms. We've shown Cartan's second structure equation,
$dω^μ_ν + ω^μ_σ ∧ ω^σ_ν = H^μ_ν$,
gives an efficient way to compute $R^μ_{νστ}$ in an o/n basis.

**Returning to our example:**
$dω^0_1 = d(g' e^0) = d(g'g dt) = (g''g + g'^2) dr ∧ dt = (g''g + g'^2) e^1 ∧ e^0$
$ω^0_μ ∧ ω^μ_1 = ω^0_1 ∧ ω^1_1 + ω^0_2 ∧ ω^2_1 + ω^0_3 ∧ ω^3_1 + ω^0_0 ∧ ω^0_1 = 0$
$\therefore H^0_1 = -(g''g + g'^2) e^0 ∧ e^1$
hence $R^0_{101} = (g''g + g'^2) = -\frac{1}{2}(g^2)'' = \frac{2M}{r^3}$
and $R^μ_{ν01} = 0$ otherwise.

**Exercise:** Find other $H^μ_ν$ and show $R_{ab}$.

### VOLUME FORM AND HODGE DUAL
We say a manifold is ORIENTABLE if it admits a nowhere vanishing n-form ($n=\dim M$) $E_{a_1, \dots, a_n}$, an orientation form.

Two such forms are equivalent if $E' = gE$ for some smooth, everywhere positive $f$. $[E]_N$ is an orientation.

A basis of vectors ${e_μ}_{μ=1}^n$ is right-handed if
$E(e_1, \dots, e_n) > 0$.

A coordinate system is right-handed if $\{ \frac{∂}{∂x^μ} \}_{μ=1}^n$ are right-handed.

---

### Lecture 22
29.11.24

**Last lecture**
- Tetrad formalism
- Cartan's 1st + 2nd structure equations
    $de^a + ω^a_b ∧ e^b = 0 \qquad ω_{μν} = -ω_{νμ} \quad$ connection 1-forms
    $dω^μ_ν + ω^μ_σ ∧ ω^σ_ν = H^μ_ν \qquad H^μ_ν = \frac{1}{2} R^μ_{νστ} e^σ ∧ e^τ \quad$ curvature 2-forms
- Manifold is orientable if it admits a nowhere vanishing n-form, $E$.
    - Two such forms equivalent if $E' = gE$ ($g>0$ everywhere)
    - $[E]_N$ is an orientation
    - ${e_μ}_{μ=1}^n$ is right-handed if $E(e_1, \dots, e_n) > 0$. coord. system is RH if $\{ \frac{∂}{∂x^μ} \}_{μ=1}^n$ is RH.

An oriented manifold with metric has a preferred normalisation for $E$. For a right-handed orthonormal basis, we define the volume form $E$ by
$E(e_1, \dots, e_n) = 1$ (indep. of choice of RH o/n basis)

If we work in a RH coord. system $\{x^μ\}_{μ=1}^n$, then
$\frac{∂}{∂x^α} = e_μ^α \frac{∂}{∂\bar{x}^μ} \implies \frac{∂}{∂\bar{x}^μ} = e_μ^α \frac{∂}{∂x^α}$
Then,
$E(\frac{∂}{∂x^1}, \dots, \frac{∂}{∂x^n}) = E(e_1^μ \frac{∂}{\partial \bar{x}^μ}, \dots, e_n^μ \frac{∂}{\partial \bar{x}^μ})$
$= \sum_{π \in Sym(n)} σ(π) e_1^{π(1)} \times \dots \times e_n^{π(n)} = \det(e_μ^α)$
but $e_α^μ e_β^ν g_{μν} = g_{αβ} \implies \det(e_μ^α) = \sqrt{|g|}$ where $g = \det(g_{αβ})$.
$\therefore E = \sqrt{|g|} dx^1 ∧ dx^2 ∧ \dots ∧ dx^n$.

Equivalently, $E_{123...n} = \sqrt{|g|}$ in coord. basis.

**EXERCISE:** In the same coord. basis
$E^{123...n} = \pm \frac{1}{\sqrt{|g|}} \begin{cases} + & \text{RIEMANNIAN} \\ - & \text{LORENTZIAN} \end{cases}$

**LEMMA:** $∇E=0$
**PROOF:** In normal coords. at $p$; $∂_μ g_{νρ}|_p = 0$, $Γ_{μν}^ρ|_p=0$.

---

$\implies ∇_μ E_{μ_1 \dots μ_n}|_{at p} = ∂_μ E_{μ_1 \dots μ_n}|_{at p} + (\dots) = 0$
tensor equation so holds everywhere.

**LEMMA:** $E_{a_1 \dots a_p c_{p+1} \dots c_n} E^{b_1 \dots b_p c_{p+1} \dots c_n} = \pm p!(n-p)! δ_{[b_1}^{a_1} δ_{b_2}^{a_2} \dots δ_{b_p]}^{a_p}$
$\begin{cases} + & \text{RIEMANNIAN} \\ - & \text{LORENTZIAN} \end{cases}$
**PROOF:** (exercise)

We can use $E$ to relate $Ω^p M$ to $Ω^{n-p} M$:

**DEF:** On an oriented manifold with metric, the HODGE DUAL of a p-form $X$ is
$(*X)_{a_1 \dots a_{n-p}} = \frac{1}{p!} E_{a_1 \dots a_{n-p} b_1 \dots b_p} X^{b_1 \dots b_p}$

From previous results we can show
**LEMMA:** $*(*X) = \pm (-1)^{p(n-p)} X$ $\begin{cases} + & \text{RIEMANNIAN} \\ - & \text{LORENTZIAN} \end{cases}$
$(*d*X)_{a_1 \dots a_{p-1}} = \pm (-1)^{p(n-p)} ∇_b X_{a_1 \dots a_{p-1}}^b$ $\begin{cases} + & \text{RIEMANNIAN} \\ - & \text{LORENTZIAN} \end{cases}$

**Examples.**
1. In Euclidean space, identifying a vector field $X^a$ with the one form $X_a$, the usual operations of vector calculus become
$∇f = df$, $\text{div}X = *d*X$, $\text{curl}X = *dX$.
note $d^2=0 \implies \text{curl} \, ∇f = 0$ and $\text{div} \, \text{curl} X = 0$.

2. Maxwell's Equations
$∇_a F^{ab} = -4π j^b$ and $∇_{[a} F_{bc]} = 0$
can be written
$d*F = -4π*j$, $dF=0$.
Poincaré's lemma implies we can write $F=dA$ for some one-form A, locally.

we have introduced this formulation w/ d & *s, turns out this makes integration on manifolds & stoke's thm quite elegant.

### INTEGRATION ON MANIFOLDS

Suppose on a manifold M we have a RH coordinate chart
$Φ: O \to U$ with coordinates $x^μ$. If $X$ is an n-form which vanishes outside $O$, we can write

---

$X = X_{1 \dots n} dx^1 ∧ \dots ∧ dx^n$.
If $ψ: O \to U$ is another RH coordinate chart with coords $\{y^μ\}$ then
$X = \tilde{X}_{1 \dots n} dy^1 ∧ \dots ∧ dy^n = \tilde{X}_{1 \dots n} \frac{∂y^1}{∂x^{μ_1}} \dots \frac{∂y^n}{∂x^{μ_n}} dx^{μ_1} ∧ \dots ∧ dx^{μ_n}$
$= \tilde{X}_{1 \dots n} \det(\frac{∂y}{∂x}) dx^1 ∧ \dots ∧ dx^n$
$\therefore X_{1 \dots n} = \tilde{X}_{1 \dots n} \det(\frac{∂y}{∂x})$

As a result
$\int_U X_{1 \dots n} dx^1 \dots dx^n = \int_U \tilde{X}_{1 \dots n} dy^1 \dots dy^n$.

We can define
$\int_M X := \int_O X_{1 \dots n} dx^1 \dots dx^n$.

On any (2nd countable) manifold, we can find a countable atlas of charts $(O_i, Φ_i)$, and smooth functions $χ_i: M \to [0,1]$ such that $χ_i$ vanishes outside $O_i$ and
$\sum_i χ_i(p) = 1$ $∀ p \in M$, and the sum is locally finite.

Then for any n-form $X$ we define
$\int_M X = \sum_i \int_M χ_i X = \sum_i \int_{O_i} χ_i X$.

This doesn't depend on a choice of $χ_i$'s.

**Remarks**
- Computation showing coord. invariance implies that for a diffeomorphism $Φ: M \to M$
$\int_M X = \int_M Φ^*X$
- If $M$ is a manifold with metric and vol. form $E$, then if $f: M \to R$ is a scalar, $fE$ is an n-form, and we can define
$\int_M f := \int_M fE$.
In local coordinates, if $f$ vanishes outside $O$.
$\int_M f = \int_O f(x) \sqrt{|g|} dx^1 \dots dx^n = \int_M f dvol_g$.

### SUBMANIFOLDS AND STOKES THEOREM

**DEF:** Suppose $S, M$ are manifolds and $\dim S = m < n = \dim M$. A smooth map $ι: S \to M$ is an embedding if it is an injection (i.e. $ι_*: T_p S \to T_{ι(p)}M$ is injective) and if $ι$ is injective i.e. $ι(p) = ι(q) \implies p=q$.
If $ι$ is an embedding, then $ι(S)$ is an embedded submanifold. If $m=n-1$ we call it a hypersurface) (mostly drop $ι$ when obvious from context and write $ι(S)=S$).

---

If $S, M$ are orientable, and $ι(S)$ is an embedded submanifold of M. We define the integral of an n-form $X$ over $ι(S)$ by
$\int_{ι(S)} X = \int_S ι^*(X)$.
Note that if $X=dY$ then
$\int_{ι(S)} dY = \int_S d(ι^* Y)$. $(d\circ ι^* = ι^* \circ d)$

### Lecture 23
2.12.24

**Last lecture**
- If M is an n-dim oriented manifold, X is an n-form defined
$\int_M X$.
- If M carries a metric (or otherwise has a preferred choice of volume form E) defined
$\int_M f := \int_M fE = \int_M f dvol_g$. E is volume form of g.
- If S is an m-dimensional manifold ($m<n$), then
$ι: S \to M$ is an embedding if it is an IMMERSION and is INJECTIVE ($ι(p)=ι(q) \implies p=q$) ($ι_*: T_p S \to T_{ι(p)}M$ injective).
If so, $ι(S)$ is an m-dim SUBMANIFOLD of M.
- If $ι(S)$ is an m-dim submanifold and Y is an m-form on M, then
$\int_{ι(S)} Y := \int_S ι^*(Y)$.

**DEF:** A manifold with boundary, M, is defined just as for a manifold, but charts are maps
$Φ_α: O_α \to U_α$, where $U_α$ is an open subset of $R^n_{\le 0} = \{(x^1, \dots, x^n) / x^1 \le 0\}$.

The boundary of M, $∂M$ is the set of points mapped in any chart to $\{x^1=0\}$. It is naturally an n-1 dim. manifold with an embedding $i: ∂M \to M$.
If M is oriented $∂M$ inherits an orientation by requiring $(x^2, \dots, x^n)$ is a RH chart on $∂M$ when $(x^1, \dots, x^n)$ is RH on M.
"in practice we are not going to use this definition but it's good to have it"

---

### STOKES THEOREM

If N is an oriented n-dim. manifold with boundary, and X is an (n-1)-form, then
$\int_N dX = \int_{∂N} X$.

Stokes theorem is the basis of all 'integration by parts' arguments.
If N carries a metric, we can reformulate Stokes theorem as the divergence theorem.

If V is a vector field on N, then we define
$(V \rfloor E)_{a_2 \dots a_n} = V^a E_{a a_2 \dots a_n}$

We can check that
$d(V \rfloor E) = (∇_a V^a) E$.

If we define the flux of V through an embedded hypersurface S by
$\int_S V \cdot dS := \int_S V \rfloor E$
then Stokes theorem implies $\int_N ∇_a V^a dvol_g = \int_{∂N} V \cdot dS$.

Recall that a hypersurface $ι(S)$ is
$\begin{cases} \text{SPACELIKE if } h=ι^*g \text{ is RIEMANNIAN} \\ \text{TIMELIKE if } h=ι^*g \text{ is LORENTZIAN} \end{cases}$
In this case we can relate $(* (V \rfloor E_g))$ to the volume form on $(S,h)$.
Pick $b_2, \dots, b_n$ a RH o/n basis on S (w.r.t. h). Then $ιb_2, \dots, ιb_n$ are o/n in N. The unit normal to S is the unique unit vector $\hat{n}$ orthogonal to $ιb_2, \dots, ιb_n$ with
$E(\hat{n}, ιb_2, \dots, ιb_n) = g(\hat{n}, \hat{n}) (= \pm 1)$.

If $ι(S) \begin{cases} \text{SPACELIKE} \iff \hat{n} \text{ TIMELIKE} \\ \text{TIMELIKE} \iff \hat{n} \text{ SPACELIKE} \end{cases}$
with this definition,
$V \rfloor E(ιb_2, \dots, ιb_n) = V^a \hat{n}_a$
thus $(ι^*(V \rfloor E_g)) = ι^*(V^a \hat{n}_a) E_h$
$\implies \int_S V \cdot dS = \int_S V^a \hat{n}_a dvol_h$

We've shown
$\int_{∂N} V^a \hat{n}_a dvol_h = \int_N ∇^a V_a dvol_g$.

---

Checking the definitions, $\hat{n}^a$ points 'out' of N for ∂N TIMELIKE (or g Riemannian) and 'into' N for ∂N SPACELIKE.

### THE EINSTEIN HILBERT ACTION

We want to derive Einstein's equations from an action principle. we expect an action of the form
$S[g, \text{MATTER}] = \int_M L(g, \text{MATTER}) dvol_g$
where L is a scalar Lagrangian. If we ignore matter for now, an obvious guess for L is L=R, scalar curvature. This gives the Einstein-Hilbert action.
$S_{EH}[g] = \frac{1}{16π} \int_M R dvol_g$.

In order to derive e.o.m. from an action, we consider $g+\delta g$ where $δg$ vanishes outside a compact (=bounded) set in M, and expand to first order in $δg$ (considered small). We require variation of $S_{EH}$ to vanish at this order. We want to compute
$δS_{EH} = S_{EH}[g+δg] - S_{EH}[g]$ (ignore $O((δg)^2)$).

First we consider $dvol_g = \sqrt{|g|} d^n x$.
To compute $δ|g|$, recall we can write the determinant as
$g = g_{μν} Δ^{μν}$ (μ means fixed value; no sum over μ)
where $Δ^{μν}$ is cofactor matrix $Δ^{μν} = (-1)^{μ+ν} | \dots |$
$\implies δg = Δ^{μν} δg_{μν}$
$δg_{μν}$ is independent of $g_{μν}$, and satisfies $Δ^{μν} = g g^{μν}$. consider under $g_{μν} \to g_{μν} + δg_{μν}$, $δg = Δ^{μν} δg_{μν} = g g^{μν} δg_{μν}$
$\therefore δ\sqrt{|g|} = \frac{1}{2}\frac{δg}{\sqrt{|g|}} = \frac{1}{2} \sqrt{|g|} g^{μν} δg_{μν}$
$\therefore δ(dvol_g) = \frac{1}{2} g^{ab} δg_{ab} dvol_g$.

---

To compute $δR_g$, we first consider $δΓ_{μν}^ρ$. The difference of two connections is a tensor, so this is a tensor $δΓ_{ab}^c$. To compute this, we can consider normal coordinates for $g_{ab}$ at some point $p$. Then since $∂_μ g_{νρ}|_p = 0$.
$δΓ_{μν}^ρ|_p = \frac{1}{2} g^{ρσ} (δg_{σν|μ} + δg_{σμ|ν} - δg_{μν|σ})|_p$
$= \frac{1}{2} g^{ρσ} (∇_μ δg_{σν} + ∇_ν δg_{σμ} - ∇_σ δg_{μν})|_p$
$\dots δΓ_{bc}^a = \frac{1}{2} g^{ad} (∇_c δg_{db} + ∇_b δg_{dc} - ∇_d δg_{bc})$.

Next we consider $δR_{νρσ}^μ$. Again work in normal coords at p.
$R_{νρσ}^μ = ∂_ρ(Γ_{νσ}^μ) - ∂_σ(Γ_{νρ}^μ) + O(ΓΓ)$
$\implies δR_{νρσ}^μ|_p = [∂_ρ(δΓ_{νσ}^μ) - ∂_σ(δΓ_{νρ}^μ)]|_p$
$= [∇_ρ δΓ_{νσ}^μ - ∇_σ δΓ_{νρ}^μ]|_p$
$\dots δR_{bcd}^a = ∇_c δΓ_{bd}^a - ∇_d δΓ_{bc}^a$
$\implies δR_{ab} = ∇_c δΓ_{ab}^c - ∇_b δΓ_{ac}^c$

observing that $δ(g^{ab}g_{bc}) = 0 \implies (δg^{ab}) = -g^{ac} g^{bd} δg_{cd}$. We finally have
$δR_g = δ(g^{ab} R_{ab}) = (δg^{ab})R_{ab} + g^{ab} δR_{ab}$
$= -R^{ab} δg_{ab} + g^{ab}(∇_c δΓ_{ab}^c - ∇_b δΓ_{ac}^c)$
$= -R^{ab} δg_{ab} + ∇_c X^c$
where $X^c = g^{ab} δΓ_{ab}^c - g^{cb} δΓ_{ab}^a$.

---

### Lecture 24
4.12.24

**Last lecture**
- Stokes / divergence theorem:
$\int_M ∇_a X^a dvol_g = \int_{∂M} X^a \hat{n}_a dvol_h$
- Einstein-Hilbert action
$S_{EH}[g] = \frac{1}{16π} \int_M R dvol_g$
- Under a variation $g \to g + δg$ ($δg$ small, vanish outside coord chart)
$δ(dvol_g) = \frac{1}{2} g^{ab} δg_{ab} dvol_g$; $δ(R_g) = -R^{ab} δg_{ab} + ∇_a X^a$
$X^a = g^{cd} δΓ_{cd}^a - g^{ac} δΓ_{cd}^d$.

We deduce from the formulae for $δR_g$, $δdvol_g$ that
$δS_{EH} = \frac{1}{16π} \int_M \{(\frac{1}{2}g^{ab}R - R^{ab}) δg_{ab} + ∇_c X^c\} dvol_g$.
$= \frac{1}{16π} \int_M -G^{ab} δg_{ab} dvol_g$
where we've used the fact that $δg$ and hence $X$ vanish on $∂M$. to drop the last term using the divergence theorem.
We immediately see that $δS_{EH}=0$ for all variations $δg_{ab}$ if and only if $g_{ab}$ solves the vacuum Einstein equations.

Suppose we also have a contribution from matter fields
$S_{tot.} = S_{EH} + S_{matter}$, $S_{matter} = \int_M L[Φ, g] dvol_g$.
Under a variation $g \to g+δg$ we must have
$δS_{matter} = \frac{1}{2} \int_M T^{ab} δg_{ab} dvol_g$
For some symmetric 2-tensor $T^{ab}$.
Varying $g$ in $S_{tot}$ gives $[G^{ab} = 8πT^{ab}]$ i.e. Einstein's equations.

**E.g.** If $ψ$ is a scalar and $L_{matter} = -\frac{1}{2} g^{ab} ∇_a ψ ∇_b ψ$, then under $g \to g+δg$
$δS_{matter} = -\int_M [\frac{1}{2} δ(g^{ab}) ∇_a ψ ∇_b ψ dvol_g + \frac{1}{2} g^{ab} ∇_a ψ ∇_b ψ δ(dvol_g)]$
$= \frac{1}{2} \int_M (g^{ac} g^{bd} ∇_c ψ ∇_d ψ - \frac{1}{2} g^{ab} g^{cd} ∇_c ψ ∇_d ψ) δg_{ab} dvol_g$
$= \frac{1}{2} \int_M T^{ab} δg_{ab} dvol_g$
where $T^{ab} = ∇^a ψ ∇^b ψ - \frac{1}{2} g^{ab} ∇_c ψ ∇^c ψ$.

---

**Exercise:** Show that varying $L=\frac{1}{2}g^{ab}∇_aψ∇_bψ$ gives the wave equation $◻ψ=0$.

It can be shown that diffeomorphism invariance of the matter action implies $∇_a T^{ab} = 0$. ($T^{ab}$ is divergence free)

### THE CAUCHY PROBLEM FOR EINSTEIN'S EQUATIONS

We expect Einstein's equations can be solved given data on a spacelike hypersurface. What is the right data?
Suppose $ι: Σ \to M$ is an embedding, such that $ι(Σ)$ is spacelike. Then $h=ι^*(g)$ is Riemannian.

Let $n$ be a choice of unit normal to $ι(Σ)$. We define for $X, Y$ vector fields on Σ
$k(X,Y) = ι^*(g(n, ∇_{\tilde{X}} \tilde{Y}))$
where $L_{\tilde{n}} \tilde{X} = \tilde{X}$, $L_{\tilde{n}} \tilde{Y} = \tilde{Y}$ on $ι(Σ)$.

We pick local coordinates $\{y^i\}$ on Σ and $\{x^μ\}$ on M such that
$ι: (y^1, y^2, y^3) \to (0, y^1, y^2, y^3)$, then $n_μ dx^μ$ say $n_μ = αδ_μ^0$.
If $X, Y$ vector fields on Σ, say $X=X^i \frac{∂}{∂y^i}$, $Y=Y^i \frac{∂}{∂y^i}$, take
$\tilde{X} = X^i \frac{∂}{∂x^i}$, $\tilde{Y} = Y^i \frac{∂}{∂x^i}$.
Then $k(X,Y) = g_{μν} n^μ (\tilde{X}^ρ ∇_ρ \tilde{Y}^ν)$
$= αδ_μ^0 \tilde{X}^ρ (∂_ρ \tilde{Y}^μ + Γ_{ρτ}^μ \tilde{Y}^τ) = \alpha (\frac{1}{2} \tilde{X}^i \tilde{Y}^j (\dots))$
(1st term vanishes $y^0=0$?).

$\dots k$ is a symmetric 2-tensor on Σ. We can show (example sheet 4) that if $(M,g)$ solves the vacuum Einstein equations, then Einstein constraint equations hold:
$R_h - k_j^i k_i^j + k_i^i k_j^j = 0$
$∇_j k_i^j - ∇_i k_j^j = 0$ } (†)

Conversely, if we are given
Σ A 3-manifold
h A Riemannian metric on Σ
k A symmetric 2-tensor on Σ

---

such that (†) hold. Then there exists a solution $(M, g)$ of the vaccum Einstein equations, and an embedding $ι: Σ \to M$ such that $h$ is the induced metric on Σ, and $k$ is the 2nd fundamental form of $ι(Σ)$.

*Diagram shows two solutions $(M_1, g_1)$ and $(M_2, g_2)$ containing the same initial data surface $(\Sigma, h, k)$ via embeddings $ι_1$ and $ι_2$. The uniqueness theorem states that there are neighbourhoods $N_1$ of $ι_1(\Sigma)$ in $M_1$ and $N_2$ of $ι_2(\Sigma)$ in $M_2$ and a diffeomorphism $\Phi: N_1 \to N_2$ between them.*

(This result is due to Choquet-Bruhat (existance) and Choquet-Bruhat + Geroch (geometric uniqueness).)

This is one way of looking at EEs as evolution problem but w/o having to begin by fixing a gauge (finding ICs that satisfy the constraint equations is difficult.)

well done 😊