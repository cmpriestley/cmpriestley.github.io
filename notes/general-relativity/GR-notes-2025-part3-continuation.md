This is the transcription for pages 59-73 of the provided PDF.

### Page 59

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
### Page 60

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
### Page 61

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
### Page 62

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
### Page 63

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
### Page 64

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
### Page 65

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
### Page 66

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
### Page 67

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
### Page 68

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
### Page 69

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
### Page 70

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
### Page 71

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
### Page 72

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
### Page 73

such that (†) hold. Then there exists a solution $(M, g)$ of the vaccum Einstein equations, and an embedding $ι: Σ \to M$ such that $h$ is the induced metric on Σ, and $k$ is the 2nd fundamental form of $ι(Σ)$.

*Diagram shows two solutions $(M_1, g_1)$ and $(M_2, g_2)$ containing the same initial data surface $(\Sigma, h, k)$ via embeddings $ι_1$ and $ι_2$. The uniqueness theorem states that there are neighbourhoods $N_1$ of $ι_1(\Sigma)$ in $M_1$ and $N_2$ of $ι_2(\Sigma)$ in $M_2$ and a diffeomorphism $\Phi: N_1 \to N_2$ between them.*

(This result is due to Choquet-Bruhat (existance) and Choquet-Bruhat + Geroch (geometric uniqueness).)

This is one way of looking at EEs as evolution problem but w/o having to begin by fixing a gauge (finding ICs that satisfy the constraint equations is difficult.)

well done 😊