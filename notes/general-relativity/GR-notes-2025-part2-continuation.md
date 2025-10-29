This is a transcription of pages 44-50 from the provided PDF document. The transcription starts from page 26 of the PDF file, which corresponds to page 44 of the complete document as requested.

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