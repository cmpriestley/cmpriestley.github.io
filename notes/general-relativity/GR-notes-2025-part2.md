Here is the complete transcription of the provided PDF document (pages 26-50) in markdown format.

---

### Page 26

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

### Page 27

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

### Page 28

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

### Page 29

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

### Page 30

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

### Page 31

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

### Page 32

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

### Page 33

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

### Page 34

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

### Page 35

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

### Page 36

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

### Page 37

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

### Page 38

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

### Page 39

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

### Page 40

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

### Page 41

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

### Page 42

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

### Page 43

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