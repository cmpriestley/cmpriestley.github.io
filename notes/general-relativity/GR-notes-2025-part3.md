Here is the full transcription of the provided PDF document (pages 51-73) in markdown format.

***

### Page 51

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

### Page 52

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

### Page 53

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

### Page 54

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

### Page 55

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

### Page 56

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

### Page 57

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

### Page 58

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