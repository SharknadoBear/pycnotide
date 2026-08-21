# Scientific equations

PycnoTide stores the constituent coefficient as

\[
C_c(d)=A_c(d)e^{-i\phi_c(d)}
\]

and predicts

\[
q(d,t)=\Re\left\{C_c(d)e^{i[\omega_c(t-t_0)+\psi_c]}\right\}.
\]

The real-valued design columns are therefore `cos(theta)` and `-sin(theta)`,
whose fitted coefficients are respectively the real and imaginary parts of `C`.

For positive-down depth and positive-up isopycnal displacement,

\[
\rho'(d,t)=\xi(d,t)\,\frac{\partial\rho_0}{\partial d}.
\]

The current-aware straight-ray helper solves

\[
\omega=\sigma+kU_\parallel,
\qquad \sigma^2=f^2+c^2k^2,
\qquad \psi=-\int k\,ds.
\]

Only the positive-wavenumber, positive-intrinsic-frequency branch continuous with
the zero-current result is admissible. This is a locally hydrostatic, straight-ray
eikonal sensitivity and not a bent-ray or full wave-action model.

Substitution of `sigma=omega-k U_parallel` gives the quadratic

\[
(U_\parallel^2-c^2)k^2-2\omega U_\parallel k+(\omega^2-f^2)=0.
\]

At every two-kilometre path node, PycnoTide rejects complex roots, negative
wavenumbers, and branches with `sigma <= |f|`, then selects the admissible root
closest to

\[
k_0=\frac{\sqrt{\omega^2-f^2}}{c}.
\]

For a trial bearing and speed, nuisance and harmonic coefficients remain linear.
The variable-projection objective is therefore

\[
J(\alpha,c)=\frac{1}{N}\left\|W^{1/2}
\left[q-X(\alpha,c)\hat\beta(\alpha,c)\right]\right\|_2^2,
\qquad
\hat\beta=\arg\min_\beta\|W^{1/2}(q-X\beta)\|_2^2.
\]

Campaign fitting uses equal total weight for each glider and equal profile weight
within a glider. The 48-hour bootstrap resamples complete profile-time blocks.
