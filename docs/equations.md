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

