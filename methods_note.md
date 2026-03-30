# Numerical Methods

We simulate the emergence of coherence using a two-dimensional time-dependent Ginzburg–Landau (TDGL) dynamics on periodic square lattices of linear size \(L\). The scalar coherence field \(\Psi\) evolves according to a relaxational stochastic dynamics of Model-A type,

\[
\Psi(\tau+\Delta\tau)=\Psi(\tau)+\Delta\tau\,\gamma
\left[
\nabla^2\Psi-2\alpha(\lambda)\Psi-4\beta\Psi^3
\right]+\eta,
\]

where \(\gamma\) is the relaxation rate, \(\beta>0\) is the quartic self-interaction coefficient, and \(\eta\) is a Gaussian white-noise term with variance controlled by an effective fluctuation scale \(T_0\). The control parameter is defined as

\[
\alpha(\lambda)=a_0(\lambda_c-\lambda),
\]

so that the disordered regime corresponds to \(\lambda<\lambda_c\) and the ordered regime to \(\lambda>\lambda_c\).

For each lattice size \(L\), the system is equilibrated over a burn-in phase and subsequently sampled over a fixed number of TDGL updates. Magnetization-like observables are computed from the spatial mean of the field, and the order parameter is defined as \(\langle |\Psi| \rangle\). The susceptibility is estimated from fluctuations of the spatial mean, and the Binder cumulant is computed from the second and fourth moments of the magnetization distribution.

To estimate the coherence correlation length, we compute the connected two-point correlation function using the FFT-based periodic autocorrelation of the centered field, followed by radial binning in two dimensions. A proxy for the correlation length \(\xi\) is then extracted from the radial decay scale of the normalized connected correlator.

Because configurations sampled along the TDGL trajectory are temporally correlated, we estimate the integrated autocorrelation time \(\tau_{\mathrm{int}}\) for the magnetization and use it to correct the effective sample size in standard-error calculations. This avoids underestimating uncertainties near criticality, where critical slowing down becomes significant.

Finite-size scaling is performed across multiple lattice sizes. The critical coupling \(\lambda_c\) is estimated primarily from Binder-cumulant crossings, with susceptibility peaks used as a secondary diagnostic. Data collapse is implemented by rescaling observables according to standard finite-size scaling forms,

\[
\langle |\Psi| \rangle L^{\beta/\nu}
\quad\text{vs}\quad
(\lambda-\lambda_c)L^{1/\nu},
\]

and

\[
\chi L^{-\gamma/\nu}
\quad\text{vs}\quad
(\lambda-\lambda_c)L^{1/\nu}.
\]

Optimal exponents are obtained by grid search over \((\nu,\beta)\) and \((\nu,\gamma)\), minimizing a bin-averaged variance collapse metric.

All ensemble-level estimates are generated from multiple independent random seeds. Raw outputs, summary statistics, and publication-ready figures are saved automatically for reproducibility.
