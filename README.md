# Coherence Criticality Simulation (2D TDGL)

This repository implements a numerical framework to study **critical behavior of a coherence field** using a two-dimensional time-dependent Ginzburg–Landau (TDGL) model.

The project investigates whether coherence, treated as a physical field, exhibits:

- phase transitions
- finite-size scaling
- universality-like behavior
- emergent collective structure

---

## 📌 Overview

We simulate a scalar field \\( \Psi(x, y, t) \\) evolving under stochastic TDGL dynamics:

\[
\Psi(t+\Delta t) = \Psi(t) + \gamma \Delta t \left[\nabla^2 \Psi - 2\alpha(\lambda)\Psi - 4\beta \Psi^3 \right] + \eta
\]

with:

\[
\alpha(\lambda) = a_0 (\lambda_c - \lambda)
\]

The control parameter \\( \lambda \\) drives the system through a **coherence phase transition**.

---

## 🔬 Key Features

- 2D lattice simulation (periodic boundary conditions)
- Radial isotropic correlation function (FFT-based)
- Correlation length estimation \\( \xi \\)
- Binder cumulant analysis
- Susceptibility and order parameter tracking
- Finite-size scaling (FSS)
- Automatic data collapse (grid search)
- Bootstrap uncertainty estimation
- Autocorrelation-time correction (critical slowing down)

---

## 📁 Project Structure
