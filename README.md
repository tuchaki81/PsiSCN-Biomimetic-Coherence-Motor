# ΨSCN Biomimetic Coherence Motor

A TDGL (Model A) simulation of a real scalar field with φ⁴ potential that demonstrates spontaneous coherence via a second-order phase transition. Designed as a biomimetic "coherence engine" candidate for next-generation AGI systems.

## Motivation
Large language models excel at pattern matching but still suffer from coherence collapse in long reasoning chains. This motor generates emergent long-range order from local noisy dynamics — inspired by critical phenomena in physics, neuroscience (global workspace ignition), and driven-dissipative systems.

The emergent `k_eff` term introduces a geometric-like feedback that grows with coherence, potentially useful for self-referential consistency or "curving" representation space toward truth.

## Features
- Tunable control parameter λ
- Order parameter, susceptibility, Binder cumulant, correlation length
- Emergent effective stiffness `k_eff(α, ζ)`
- Reproducible results with plots and JSON export

## How to Run
```bash
pip install numpy matplotlib
python psiscn_motor.py
