# Truth-vs-prediction correlations (ρ = mean ± std over 3 seeds)

| Model | ρ(c_Ht) | MSE | bias(c_Ht) |
| --- | --- | --- | --- |
| 2D Histogram | — | — | -0.045±0.020 |
| MLP | 0.326±0.005 | 0.061±0.000 | +0.026±0.000 |
| Transformer | 0.315±0.002 | 0.062±0.001 | +0.023±0.000 |
| Transformer (LLoCa) | 0.291±0.011 | 0.059±0.001 | +0.022±0.000 |
| LGATr | 0.315±0.002 | 0.057±0.000 | +0.023±0.000 |
| LorentzNet | 0.208±0.004 | 0.086±0.001 | +0.021±0.001 |

_MLE bias = θ̂ − θ_true, where θ̂ is the minimum of each coefficient's averaged 1D marginal of the seed-averaged LLR (the same marginal the plots show), parabola-refined sub-grid; ± is the per-seed spread. `unconstr.` marks coefficients whose 68% marginal reaches the grid edge — the data does not bound them within the scan range._
