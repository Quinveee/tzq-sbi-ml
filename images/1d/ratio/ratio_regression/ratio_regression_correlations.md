# Truth-vs-prediction correlations (ρ = mean ± std over 3 seeds)

| Model | log r corr. | MSE | bias(c_Ht) |
| --- | --- | --- | --- |
| 2D Histogram | — | — | -0.045±0.020 |
| MLP | 0.320±0.005 | 0.007±0.000 | +0.026±0.001 |
| Transformer | 0.272±0.007 | 0.007±0.000 | +0.024±0.002 |
| Transformer (LLoCa) | 0.326±0.001 | 0.007±0.000 | +0.023±0.000 |
| LGATr | 0.257±0.004 | 0.007±0.000 | +0.023±0.001 |
| LorentzNet | 0.298±0.001 | 0.007±0.000 | +0.021±0.000 |

_MLE bias = θ̂ − θ_true, where θ̂ is the minimum of each coefficient's averaged 1D marginal of the seed-averaged LLR (the same marginal the plots show), parabola-refined sub-grid; ± is the per-seed spread. `unconstr.` marks coefficients whose 68% marginal reaches the grid edge — the data does not bound them within the scan range._
