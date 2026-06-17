# Truth-vs-prediction correlations (ρ = mean ± std over 3 seeds)

| Model | log r corr. | MSE | bias(c_Ht) | bias(c_tW) | bias(c_tB) |
| --- | --- | --- | --- | --- | --- |
| 2D Histogram | — | — | -0.195±0.018 | +0.010±0.009 | +0.082±0.010 |
| MLP | 0.254±0.004 | 0.008±0.000 | +0.002±0.003 | +0.031±0.003 | +0.076±0.006 |
| Transformer | 0.220±0.010 | 0.007±0.000 | -0.011±0.017 | +0.019±0.009 | +0.027±0.004 |
| Transformer (LLoCa) | 0.287±0.003 | 0.007±0.000 | -0.029±0.007 | +0.008±0.013 | +0.051±0.009 |
| LGATr | 0.215±0.008 | 0.007±0.000 | -0.052±0.007 | +0.005±0.007 | +0.038±0.005 |
| LorentzNet | 0.273±0.002 | 0.007±0.000 | -0.009±0.008 | +0.029±0.006 | +0.022±0.009 |

_MLE bias = θ̂ − θ_true, where θ̂ is the minimum of each coefficient's averaged 1D marginal of the seed-averaged LLR (the same marginal the plots show), parabola-refined sub-grid; ± is the per-seed spread. `unconstr.` marks coefficients whose 68% marginal reaches the grid edge — the data does not bound them within the scan range._
