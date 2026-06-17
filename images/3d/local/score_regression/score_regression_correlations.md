# Truth-vs-prediction correlations (ρ = mean ± std over 3 seeds)

| Model | ρ(c_Ht) | ρ(c_tW) | ρ(c_tB) | MSE | bias(c_Ht) | bias(c_tW) | bias(c_tB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2D Histogram | — | — | — | — | -0.195±0.018 | +0.010±0.009 | +0.082±0.010 |
| MLP | 0.336±0.002 | 0.020±0.001 | 0.013±0.003 | 0.022±0.000 | -0.001±0.006 | +0.025±0.013 | +0.087±0.004 |
| Transformer | 0.341±0.007 | 0.028±0.002 | 0.009±0.003 | 0.021±0.000 | -0.012±0.005 | -0.005±0.004 | +0.071±0.006 |
| Transformer (LLoCa) | 0.348±0.004 | 0.020±0.006 | 0.013±0.003 | 0.021±0.000 | -0.010±0.001 | +0.007±0.004 | +0.057±0.004 |
| LGATr | 0.315±0.003 | 0.017±0.001 | 0.009±0.002 | 0.020±0.000 | -0.035±0.005 | +0.010±0.006 | +0.029±0.004 |
| LorentzNet | 0.214±0.003 | 0.005±0.009 | 0.002±0.008 | 0.030±0.000 | -0.017±0.002 | +0.021±0.003 | +0.038±0.009 |

_MLE bias = θ̂ − θ_true, where θ̂ is the minimum of each coefficient's averaged 1D marginal of the seed-averaged LLR (the same marginal the plots show), parabola-refined sub-grid; ± is the per-seed spread. `unconstr.` marks coefficients whose 68% marginal reaches the grid edge — the data does not bound them within the scan range._
