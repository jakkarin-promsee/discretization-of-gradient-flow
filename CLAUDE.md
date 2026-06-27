# CLAUDE.md

Guidance for working in this repository.

## What this project is

An educational study that **discretizes gradient flow** and compares three optimizers
on a regression problem, implemented from scratch in **pure NumPy** (no autograd, no
PyTorch/TF). Originally a year-2 linear algebra final project; now being revived as a
portfolio piece.

The three (four) optimizers:

| Optimizer | Update rule | Class |
| --- | --- | --- |
| Explicit GD (1st-order / forward Euler) | $\theta_{k+1} = \theta_k - \eta\nabla L$ | `ExplicitLrModel` |
| Implicit GD (backward Euler on quadratic) | $\theta_{k+1} = (I+\eta H)^{-1}(\theta_k + \eta(H\theta_k-\nabla L))$ | `ImplicitLrModel` |
| Newton GD (jump to quadratic vertex) | $\theta_{k+1} = \theta_k - H^{-1}\nabla L$ | `NewtonLrModel` |
| Newton GD + damping + Armijo line search | as above, stabilized | `OptimizeNewtonLrModel` |

The README is the project report: it contains the experiment results and the linear-algebra
derivations (Taylor expansion → each discretization, plus Hessian properties: symmetry,
quadratic form, eigen-decomposition, orthogonality).

## Layout

```
custom_library/            # A mini Keras-like DL framework, built on NumPy
  core/
    baseLrModels.py                  # BaseLrModel: layer mgmt, add(), predict(), save/load, rollback
    baseLrModelsImplementHessain.py  # adds flatten/unflatten params, compute_loss/gradient/hessian
  layers/
    denseLayer.py                    # DenseLayer (He init); W, b, params_count()
    activationLayer.py               # ActivationLayer (ReLU only) forward()/backward()
  models/                            # one fit() loop per optimizer (the "strategies")
    explicitLrModel.py  implicitLrModel.py  newtonLrModel.py  optimizeNewtonLrModel.py
  utils/
    history.py                       # History: metric log, best-model store, plotting
    metrics.py                       # mae/mse/rmse/r2_score/accuracy
    dataUtils.py                     # split_data(), sort_data()
main.ipynb                 # Experiment lab (88 cells). Cells 0-44 = main study; 45+ = scratch/"playground"
model_saves/               # Pickled trained models + datasets (timestamped)
docs-src/                  # Images referenced by README (latex renders, plots)
temp/                      # Draft READMEs (Thai/English) + rendered PDF — working scratch
```

## Architecture notes / conventions

- **Inheritance:** `ExplicitLrModel` → `BaseLrModel`. The Hessian-based models
  (`Implicit`, `Newton`, `OptimizeNewton`) → `BaseLrModelsImplementHessain` → `BaseLrModel`.
- **Params as one flat vector:** Hessian math needs `theta` as a 1-D vector.
  `flatten_params()`/`unflatten_params()` convert between the layer list and that vector;
  `shapes` records how to reassemble. ReLU/activation layers are skipped (no params).
- **Hessian is numerical, not analytic:** `compute_hessian()` does a **central finite
  difference of the analytic gradient** — for an N-param model it runs `2N` gradient
  evaluations and deep-copies all layers each step, producing an `N×N` matrix. Correctness
  of the Hessian therefore depends entirely on `compute_gradient()` (hand-coded backprop).
- **Loss vs. reported metric:** training minimizes **MSE**, but everything logged/printed
  (`batch`, `acc`, `val`) is **MAE**. Note `acc`/`accuracy` here means error, not accuracy —
  lower is better.
- **`intro_navigate`:** Implicit/OptimizeNewton can take a couple of plain-GD warm-up steps
  first (`introUsingNormalGD = 2`) to move off a bad start before the curvature step.
- **History tracking:** `fit()` records train + eval metrics **every batch** (not every
  epoch) and re-predicts the full train and eval sets each batch.
- **Naming:** "Hessain" (should be Hessian) and "Sysmetric" appear in filenames, class names,
  and identifiers throughout — they are load-bearing, so renaming is a deliberate refactor.

## Running it

There is no environment file yet. Dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`
(notebook), plus Jupyter. The notebook expects to run from the repo root (it appends
`os.getcwd()` to `sys.path` and imports `custom_library`). Typical flow inside `main.ipynb`:
build a `baseModel`, `copylayers()` it into each optimizer so they start from identical
weights, `fit()` each, then compare via `History.evaluate()` / `display_trend()`.

## Gotchas to be aware of when editing

- `History.save_model()` stores a **reference** to the live layer list, not a copy, so the
  "best" weights get mutated by later in-place updates — best-model rollback is unreliable.
- `metrics.accuracy()` computes an accuracy value but `return`s the MAE instead.
- The `fit()` loop body (forward/backward, predict, evaluate, save, progress bar) is
  copy-pasted across all four model files; the forward pass exists in four places
  (`predict`, `compute_loss`, `compute_gradient`, `ExplicitLrModel.fit`).
- No random seed is set, so experiment runs are not reproducible.
- `__pycache__/` and the `.pkl` artifacts are currently committed (no `.gitignore`).
