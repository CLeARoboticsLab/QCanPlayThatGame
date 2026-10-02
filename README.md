# LearningValueFunctions

Fitted Q-iteration for a two-player zero-sum game. The Q-function is parameterized as

```
Q(x,u,v) = A(x,u) - B(x,v) + u' C(x) v
```

with partially input-convex networks (PICNN) for `A` and `B`, and a small MLP (`MatrixNet`) for `C(x)`. Each iteration samples one-step transitions, solves the inner stage game with SeCoND + Clarabel, and takes a semi-gradient TD update.

The included example is **`non_lq`**: two stacked planar double integrators (`x ∈ ℝ⁸`, `u,v ∈ ℝ²`) with a nonlinear (exponential) separation stage cost.

Start a threaded REPL from the repository root with `julia -t auto --project=.`.

## Setup

Requires Julia 1.11 or newer.

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Run a single training job

Edit `CONFIG` in `experiments/learn_q_lq_game.jl` (seed, `num_samples_per_iter`, `num_iters`, `γ`, `picnn_wz_softplus`, `picnn_z_gate_softplus`, …), then:

```julia
include("experiments/learn_q_lq_game.jl")
learn_q_lq()   # uses CONFIG
learn_q_lq(; cfg = merge(CONFIG, (;
    rng_seed = 1,
    num_samples_per_iter = 100,
    picnn_wz_softplus = true,
    picnn_z_gate_softplus = true,
)))
```

## Sweep seeds and sample sizes

`experiments/run_learn_q_lq_ns_gs_seed_sweep.jl` runs the same trainer over

- seeds `1:S` (default `S = 3`)
- `num_samples_per_iter` in `{100, 200, 300, 400, 500}`
- `num_grad_steps = 1`

```julia
include("experiments/run_learn_q_lq_ns_gs_seed_sweep.jl")
run_learn_q_lq_ns_gs_seed_sweep(; S = 2, num_iters = 100, picnn_wz_softplus = true, picnn_z_gate_softplus = true)
run_learn_q_lq_ns_gs_seed_sweep(; seeds = 1:3, ns_values = (100,), picnn_wz_softplus = true, picnn_z_gate_softplus = true)
```

## Metrics

Each run writes a CSV under `runs/` (gitignored), named like

`runs/non_lq_seed1_ns100_gs1_gamma0.95_<timestamp>.csv`

Columns include iteration, reward-normalized TD (`td_mean_abs`, `td_rms`), post-update `td_loss`, control norms, and SeCoND iteration counts. Set `csv_enable = false` to disable, or `csv_path` to pick a file. Progress is also printed every `log_every` iterations (default 10).

`td_mean_abs` and `td_rms` are pre-update residuals divided by `(r_max - r_min) / (1 - γ)`. With `td_clip_q_targets = true`, both `Q(x,u,v)` and the bootstrap `Q(x',·)` are clipped to `[r_min, r_max] / (1 - γ)`.
