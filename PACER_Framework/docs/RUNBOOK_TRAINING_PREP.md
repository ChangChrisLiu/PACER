# Runbook 2 — Build the training datasets/views and train candidates (pipeline steps 4–5)

Input: a validated, split post-training buffer (Runbook 1).
Output: **eight trainer-ready datasets/views** — the PACER-weighted view(s)
plus seven baselines — each exported as `rows.jsonl` + `manifest.json` for
your own training stack.

## 1. Declare the candidate set

Use the built-in candidate pool as-is, or declare your own finite pool BEFORE
any validation scoring exists:

```python
from pacer_framework import CANDIDATE_POOL
candidates = dict(CANDIDATE_POOL)              # built-in candidate set
# Optional: declare additional eta configs in code before scoring, then validate
# them with the eta module's bounds helpers.
```

Eight free fields are searched; the nine fixed protocol fields
(β_dir = β_prov = 0, β_op = 0.05, γ_clean = γ_auto_success = 1,
γ_failure = γ_excluded = 0, f_clean = 0.80, f_corr = 0.65) stay fixed.

## 2. Compute evidence, gates, scores, and weights per candidate

```python
from pacer_framework import compile_weights
weighted_rows, manifest = compile_weights(split_rows, eta, eta_id="terminal_stop_heavy")
```

What happens inside (all framework-defined):
- evidence `e_k(i)` over {prog, prox, term, dir, stop, op, prov} from each
  row's geometry/labels, with the target-consistency factor T_i zeroing the
  geometric four on missing/inconsistent target metadata (eq:app_geom);
- the eligibility gate g_i = G_split·G_mask·G_prov·G_safe·G_role (eq:app_gate);
- process score s_η = Σ β_k e_k, robust stratum centering over
  (component, phase, role) with median / 1.4826·MAD and the
  (c,φ,ρ)→(c,φ)→global fallback (eq:app_center, eq:app_fallback);
- the row weight w_i = g_i·clip(max(f_ρ, γ_ρ·exp((s−b)/(τσ+ε))), 0, w_max)
  (eq:weight).

**Pre-training audit (do not skip):** on every manifest check
`forbidden_positive_weight_rows == 0`, `loss_normalizer > 0`,
`weight_max <= eta.w_max`, and a non-collapsed `n_eff_horizons`. These are the
A_audit inputs of the feasibility predicate; `evaluate_candidate` re-checks
them after training.

## 3. Build all eight datasets/views

```python
from pacer_framework import BASELINE_MODES, compile_baseline, export_training_view

for mode in BASELINE_MODES:            # built-in baseline/training views
    if mode == "pacer_selected":
        continue                       # candidates are exported in step 2's loop
    rows_m, manifest_m = compile_baseline(split_rows, mode)
    export_training_view(rows_m, manifest_m, out_dir=f"views/{mode}")
```

| View | Rows used | Weight |
|---|---|---|
| `pacer_selected` | gate-eligible | `w_i(η)` from step 2 (one view per candidate; the selected one is decided in Runbook 3) |
| `vanilla_post_sft` | clean + correction | 1 (no PACER gate; shared split/mask/safety filters) |
| `uniform_all_eligible` | all gate-eligible | 1 |
| `outcome_only` | clean + auto_success | 1 |
| `clean_only` | supplementary clean | 1 |
| `correction_only` | correction | 1 |
| `fixed_geometry` | gate-eligible | `g_i·max(0, rank_weight)` (needs `rank_weight` on rows) |
| `no_post_train` | none | — (the reference policy itself; nothing to train) |

Every exported `rows.jsonl` row carries a top-level `loss_weight`; the
manifest reports the view's accounting (positively weighted rows, valid
horizons, nominal-start rows).

## 4. Train the candidate VLAs externally (pipeline step 5)

In YOUR training stack — any VLA whose action head exposes a per-horizon
supervised loss — implement the weighted objective. You may wrap that training
entrypoint with `TrainerHooks.train_weighted_view(...)`, but PACER does not
require you to import the hook at runtime; the exported files are plain JSONL +
JSON.

```
loss = sum_i sum_h loss_weight_i * mask_ih * native_action_loss(i, h)
       / max(sum_i sum_h loss_weight_i * mask_ih, eps)
```

Hold everything else identical across every candidate and baseline: the same
starting checkpoint π_θ0, the same adapter recipe (LoRA or full), the same
optimizer and update budget, uniform minibatch sampling (weights act only
through the loss multiplier), and your normalization statistics resolved per
view manifest. Train one fresh adapter per PACER candidate and one per trained
baseline. The only PACER-vs-baseline asymmetry allowed by the protocol is that
PACER trains its M candidates and selects one on validation.

Next: `docs/RUNBOOK_EVALUATION.md` (steps 6–7).
