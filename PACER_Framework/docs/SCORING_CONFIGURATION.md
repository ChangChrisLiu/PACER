# Configurable validation scoring

The default is a direct weighted row score followed by a component-balanced
mean. `component_balanced_j_val`, `evaluate_candidate`, the default
`robust_profile_j_val` entry point and the runtime row-scoring facade share this
calculation. Whole-chunk trajectory alignment is the construction default.

Coefficients are application configuration, not universal optimal settings or
training parameters. Export the machine-readable starting configuration and
adapt it to your own validation protocol; no application-specific recipe is
required by this interface.

## Run immediately

Install either the minimal framework distribution or the full runtime
distribution, not both in one environment. The full distribution includes the
same framework source.

```bash
python -m pacer_framework.check_setup
python -m pacer_framework.demo --out-dir .local/demo-default
python -m pacer_framework.scoring --write-config .local/scoring.json
# Edit the generated configuration for your application.
python -m pacer_framework.demo --scoring-config .local/scoring.json --out-dir .local/demo-custom
```

The demo uses synthetic records and counts. It checks contracts, splits data,
exports training views, scores candidate and reference predictions, runs audits
and selects a candidate. It does not train a policy or connect to hardware.
`evaluation.json` records the configuration and its fingerprint.

For your own precomputed validation rows:

```bash
python -m pacer_framework.scoring --rows validation_rows.jsonl --config .local/scoring.json --output .local/scores.json
```

This command computes scores, not complete candidate feasibility. Existing
configuration/score files are not replaced unless `--force` is given. Keep
application-specific settings and outputs outside version control; `.local/`
is ignored by this repository.

## Configuration API

```python
from pacer_framework import ScoringConfig

config = ScoringConfig()
config.save("scoring.json")
config = ScoringConfig.load("scoring.json")
metadata = config.to_dict()
protocol_id = config.fingerprint
```

Fields:

- `submetric_weights`: a finite nonnegative map over supported validation terms;
- `reference_alignment_mode`: trajectory mode or explicit endpoint mode;
- `delta_reg`: the shared no-regression margin;
- `require_reference_component_scores`: reference auditing, required by default.

A supplied coefficient map replaces the default map. Export the complete map
before editing if other entries should remain unchanged. A positive coefficient
is required; active no-regression also requires active geometric evidence.
Unknown fields, names, ambiguous aliases and invalid types fail loudly.
Configuration mappings are copied and immutable; construct a new configuration
to change the protocol.

Existing `submetric_weights` arguments remain available. Pass a complete config
or individual overrides, not both. JSON inputs remain editable: the fingerprint
is computed after loading and stored in outputs, not trusted as an input field.
It records exact supplied settings, not normalized equivalence classes. Also
record the software version and validation-row manifest for reproducibility.

## Metric contract

The standard terms are progress, proximity, terminal, direction, align and
no-regression, with stop/handoff included on applicable rows.

- Progress compares initial distance with the minimum attained along the chunk.
- Proximity evaluates endpoint distance to the declared target.
- Terminal checks the declared region and pose predicate, not physical success.
- Direction compares net EEF/TCP displacement with the start-to-target vector.
- Align compares concatenated adjacent EEF/TCP displacements with a matched
  reference trajectory. Endpoint alignment remains an explicit alternative.
- No-regression compares candidate and baseline-policy geometry under the same
  coefficients and margin.

Unavailable terms are omitted and remaining coefficients normalize per row;
observed zeros are retained. A short matched trajectory is an error, not a reason
to silently switch representation or remove alignment. Intentionally reduced
diagnostics have a different interpretation from the complete protocol.

Blocked positive-role rows score zero and stay in the component denominator.
Failure/excluded rows are audit-only. The score averages positive rows within
components, then components equally. It is not a success probability.

## Reference data and consistent rescoring

The matched trajectory used for align and the baseline policy used for
no-regression are different reference objects. Keep their sources explicit.

Store baseline geometry in `reference_submetrics` when coefficients may change:

```python
from pacer_framework import open_loop_submetrics, component_balanced_j_val

candidate_metrics = open_loop_submetrics(
    predicted_positions, target_point=target_point,
    reference_positions=matched_reference_positions, scoring_config=config,
)
baseline_metrics = open_loop_submetrics(
    baseline_positions, target_point=target_point, scoring_config=config,
)
row = {
    "row_id": observation_id, "component": component, "role": role,
    "submetrics": candidate_metrics,
    "reference_submetrics": baseline_metrics,
    "reference_alignment_mode": config.reference_alignment_mode,
    **blocker_flags,
}
score, scored_rows, manifest = component_balanced_j_val([row], scoring_config=config)
```

Choose and freeze the comparison margin for your numerical precision before
ranking; do not adapt it after inspecting outcomes.

The scorer recomputes no-regression from matching nonempty active geometric
support, replacing a stale cached indicator without mutating inputs.
`with_no_regression(..., scoring_config=config)` is also available.

Cached-only indicators remain a compatibility input. Manifests distinguish
provided, recomputed and disagreeing indicators, including cached values used
under custom weights. Without reference data, the scorer cannot reconstruct or
certify the original comparison. Changing alignment mode requires raw traces:
a conflicting declared cache mode fails, while an undeclared scalar does not
prove how alignment was constructed; manifests count such unlabelled scoring
rows.

## Audited candidate evaluation

```python
from pacer_framework import evaluate_candidate, select_candidate

result = evaluate_candidate(
    validation_rows=candidate_rows,
    reference_validation_rows=baseline_rows,
    weight_manifest=training_weight_manifest,
    w_max=eta.w_max,
    scoring_config=config,
)
selected = select_candidate(
    {candidate_id: result["j_val"]},
    {candidate_id: result["feasible"]},
)
```

Candidate and baseline rows must share stable positive-row identities,
components and roles. Both sides are rescored with the same configuration.
Matched baseline geometry determines candidate no-regression; the baseline is
compared to itself. Missing reference component scores fail closed by default.
Conflicting embedded and matched baseline geometry is rejected; supply one
authoritative reference representation or matching active geometric values.
Precomputed `reference_component_scores` remain an alternative caller-owned
same-protocol input; do not pass both reference forms.

Full audits require explicit boolean blocker inputs on candidate and supplied
baseline rows; missing flags are not observations of safe behavior.
Training-view/ESS audits, wrong-target and unsafe checks, scoring support,
component coverage and component-level non-regression remain required. An
explicit reference-audit opt-out is for labelled diagnostics, not a complete
selection audit; `A_reg_reference_audit_skipped` marks it explicitly. No feasible candidate means reference-policy fallback.

Freeze one protocol for all candidates before protected evaluation. Scoring
configuration does not change training eta, masks, losses or model checkpoints.
Named factor-family/q25 diagnostics remain explicit compatibility opt-ins;
see [diagnostic profiles](TRAJECTORY_ROBUST_SCORING.md) and the release notes.
