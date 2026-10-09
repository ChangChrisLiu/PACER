# Explicit diagnostic-profile compatibility

The default calculation is the configurable direct component-balanced mean in
[Scoring configuration](SCORING_CONFIGURATION.md). Both
`component_balanced_j_val` and `robust_profile_j_val` without a profile use that
route. Whole-chunk trajectory alignment is the construction default.

Previously named factor-family and component-role lower-quartile profiles remain
available for explicit replay through `scorer_profile(name)` and
`robust_profile_j_val(rows, profile=name)`. Their numerical definitions are
preserved; they are not the default selector. Hierarchical weighting and direct
weighting are not equivalent when metrics are missing. A profile carries its
own contract and cannot be mixed with direct `ScoringConfig` overrides.

Choose a diagnostic only when its interpretation belongs to your declared
protocol. Record its name, inputs, applicability and aggregation in the manifest.
Do not treat its absolute scalar as interchangeable with a component-mean score.
No diagnostic replaces held-out task success or full candidate feasibility.
