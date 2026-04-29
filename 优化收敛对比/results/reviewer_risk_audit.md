# Reviewer Risk Audit

- Overall residual risk: **low**
- Passed checks: 6/6

## Checks

- PASS `runs_count`: 30 independent runs detected.
- PASS `not_overclaiming_all_metrics`: Knowledge-constrained NSGA-II is best on 6/7 audited metrics.
- PASS `moead_tradeoff_not_hidden`: MOEA/D is worst on 3/3 key set-quality metrics, but shows local strength: top-2 F3-CTA, fastest mean convergence generation.
- PASS `key_statistics`: Holm-significant comparison rate on HV/IGD/PAM is 1.00.
- PASS `kcsr_not_core_claim`: KCSR has no method-level spread; report only as feasibility evidence.
- PASS `baseline_sensitivity_documented`: Selected baseline config file exists: True.

## Conservative Claim

Use a robust-overall claim centered on HV, IGD, PAM, convergence speed, and statistical evidence. Do not claim universal single-objective dominance.

## Average Rank Snapshot

| metric            |   Knowledge-constrained NSGA-II |   MOEA/D |   NSGA-II |   NSGA-III |
|:------------------|--------------------------------:|---------:|----------:|-----------:|
| Average_objective |                           1.967 |    3.000 |     2.400 |      2.633 |
| Final_F1_IV       |                           1.267 |    3.700 |     2.567 |      2.467 |
| Final_F2_DEG      |                           1.300 |    3.967 |     2.333 |      2.400 |
| Final_F3_CTA      |                           2.667 |    2.433 |     2.367 |      2.533 |
| HV                |                           1.050 |    3.600 |     2.600 |      2.750 |
| IGD               |                           1.133 |    3.567 |     2.767 |      2.533 |
| PAM               |                           1.000 |    3.733 |     2.567 |      2.700 |
