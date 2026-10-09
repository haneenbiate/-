# Actual validation error analysis

Rows: 2509. Errors: 589.
Accuracy: 0.765245; Macro-F1: 0.750428.

The following patterns describe observed predictions; they do not establish linguistic causes.

- NEG: support 923, recall 0.864572, F1 0.800803; errors to NEU: 92, POS: 33.
- NEU: support 1149, recall 0.709312, F1 0.765977; errors to NEG: 214, POS: 120.
- POS: support 437, recall 0.702517, F1 0.684504; errors to NEU: 72, NEG: 58.

See baseline_subgroups.json for dialect/sarcasm/negation and high-risk flag strata,
baseline_sensitivity.json for the high-risk-excluded subset, and validation_errors.csv
for original-text inspection. No automatic assertion about sarcasm, negation scope,
or dialect causing an error is made. Manual qualitative interpretation remains separate.
