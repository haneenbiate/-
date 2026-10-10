# Task 04 Colab analysis recovery

`CalledProcessError` with exit status 2 identifies a failed CLI command; it does
not identify the validation or dependency error. The line containing `call(...)`
is also the invocation, not the cause. Look for `ERROR:` or inspect captured stderr.
The subsequently supplied 2026-10-10 stderr identifies an inherited Colab
`MPLBACKEND=module://matplotlib_inline.backend_inline` setting. The isolated
CLI environment cannot resolve that notebook backend. Set `ENV['MPLBACKEND'] =
'Agg'` before retrying the analysis cell; this backend writes the same PDF/SVG
figures without requiring a notebook display. The updated notebook explicitly
sets this variable for subprocesses. This changes rendering configuration only.
The same saved-report analysis succeeds locally.

Run this **new diagnostic cell** in the existing Colab session. It uses a fresh
output directory and preserves your earlier outputs, benchmark and checkpoint:

```python
import subprocess, uuid
ENV['MPLBACKEND'] = 'Agg'
diagnostic = ROOT / ('analysis-diagnostic-' + uuid.uuid4().hex[:8])
result = subprocess.run(
    [str(PY), '-m', 'nasn.task04', 'analyze', '--run', str(BASELINE_REPORTS),
     '--output', str(diagnostic)],
    cwd=str(REPO), env=ENV, capture_output=True, text=True,
)
print('Exit code:', result.returncode)
print(result.stdout)
print(result.stderr)
print('Diagnostic output:', diagnostic)
```

If exit code is nonzero, retain the full `ERROR:` message. Do not skip hash checks
or inference verification, substitute model weights, or rerun training. If it is
zero, this analysis attempt completed; that still does not establish completion
of perturbed inference. An original failure may have left an output directory,
and retrying the same path fails explicitly with `Output exists`.

The updated notebook separates preparation and saved-prediction analysis into
independent cells. Re-running the **analysis cell only** selects a new output
directory when needed and exports that actual completed path. Failed commands
print captured stderr and preserve `command-failure-*.json` in Drive. The CLI
also writes `run_status.json` with the failure reason once an analysis output
directory has been created.

A separate portability defect was found during inspection: subgroup reports used
exact floating-point equality, unlike the other metric checks. Different CPU
reductions can yield last-bit differences without changing labels or counts.
Float metrics now use the existing 1e-12 absolute verification tolerance;
structure, counts, labels and status remain exact. NaN, substantial metric
differences, changed counts and changed labels are still rejected. Regression
tests reproduce this defect and verify those rejection cases. This is a verified
software fix, not proof that it caused the user's Colab failure.

No baseline training code, data split, perturbation rule, selected checkpoint or
saved experimental report is changed by this recovery patch. The updated
notebook remains unexecuted here; Colab completion must be evidenced by its actual
run outputs.
