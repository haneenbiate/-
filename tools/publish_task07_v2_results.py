"""Publish verified completed v2 evidence, tables and paired-contrast figures."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from nasn.task05 import fresh,registry,save,seal
from nasn.task04 import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    source=Path(args.analysis);registry(source)
    evidence=read(source/'verification.json')
    if evidence['status']!='verified_completed_v2_results' or not evidence['all_metric_tables_and_decision_recomputed']:
        raise ValueError('Completed and independently recomputed results required')
    if evidence['archive_sha256']!='6306331ae954f9778bec80d9ee2e1d55fa41744b29221b4162402ab52a0f286f':
        raise ValueError('This publication describes the fixed supplied October 14 experiment; create a new report for another run')
    # Verify the original immutable payloads again before publishing them.
    for name,digest in evidence['original_payload_sha256'].items():
        if sha256(source/'original'/name)!=digest:raise ValueError(f'Original report bytes changed: {name}')
    output=fresh(args.output)
    shutil.copytree(source/'original',output/'original')
    for file in source.iterdir():
        if file.is_file() and file.name!='artifact_hashes.json':shutil.copyfile(file,output/file.name)
    result=pd.read_csv(source/'primary_and_provisional_results.csv')
    result.to_latex(output/'primary_and_provisional_results.tex',index=False,float_format='%.4f',escape=True)
    clean=pd.read_csv(source/'all_clean_metrics.csv').set_index('strategy')
    primary=result[result.source_cohort.eq('source_human_confirmed')].copy()
    for metric in ['accuracy','macro_f1']:primary['all_clean_'+metric]=primary.strategy.map(clean[metric])
    fields=['strategy','n_variants','n_source_ids','all_clean_accuracy','all_clean_macro_f1',
        'perturbed_accuracy_estimate','perturbed_macro_f1_estimate','perturbed_macro_f1_ci_low',
        'perturbed_macro_f1_ci_high','flip_rate_estimate','flip_rate_ci_low','flip_rate_ci_high']
    primary[fields].to_csv(output/'primary_comparison_table.csv',index=False)
    primary[fields].to_latex(output/'primary_comparison_table.tex',index=False,float_format='%.4f',escape=True)
    contrasts=pd.read_csv(source/'recomputed_paired_contrasts.csv')
    scope=contrasts[contrasts.cohort.eq('source_human_confirmed')&contrasts.metric.eq('perturbed_macro_f1')&contrasts.reference.ne('RAW')].copy()
    scope=scope.set_index('reference').loc[['TAT','REP2','NO_CONTEXT']].reset_index()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'task07-v2-verified'})
    fig,ax=plt.subplots(figsize=(7,3.5));point=scope.estimate.to_numpy()*100;y=np.arange(len(scope))
    ax.errorbar(point,y,xerr=np.vstack([point-scope.ci_low.to_numpy()*100,scope.ci_high.to_numpy()*100-point]),fmt='o',capsize=4,color='#245a81')
    ax.axvline(0,color='#777777',linewidth=.8);ax.set_yticks(y,scope.reference)
    ax.set_xlabel('NASN2 minus simple baseline perturbed Macro-F1 (percentage points)')
    ax.set_title('394 source-eligible variants: paired original-ID cluster 95% intervals')
    fig.tight_layout()
    for extension in ['pdf','svg']:
        meta={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(output/f'NASN2_vs_simple_macro_f1.{extension}',metadata=meta)
    plt.close(fig)
    (output/'methodology.md').write_text('''# Task 07 completed v2 results

All 26 supplied light-report payload hashes and 62,350 prediction identities match
the frozen Task 07 code, source manifests, rules and selected checkpoint registry.
Every per-rule metric, contrast, original-ID cluster interval, all-clean score and
paired interval, and scientific decision is recomputed from actual saved outputs.
Original uploaded bytes are preserved under original/. The recorded external run
used Tesla T4, CUDA 12.6, Torch 2.8.0+cu126, seed 42, float32, length 128 and batch
8; total recorded run duration is 510.8631 seconds. No local model inference or
weight reload is claimed. Selected checkpoint verification is external recorded
load/hash evidence corroborated by RAW clean baseline reproduction.

The primary cohort contains 394 user-declared eligible source variants across
274 original IDs. New normalized text is not human-validated and reviewer
independence remains unverified. These are conditional exploratory scores against
original sentiment labels, not a certified meaning-preserving robustness benchmark.
9,557 unreviewed variants are reported separately; 10 reviewed exclusions have no
robustness metrics. All-clean table columns use 2,509 unique development rows;
paired pooled clean metrics use matched variant weighting, explicitly separately.

NASN2 and strongest simple baseline NO_CONTEXT both have primary perturbed Macro-F1
0.655111 and clean/perturbed flip rate 7/394 (1.7766%). Their perturbed predictions
are identical across all 394 primary rows, despite 11 different effective input
sequences. Paired NASN2 gain over NO_CONTEXT is 0, with empirical interval [0,0].
This is no observed incremental classification benefit from the bounded context
guard in this cohort; a degenerate empirical interval does not prove zero effect
in unseen text. The baseline shares marker/critical-word safety; this ablation
tests the added context guard, not the value of all negation protection.

Against TAT and REP2, NASN2 point gains are 0.008878 and 0.005932; their paired
95% intervals [-0.002942,0.023442] and [-0.002208,0.016312] include zero. NASN2's
point all-clean Macro-F1 is slightly higher than NO_CONTEXT (0.751225 vs 0.750904),
corresponding to one changed clean prediction. This does not satisfy the frozen
positive perturbed-gain requirement. RAW primary Macro-F1 is 0.640226; the combined
simple method already provides the same primary gain as NASN2. It cannot be
attributed uniquely to context awareness or treated as a novelty claim.

The predetermined gate fails. Recommend abandoning the NASN2 superiority claim or
reframing as a transparent Arabic preprocessing/ablation audit. Do not add post-hoc
rules, alter the selected checkpoint or reinterpret provisional results as verified
semantics. Retain old preparation/blocker reports as historical stages. New semantic
reviews and provenance evidence remain required for meaning-preservation claims,
but are not needed to observe failure of this performance gate on current labels.

Intervals use 2,000 seed-42 paired resamples. Pooled observations move together by
original source ID; per-rule strata use one variant per ID and minimum n=30.
They condition on one trained checkpoint and a nonrandom development review sample,
omit training/reviewer uncertainty, and are not simultaneous significance tests.
Exact-text duplicates with separate IDs and prior validation reuse further limit
inference. No official test text/labels are read/evaluated, no model is trained,
and no normalization rule or decision threshold is changed.

Reproduce using `python -m nasn.task07_v2_results --archive /path/to/original.zip
--output runs/task07-v2-verified-new`, then `python -m tools.publish_task07_v2_results
--analysis runs/task07-v2-verified-new --output runs/task07-v2-public-new`.
Use the pinned environment and MPLBACKEND=Agg as documented in
[the complete results report](../../docs/task07_v2_results.md).
''',encoding='utf-8')
    save(output/'publication.json',{'verified_analysis_registry_sha256':sha256(source/'artifact_hashes.json'),
        'original_archive_sha256':evidence['archive_sha256'],'publisher_sha256':sha256(Path(__file__)),
        'rules_or_decision_thresholds_changed':False,'raw_text_or_weights_published':False})
    seal(output)
    print('Published verified completed v2 results:',output)


if __name__=='__main__':main()
