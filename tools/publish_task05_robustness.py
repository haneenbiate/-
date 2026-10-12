"""Publish verified aggregate robustness results and standard PDF/SVG figures."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from nasn.task05 import fresh,registry,save,seal
from nasn.preflight import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['analysis','import-directory','tokenizer-directory','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    source=Path(args.analysis);registry(source)
    source_status=json.loads((source/'run_status.json').read_text())
    imported=Path(args.import_directory)
    provenance=json.loads((imported/'import_provenance.json').read_text())
    require(source_status['confirmed_n']==394 and source_status['human_review_completed'],
            'This publication describes the fixed finalized 394-variant review')
    require(provenance['source_archive_sha256']=='5cb2df58f092169b66dde3156a07cea025ba626293c234eab9a7ef0a6dbd14a5',
            'Publication source differs from the supplied completed Task 04 archive')
    output=fresh(args.output)
    allowed=['confirmed_robustness.csv','confirmed_robustness.tex','tatweel_summary.csv','tatweel_summary.tex',
             'tatweel_matched_intensity_metrics.csv','tatweel_matched_intensity_ids.csv',
             'task04_original_metrics.csv','evidence.json','run_status.json']
    for name in allowed:(output/name).write_bytes((source/name).read_bytes())
    (output/'archive_import_provenance.json').write_bytes((imported/'import_provenance.json').read_bytes())
    (output/'tokenizer_verification.json').write_bytes((Path(args.tokenizer_directory)/'verification.json').read_bytes())
    # Publish model evidence separately from source human decisions/rationales.
    review=Path('reports/task05-final-review');registry(review)
    human=pd.read_csv(review/'confirmed_evaluation_manifest.csv')
    ids=set(human.loc[human.eligible_confirmed,'variant_id'])
    changed=pd.read_csv(imported/'inference/perturbed_predictions.csv')
    clean=pd.read_csv(imported/'inference/clean_predictions.csv').set_index('source_row_id')
    selected=changed[changed.variant_id.isin(ids)].copy()
    require(len(selected)==394 and set(selected.variant_id)==ids,'Published model examples differ from confirmed review')
    fields=['variant_id','source_row_id','original_sha256','perturbed_sha256','gold_label','gold_id','rule','intensity',
            'predicted_id','predicted_label','confidence','probability_NEG','probability_NEU','probability_POS',
            'clean_id','clean_confidence','prediction_flipped']
    model=selected[fields].copy()
    for label in ['NEG','NEU','POS']:model['clean_probability_'+label]=model.source_row_id.map(clean['probability_'+label])
    model.to_csv(output/'confirmed_model_predictions.csv',index=False)
    token_details=pd.read_csv(source/'tatweel_paired_audit.csv')
    token_details.drop(columns=['left_token_ids','right_token_ids','left_full_token_ids','right_full_token_ids'],errors='ignore').to_csv(output/'tatweel_probability_differences.csv',index=False)
    table=pd.read_csv(source/'confirmed_robustness.csv')
    point=table[table.subset.eq('all_reviewed') & table.subgroup.eq('all')]
    rows=[]
    for (rule,intensity,n),part in point.groupby(['rule','intensity','n']):
        common={'rule':rule,'intensity':int(intensity),'n':int(n)}
        if not part.status.eq('computed').any():rows.append({**common,'status':'insufficient_sample_size'});continue
        indexed=part.set_index('metric')
        record={**common,'status':'computed'}
        for metric in ['clean_accuracy','perturbed_accuracy','clean_macro_f1','perturbed_macro_f1',
                       'macro_f1_drop','flip_rate','clean_NEG_f1','perturbed_NEG_f1',
                       'clean_NEU_f1','perturbed_NEU_f1','clean_POS_f1','perturbed_POS_f1']:
            for column in ['estimate','ci_low','ci_high']:record[metric+'_'+column]=float(indexed.loc[metric,column])
        rows.append(record)
    primary=pd.DataFrame(rows);primary.to_csv(output/'primary_confirmed_results.csv',index=False)
    (output/'primary_confirmed_results.tex').write_text(primary.to_latex(index=False,float_format='%.4f'))
    # Figures show within-cohort paired changes; rows use distinct review cohorts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scope=point[point.metric.eq('macro_f1_drop') & point.status.eq('computed')].sort_values(['rule','intensity'])
    require(((scope.ci_low<=0)&(scope.ci_high>=0)).all(),'Publication interval conclusion does not match results')
    matplotlib.rcParams['svg.hashsalt']='nasn-task05-robustness'
    matplotlib.rcParams['pdf.fonttype']=42
    matplotlib.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    y=np.arange(len(scope));values=scope.estimate.to_numpy()*100
    ax.errorbar(values,y,xerr=np.vstack([values-scope.ci_low.to_numpy()*100,scope.ci_high.to_numpy()*100-values]),
                fmt='o',capsize=4,color='#245a81')
    ax.axvline(0,color='#777777',linewidth=.8)
    ax.set_yticks(y,[f'{r.rule} {int(r.intensity)} (n={int(r.n)})' for r in scope.itertuples()])
    ax.set_xlabel('Clean minus perturbed Macro-F1 (percentage points)')
    ax.set_title('Confirmed review cohorts: paired 95% bootstrap intervals')
    fig.tight_layout()
    for extension in ['pdf','svg']:
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(output/f'confirmed_macro_f1_drop.{extension}',metadata=metadata)
    plt.close(fig)
    neg=table[table.subset.eq('all_reviewed') & table.subgroup.eq('negation_present') & table.metric.eq('flip_rate') & table.status.eq('computed')]
    combinations=sorted(set(zip(neg.rule,neg.intensity)))
    fig,ax=plt.subplots(figsize=(7.2,4))
    for label,color,offset in [('yes','#245a81',-.12),('no','#b45c22',.12)]:
        p=neg[neg.value.eq(label)].sort_values(['rule','intensity'])
        x=np.array([combinations.index((r.rule,r.intensity)) for r in p.itertuples()])+offset
        value=p.estimate.to_numpy()*100
        ax.errorbar(x,value,yerr=np.vstack([value-p.ci_low.to_numpy()*100,p.ci_high.to_numpy()*100-value]),
            fmt='o',capsize=3,color=color,label='Human negation '+label)
        for xpos,val,n in zip(x,value,p.n):ax.annotate(f'n={int(n)}',(xpos,val),xytext=(4,5),textcoords='offset points',fontsize=7)
    ax.set_xticks(range(len(combinations)),[f'{rule} {int(level)}' for rule,level in combinations])
    ax.set_ylabel('Prediction flip rate (%)');ax.set_title('Human negation strata: confirmed candidates')
    ax.legend();fig.tight_layout()
    for extension in ['pdf','svg']:
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(output/f'confirmed_negation_flip_rates.{extension}',metadata=metadata)
    plt.close(fig)
    (output/'methodology.md').write_text('''# Task 05 confirmed orthographic robustness results

## Evidence and execution

The supplied Task 04 ZIP reports a completed Tesla T4 inference run lasting
264.5467 seconds. All 39 registered payload hashes match; 40 original files
(including the checksum registry) are retained unchanged. A derived flat
inference registry references the exact exported bytes. No CSV is reserialized
during import. Task 02 source/manifests, all 2,509 clean prediction IDs/golds,
the selected epoch-2 checkpoint registry and paired probabilities are verified
against the completed Task 03B evidence. All 12,545 variant IDs/text hashes map
to deterministic original candidates: 9,961 applicable and 2,584 no-ops.
The recorded test checksum is metadata only; official test text is never read.

The model weight bytes are not in the archive or local workspace. Weight
verification/reload belongs to the supplied completed external run; this
analysis checks its recorded selected hashes and clean predictions. No local
model reload, new inference, fitting, checkpoint selection or test evaluation
is claimed. Selected model SHA-256:
`5c9fd87e2639a967d2c7517d5bb4189a6cbcb3bb477f9998e105682fcc249840`.

Four selected tokenizer files are independently byte-verified. The pinned
official small snapshot files were already available locally. Loading its
official tokenizer and applying the baseline's declared truncation=128 and
max-length padding restores its saved backend serialization state. The
synthetic probe sets that state only and is not research data. Every saved file,
including tokenizer.json, must match the selected checkpoint hashes exactly;
no alternative tokenizer is accepted. No model weights are loaded. All recorded
token counts, full/effective change flags and truncation flags reproduce across
all 12,545 candidates, independently checking exported tokenization evidence.

## Confirmed versus provisional populations

All 404 human assignments are resolved. Only 394 eligible reviewed variants are
confirmed (184 human-negation, 210 non-negation); 10 changed/ambiguous/uncertain
variants remain excluded. Human identities/blinding are user-declared, not
independently authenticated. Eligibility never propagates to other variants.
Each rule/intensity compares clean and perturbed predictions on its own matched
original IDs. Different rows have different review cohorts and cannot be read
as a causal intensity trend. The Alef cohort has n=24, below the predeclared
minimum of 30: metrics and intervals are explicitly unavailable. Original Task
04 provisional metrics remain byte-identical in task04_original_metrics.csv.

Seed-42 percentile 95% intervals use 2,000 paired bootstrap resamples by original
row ID within each stratum. Accuracy, fixed NEG/NEU/POS Macro-F1 and per-class F1,
clean-minus-perturbed drops and flip/error-transition rates are saved. Fixed
labels and absent-class F1=0 are preserved. Intervals condition on this checkpoint
and nonrandom stratified review sample, exclude training/reviewer uncertainty,
and are not simultaneous. Empirical zero-event bootstrap intervals can degenerate
to [0,0]; they do not prove zero risk in unseen text. Minimum n=30 is enforced
for reported metric strata. No primary Macro-F1 drop interval excludes zero;
the signed point changes do not establish a reliable direction of population
performance change. Individual prediction flips are directly observed.

## Negation and near-duplicate sensitivity

Human negation presence and whether it affects sentiment are separate subgroup
fields. The lexical cue heuristic is separately labeled and is not semantic
ground truth. Subgroup tables show counts and paired intervals; class/dialect
composition and small samples confound causal interpretations of group
differences. The frozen high-risk cross-split near-duplicate flags are used only
for a separately reported sensitivity subset; no source row, split or model is
changed, deleted or merged. Flagged pairs still require inspection. Ambiguous
semantic transformations are excluded from confirmed results, so provisional
instability remains distinct from confirmed meaning-preserving robustness.

## Tatweel intensity feasibility

Within all 2,474 applicable original IDs, intensity 1 and 2 have identical full
and effective token sequences and identical final predictions. One probability
vector differs at exact numeric precision, none differs above 1e-6. Clean-versus-
either intensity changes token sequences/probability vectors for 2,474 IDs and
final predictions for 176 IDs. These full-population counts remain provisional
because most candidates lack human review. Source token IDs/probability details
remain in ignored private analysis artifacts, not public raw-tweet exports.

The valid intensity comparison on human-confirmed examples uses the intersection
of both reviewed intensities: 35 original IDs, or 31 after high-risk exclusion.
Both comparisons observe zero prediction flips and zero Macro-F1 difference
between intensity 1 and 2. The separately sampled tatweel-1 and tatweel-2 cohorts
(94 and 93 variants) therefore must not be used to claim a dose effect. On this
benchmark the two character-level intensities do not yield different token-level
inputs. Tokenization associations are descriptive, not causal proof of errors.

All human/model evidence and original reports are preserved. No NASN method is
implemented. The remaining limits are small/nonrandom review cohorts, subjective
human scope judgments, unresolved near-duplicate risks and unavailable local
weight bytes. Results cover validation only and are not official test results.
''',encoding='utf-8')
    save(output/'publication.json',{'source_analysis_registry_sha256':sha256(source/'artifact_hashes.json'),
        'private_token_ids_and_probabilities_sha256':sha256(source/'tatweel_paired_audit.csv'),
        'publisher_sha256':sha256(Path(__file__)),'full_tweet_text_published':False,
        'token_ids_published':False,'model_weights_published':False})
    seal(output)
    print('Published confirmed robustness evidence:',output)


if __name__=='__main__':main()
