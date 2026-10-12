"""Publish text-free finalized human-review tables, without model inference."""
import argparse
import json
from pathlib import Path

import pandas as pd

from arsarcasm_audit.audit import sha256
from nasn.preflight import require
from nasn.task05 import csv,registry,seal,save,fresh,DECISIONS,agreement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['intake','review-package','llm-audit','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    intake,package,llm_audit=map(Path,[args.intake,args.review_package,args.llm_audit])
    registry(intake);registry(package);registry(llm_audit);registry(intake/'decisions')
    evidence=json.loads((intake/'intake_evidence.json').read_text())
    require(evidence['resolved_decisions_registry_sha256']==sha256(intake/'decisions'/'artifact_hashes.json'),
            'Final decision registry differs from intake evidence')
    require(evidence['review_package_registry_sha256']==sha256(package/'artifact_hashes.json'),
            'Review package differs from final intake')
    status=json.loads((intake/'run_status.json').read_text())
    require(status['human_review_completed'] and status['resolved_n']==status['queue_n']==404 and status['unresolved_disagreements']==0,
            'Adjudication incomplete')
    f=csv(intake/'decisions'/'resolved_decisions.csv')
    require(not f.variant_id.duplicated().any() and len(f)==404,'Final decision IDs invalid')
    metadata=csv(package/'queue_registry.csv')
    require(set(f.variant_id)==set(metadata.variant_id),'Final IDs differ from original queue')
    data=metadata.merge(f,on='variant_id',validate='one_to_one')
    eligible=data.eligible_confirmed.eq('True')
    require(int(eligible.sum())==status['confirmed_n'],'Confirmed count mismatch')
    output=fresh(args.output)
    # No rationale, reviewer identities or full text is published.
    columns=['variant_id','source_row_id','original_sha256','perturbed_sha256','rule','intensity',
             'eligible_confirmed','negation_present','negation_affects_sentiment','high_risk_near_duplicate']
    data[columns].to_csv(output/'confirmed_evaluation_manifest.csv',index=False)
    distribution=[]
    for field in DECISIONS:
        for value,n in f[field].value_counts().items():distribution.append({'field':field,'value':value,'n':int(n)})
    pd.DataFrame(distribution).to_csv(output/'decision_distributions.csv',index=False)
    data.loc[~eligible].groupby(['semantic_validity','sentiment_preserved','negation_present','negation_affects_sentiment']).size().rename('n').reset_index().to_csv(output/'excluded_categories.csv',index=False)
    rows=[]
    for subset,part in [('all_confirmed',data[eligible]),('exclude_high_risk',data[eligible & data.high_risk_near_duplicate.eq('False')])]:
        for (rule,intensity,negation,effect),group in part.groupby(['rule','intensity','negation_present','negation_affects_sentiment']):
            rows.append({'subset':subset,'rule':rule,'intensity':int(intensity),'human_negation_present':negation,
                         'human_negation_affects_sentiment':effect,'variants_n':len(group),'unique_original_ids':group.source_row_id.nunique()})
    table=pd.DataFrame(rows);table.to_csv(output/'confirmed_sample_sizes.csv',index=False)
    (output/'confirmed_sample_sizes.tex').write_text(table.to_latex(index=False))
    pd.DataFrame([{'n_variants':int(eligible.sum()),'n_unique_original_ids':data[eligible].source_row_id.nunique(),
                  'human_negation_yes':int((eligible & data.negation_present.eq('yes')).sum()),
                  'human_negation_no':int((eligible & data.negation_present.eq('no')).sum()),
                  'excluded_variants':int((~eligible).sum())}]).to_csv(output/'review_summary.csv',index=False)
    llm=csv(llm_audit/'submitted_llm.csv')
    require(set(llm.variant_id)==set(f.variant_id),'LLM assignments differ')
    _,llm_agreement,_=agreement(f,llm,404)
    llm_table=pd.DataFrame(llm_agreement)
    llm_table['interpretation']='agreement_with_user_declared_adjudicated_human_reference; not MARBERT robustness'
    llm_table.to_csv(output/'llm_vs_final_human_agreement.csv',index=False)
    (output/'llm_vs_final_human_agreement.tex').write_text(llm_table.drop(columns='interpretation').to_latex(index=False,float_format='%.4f'))
    for name in ['agreement.csv','coverage_by_stratum.csv','review_coverage.json']:
        (output/name).write_bytes((intake/'decisions'/name).read_bytes())
    status.update(status='human_review_complete_model_predictions_pending',
        confirmed_human_negation_yes=int((eligible & data.negation_present.eq('yes')).sum()),
        confirmed_human_negation_no=int((eligible & data.negation_present.eq('no')).sum()),
        excluded_variants=int((~eligible).sum()),robustness_metrics_computed=False,
        task04_perturbed_predictions_available_locally=False,
        official_test_text_read=False,checkpoint_loaded=False)
    save(output/'run_status.json',status)
    save(output/'evidence.json',{'source_workbook_sha256':evidence['source_workbook_sha256'],
        'timestamp_timezone':evidence['timestamp_timezone'],'timestamp_timezone_basis':'User explicitly confirmed Baghdad time on 2026-10-12',
        'human_source':'user_declared','independent_identity_and_blinding_verification':False,
        'root_intake_registry_sha256':sha256(intake/'artifact_hashes.json'),
        'decisions_registry_sha256':sha256(intake/'decisions'/'artifact_hashes.json'),
        'review_package_registry_sha256':sha256(package/'artifact_hashes.json'),
        'llm_audit_registry_sha256':sha256(llm_audit/'artifact_hashes.json'),
        'formatting_repair_cells_n':evidence['formatting_repair_cells_n'],
        'human_decisions_modified':False,'code_sha256':{'summary':sha256(Path(__file__)),'intake':evidence['code_sha256']},
        'llm_model_revision_prompt_and_settings_unverified':True})
    (output/'methodology.md').write_text('''# Task 05 completed human semantic review

The user confirmed Baghdad time. All 213 native Excel timestamps are converted
to ISO 8601 with Asia/Baghdad offset +03:00, preserving local clock values.
40 text cells in 20 variants lost only ASCII edge spaces (19 trailing pairs and
one leading pair); the explicit repair restores hash-registered original text.
The source XLSX, raw cells, auxiliary diffs and every repair are retained privately.
No words, internal spaces, negation or human judgments are changed. Final decisions
are recomputed from both independent review files plus explicit third-reviewer
adjudication. A pandas row-index portability bug in expected queue comparison was
fixed by resetting its index; an end-to-end test covers a nonzero conflict index.

All 404 assignments are resolved: 394 eligible, including 184 human-negation
and 210 non-negation variants; 10 excluded (3 ambiguous meaning, 1 changed meaning
with uncertain sentiment, 6 preserved meaning with uncertain negation effects).
Confirmation is per reviewed variant only; no approval transfers to other tweets
or entire perturbation rules. Counts and tables concern the stratified review
sample, not the entire validation partition. Repeated original IDs require paired
original-ID bootstrap per rule/intensity when inference results are available.
Pre-adjudication agreement is retained rather than replaced with final consensus.
Human identities, independence and blinding are user-declared, not authenticated.

LLM agreement with the finalized human reference is reported separately on all
404 assignments, retaining uncertainty and changed/ambiguous categories. Nominal
kappa and raw agreement depend strongly on marginal label prevalence; high raw
agreement need not indicate reliable identification of changed/ambiguous cases.
The LLM's model revision, prompt and settings remain unverified. These are
post-hoc comparisons of provided annotations, not a new LLM annotation experiment
or MARBERT validation/robustness metrics. LLM judgments never adjudicate humans.

The supplied Task 03B ZIP contains original validation predictions, but Task 04
perturbed predictions and full inference evidence have not been supplied locally.
Consequently Accuracy/Macro-F1 drops, flip rates, paired bootstrap intervals and
tatweel token/probability comparisons remain unavailable. No model is trained,
reselected or evaluated on official test text. Model analysis resumes only when
the saved clean/perturbed probabilities, stable IDs and checkpoint/input provenance
are verified. The selected tokenizer is needed for full/effective token-ID
comparison; it is not needed merely to explain annotation counts.
''',encoding='utf-8')
    seal(output)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
