"""Publish Task 06 preparation evidence and verified historical B0 comparisons."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from arsarcasm_audit.audit import sha256
from nasn.preflight import require,library_versions
from nasn.normalization import STRATEGIES
from arsarcasm_audit.leakage import group_id
from nasn.task04 import read,paired_bootstrap
from nasn.task05 import csv,fresh,registry,seal,save,verify_preparation,validate_saved_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','preparation','inference','attempt','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--baseline',default='reports/task03b/marbert-v1-20261009T213143Z-be8b8811')
    args=p.parse_args()
    prepared=Path(args.prepared); registry(prepared)
    queue,manifest,baseline,evidence=verify_preparation(args.preparation,args.baseline,read('configs/marbert_v1.json'))
    clean,changed,_=validate_saved_predictions(args.inference,manifest,baseline,evidence)
    review=Path('reports/task05-final-review');registry(review)
    decisions=csv(review/'confirmed_evaluation_manifest.csv')
    ids=set(decisions.loc[decisions.eligible_confirmed.eq('True'),'variant_id'])
    eligible=changed[changed.applicable].copy()
    eligible['source_human_confirmed']=eligible.variant_id.isin(ids)
    reviewed_ids=set(decisions.variant_id)
    eligible['source_review_status']=['eligible_user_declared_human' if i in ids else
        'reviewed_excluded' if i in reviewed_ids else 'unreviewed_provisional' for i in eligible.variant_id]
    require(len(eligible)==9961 and eligible.source_human_confirmed.sum()==394,'B0 cohort coverage differs')
    output=fresh(args.output)
    for name in ['normalization_manifest.csv.gz','character_token_audit.csv','B3_B4_guard_effect.csv',
                 'development_collisions.csv','normalization_rules.json','evidence.json','run_status.json']:
        (output/name).write_bytes((prepared/name).read_bytes())
    normalized=pd.read_csv(prepared/'normalized_inputs.csv.gz',keep_default_na=False)
    train=pd.read_csv('reports/task02/train_manifest.csv',dtype=str)
    require(sha256('reports/task02/train_manifest.csv')==evidence['manifest_sha256']['train_manifest.csv'],
            'Active training manifest changed before post-normalization audit')
    exact,canonical=set(train.text_sha256),set(train.group_id)
    mapping={r.normalized_sha256:(r.normalized_sha256 in exact,group_id(r.normalized_text) in canonical)
             for r in normalized.drop_duplicates('normalized_sha256').itertuples()}
    normalized['exact_training_overlap']=normalized.normalized_sha256.map(lambda h:mapping[h][0])
    normalized['canonical_training_overlap']=normalized.normalized_sha256.map(lambda h:mapping[h][1])
    normalized.groupby(['strategy','kind'])[['exact_training_overlap','canonical_training_overlap']].sum().reset_index().to_csv(output/'post_normalization_training_overlap.csv',index=False)
    # Historical B0 is actual saved inference; B1-B4 have no new classifier outputs.
    rows=[]; detailed=[]
    for semantic,part in [('source_human_confirmed',eligible[eligible.source_review_status.eq('eligible_user_declared_human')]),
                          ('source_unreviewed_provisional',eligible[eligible.source_review_status.eq('unreviewed_provisional')]),
                          ('source_reviewed_excluded',eligible[eligible.source_review_status.eq('reviewed_excluded')])]:
        for (rule,intensity),group in part.groupby(['rule','intensity']):
            common={'semantic_cohort':semantic,'subset':'all_development','rule':rule,'intensity':int(intensity),
                'n':len(group),'new_normalization_human_validated':False,'seed':42,'bootstrap_repetitions':2000}
            excluded=semantic=='source_reviewed_excluded'
            result=paired_bootstrap(group,2000) if len(group)>=30 and not excluded else None
            row={**common,'strategy':'B0','status':'excluded_by_source_human_review' if excluded else 'historical_saved_predictions' if result else 'insufficient_sample_size'}
            for metric in ['clean_accuracy','perturbed_accuracy','clean_macro_f1','perturbed_macro_f1','macro_f1_drop','flip_rate']:
                row[metric]=None if result is None else result[metric]['estimate']
                row[metric+'_ci_low']=None if result is None else result[metric]['ci_low']
                row[metric+'_ci_high']=None if result is None else result[metric]['ci_high']
            rows.append(row)
            if result:
                for metric,value in result.items():detailed.append({**common,'strategy':'B0','status':'historical_saved_predictions','metric':metric,**value})
            for strategy in STRATEGIES[1:]:
                rows.append({**common,'strategy':strategy,'status':'excluded_by_source_human_review' if excluded else 'not_run_selected_weights_unavailable',
                    **{m:None for m in ['clean_accuracy','perturbed_accuracy','clean_macro_f1','perturbed_macro_f1','macro_f1_drop','flip_rate']}})
    table=pd.DataFrame(rows)
    table.to_csv(output/'comparison_table.csv',index=False)
    table.to_latex(output/'comparison_table.tex',index=False,float_format='%.4f')
    pd.DataFrame(detailed).to_csv(output/'B0_historical_paired_metrics.csv',index=False)
    attempt=Path(args.attempt);registry(attempt)
    attempt_status=read(attempt/'run_status.json')
    require(attempt_status['status']=='blocked_or_failed' and attempt_status['metrics_computed'] is False,'Local attempt was not a documented block')
    require('Selected trained checkpoint directory unavailable' in attempt_status['reason'],
            'This preparation report requires the actual missing-selected-checkpoint blocker, not an unrelated failure')
    save(output/'blocker_report.json',{'status':'new_strategy_inference_blocked',
        'reason':attempt_status['reason'],'selected_checkpoint_directory_available':False,
        'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda,
        'local_versions':library_versions(),'selected_checkpoint_sha256':evidence['checkpoint_sha256'],
        'durable_checkpoint_path_from_completed_task03b':evidence['baseline_run_status']['best_checkpoint'],
        'no_pretrained_or_random_fallback':True,'new_B1_B4_model_metrics_available':False,
        'existing_B0_metrics_are_historical_saved_predictions':True,
        'attempt_status':attempt_status,'attempt_registry_sha256':sha256(attempt/'artifact_hashes.json')})
    save(output/'publication.json',{'source_preparation_registry_sha256':sha256(prepared/'artifact_hashes.json'),
        'actual_task04_inference_registry_sha256':sha256(Path(args.inference)/'artifact_hashes.json'),
        'confirmed_manifest_sha256':sha256(review/'confirmed_evaluation_manifest.csv'),
        'publisher_sha256':sha256(Path(__file__)),'raw_text_published':False,'weights_published':False,
        'private_new_review_queue_sha256':sha256(prepared/'new_normalization_review_queue.csv'),
        'new_human_decisions_generated':False})
    (output/'methodology.md').write_text('''# Task 06B NASN prototype feasibility — preparation complete, inference blocked

Five fixed text-only methods B0–B4 are implemented. B0 retains raw input; B1
removes tatweel; B2 additionally applies NFC/whitespace normalization; B3 selectively
removes tatweel/NFC and allowlisted repetition, with shared critical-word protection;
B4 adds a bounded lexical negation-context guard. Gold labels, predictions, review
decisions and test information never choose an edit. Full exact rules, limits and
reproduction commands are in [the Task 06 methodology](../../docs/task06.md).

The exact selected tokenizer bytes are verified. Actual character/token auditing
covers 2,509 clean validation originals and 9,961 applicable Task 04 variants for
all five strategies: 62,350 records. Reversible private source/target logs recover
every input exactly. Public IDs/hashes/token counts are retained without full
text or token IDs. Post-normalization clean duplicate groups/label conflicts are
reported as an audit, never merged or used to select transformations.

394 reviewed source variants remain conditionally eligible; 9,557 source variants
are unreviewed provisional, and 10 reviewed exclusions are kept separate with no
robustness metrics. Reviewer independence is unverified in Task 06A. New normalized
clean/perturbed strings have no human semantic validation, and a private blank
review queue is prepared. Original review approval never propagates to new text.

Comparison tables show actual historical B0 saved predictions on the two eligible
reporting cohorts. Paired seed-42 intervals use 2,000 original-ID resamples within
rule/intensity, fixed NEG/NEU/POS labels and minimum n=30. B1–B4 scores remain
unavailable. The actual local CLI failed because the selected trained checkpoint
directory is unavailable; CUDA is also unavailable. No substitute encoder or
random classifier is used. The recorded epoch-2 model hash and durable external
path are in blocker_report.json. No NASN improvement is claimed.

The generated Colab notebook pins the immutable implementation, verifies GPU/VRAM,
CUDA kernels, every selected model/tokenizer/config file and B0 clean baseline
predictions before metric reporting. It performs no training or checkpoint
selection. Outputs/checksums remain durable on Drive, with a lightweight report
ZIP. The notebook is prepared and syntax-tested locally, not executed in Colab
here. A completed verified run and semantic review are remaining external steps.

Limits: post-hoc development on previously analyzed validation data, unverified
human provenance, prevalence-sensitive Kappa, nonrandom/small review strata,
subjective negation scope, uncertain-case exclusions, duplicate/near-duplicate
risks, finite lexicons and scope-window false positives/negatives. Tokenization
changes are descriptive correlations, not proof of semantic or causal effects.
There is no automatic winning strategy, parameter tuning, official test access,
new trained model or fabricated performance result. Prior evidence is unchanged.
''',encoding='utf-8')
    seal(output)
    print('Published preparation, historical B0 and explicit B1-B4 blockers:',output)


if __name__=='__main__':main()
