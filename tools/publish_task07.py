"""Publish verified v1 evidence and explicit v2 preparation/blocker status."""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch

from arsarcasm_audit.audit import sha256
from nasn.task05 import fresh,registry,seal,save
from nasn.task04 import read
from nasn.task07 import verified_v1_predictions
from nasn.task04 import paired_bootstrap
from nasn.metrics import calculate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['intake','diagnosis','prepared','attempt','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--extra-examples',required=True)
    args=p.parse_args();intake,diagnosis,prepared,attempt=map(Path,[args.intake,args.diagnosis,args.prepared,args.attempt])
    verified_v1_predictions(intake)
    for path in [diagnosis,prepared,attempt]:registry(path)
    status=read(attempt/'run_status.json')
    if status['status']!='blocked_or_failed' or status['metrics_computed']:
        raise ValueError('This preparation-only publisher must not mislabel a completed v2 run')
    output=fresh(args.output)
    # Original full light report payloads are text-free and preserved byte-for-byte.
    shutil.copytree(intake,output/'v1-reproduction')
    diag=output/'v1-diagnosis';diag.mkdir()
    for name in ['pipeline_equality.csv','all_B3_B4_differences.csv','all_elongation_token_exceptions.csv',
                 'guard_difference_summary.csv','annotation_quality_status.json','evidence.json']:
        shutil.copyfile(diagnosis/name,diag/name)
    seal(diag)
    v2=output/'v2-preparation';v2.mkdir()
    for name in ['normalization_manifest.csv.gz','character_token_audit.csv','normalization_rules.json','evidence.json','run_status.json']:
        shutil.copyfile(prepared/name,v2/name)
    seal(v2)
    rows=[]
    predictions=verified_v1_predictions(intake)
    clean_rows=[];per_class=[]
    for method,g in predictions[predictions.kind.eq('clean')].groupby('strategy'):
        values,classes,_=calculate(g.gold_id,g.predicted_id)
        clean_rows.append({'strategy':method,**values})
        per_class += [{'strategy':method,**row} for row in classes]
    pd.DataFrame(clean_rows).to_csv(output/'v1_all_clean_metrics.csv',index=False)
    pd.DataFrame(per_class).to_csv(output/'v1_clean_per_class.csv',index=False)
    paired=[]
    changed=predictions[predictions.kind.eq('perturbed')]
    b3=changed[changed.strategy.eq('B3')].set_index('input_id')
    b4=changed[changed.strategy.eq('B4')].set_index('input_id').loc[b3.index].copy()
    b4['clean_id']=b3.predicted_id  # Explicit comparison of B3/B4 on SAME perturbed text.
    for (cohort,rule,intensity),g in b4.groupby(['source_review_status','rule','intensity']):
        common={'source_cohort':cohort,'rule':rule,'intensity':int(intensity),'n':len(g),
            'left':'B3_perturbed','right':'B4_perturbed','seed':42,'bootstrap_repetitions':2000}
        if cohort=='reviewed_excluded' or len(g)<30:
            paired.append({**common,'status':'excluded_or_insufficient_sample_size'});continue
        for metric,value in paired_bootstrap(g,2000).items():paired.append({**common,'status':'computed','metric':metric,**value})
    pd.DataFrame(paired).to_csv(output/'v1_B3_B4_perturbed_paired.csv',index=False)
    source=predictions[predictions.strategy.eq('B0')&predictions.kind.eq('perturbed')]
    for cohort,group in source.groupby('source_review_status'):
        for (rule,intensity),g in group.groupby(['rule','intensity']):
            for method in ['RAW','TAT','REP2','NO_CONTEXT','NASN2']:
                rows.append({'method':method,'source_cohort':cohort,'rule':rule,'intensity':int(intensity),'n':len(g),
                    'status':'excluded_by_source_review' if cohort=='reviewed_excluded' else 'v2_model_comparison_not_run',
                    'clean_accuracy':None,'perturbed_accuracy':None,'clean_macro_f1':None,'perturbed_macro_f1':None,
                    'prediction_flip_rate':None,'paired_ci_low':None,'paired_ci_high':None,
                    'note':'Historical raw v1 scores are in v1-reproduction; no proxy reuse for new methods'})
    pd.DataFrame(rows).to_csv(output/'v2_comparison_status.csv',index=False)
    save(output/'blocker_report.json',{'reason':status['reason'],'actual_local_attempt':status,
        'cuda_available':torch.cuda.is_available(),'cuda_build':torch.version.cuda,
        'trained_weight_bytes_uploaded':False,'no_model_substitution':True,
        'selected_checkpoint_sha256':read(prepared/'evidence.json')['checkpoint_sha256'],
        'needed':'Execute the pinned Task 07 Colab notebook with the existing selected Drive checkpoint; return its report ZIP',
        'v1_predictions_available_and_verified':True,'v2_predictions_available':False})
    save(output/'scientific_decision.json',read(attempt/'scientific_decision.json'))
    save(output/'publication_evidence.json',{'source_task06_archive_sha256':read(intake/'intake_status.json')['source_archive_sha256'],
        'v1_external_registry_sha256':sha256(intake/'report_checksums.json'),
        'diagnostic_private_full_examples_sha256':sha256(diagnosis/'all_B3_B4_differences_private.csv'),
        'elongation_private_full_examples_sha256':sha256(Path(args.extra_examples)/'all_elongation_exceptions_private.csv'),
        'v2_new_human_review_queue_sha256':sha256(prepared/'new_human_review_queue.csv'),
        'v2_reviewer_a_zip_sha256':sha256(prepared/'new_human_reviewer_a.zip'),
        'v2_reviewer_b_zip_sha256':sha256(prepared/'new_human_reviewer_b.zip'),
        'v2_private_registry_sha256':sha256(prepared/'artifact_hashes.json'),
        'registered_subdirectories':{name:sha256(output/name/'artifact_hashes.json') for name in ['v1-reproduction','v1-diagnosis','v2-preparation']},
        'publisher_sha256':sha256(Path(__file__)),'full_raw_text_published':False,'weight_bytes_published':False,
        'human_provenance_verified':False,'new_independent_human_IAA_computed':False})
    seal(output)
    print('Published verified v1 and prepared v2 with explicit undetermined decision:',output)


if __name__=='__main__':main()
