"""Task 06B: fixed normalization preparation and validation-only checkpoint inference."""
import argparse
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import LABEL2ID, require, library_versions
from .normalization import STRATEGIES, DEFAULT_RULES, load_rules, normalize, restore, text_hash
from .task04 import (validation_inputs, verify_run, verify_external_reports, build_manifest,
                     review_sample, read, paired_bootstrap)
from .task05 import csv, fresh, registry, seal, save, verify_preparation

BASELINE = 'reports/task03b/marbert-v1-20261009T213143Z-be8b8811'


def restore_task04(output, config):
    """Explicitly rehydrate Task 04 text from pinned training, not regenerate splits."""
    public=Path('reports/task04')
    # This historical publication registry includes a nested failed-attempt log.
    # Preserve its original layout rather than applying the flat private registry.
    public_hashes=read(public/'artifact_hashes.json')
    require(public_hashes,'Empty original Task 04 registry')
    for name,digest in public_hashes.items():
        path=public/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                path.resolve().is_relative_to(public.resolve()) and sha256(path)==digest,
                f'Original Task 04 checksum differs: {name}')
    validation,_,evidence=validation_inputs(config)
    manifest=build_manifest(validation)
    expected=pd.read_csv(io.StringIO(manifest.to_csv(index=False)),dtype=str,keep_default_na=False)
    recorded=csv(public/'perturbation_manifest.csv')
    require(expected.drop(columns=['original_text','perturbed_text']).equals(recorded),
            'Restored Task 04 text/metadata differ from registered original manifest')
    queue=review_sample(manifest)
    public_queue=csv(public/'human_review_queue.csv')
    expected_queue=pd.read_csv(io.StringIO(queue.to_csv(index=False)),dtype=str,keep_default_na=False)
    require(expected_queue.drop(columns=['original_text','perturbed_text']).equals(public_queue),
            'Restored original review queue differs')
    output=fresh(output)
    manifest.to_csv(output/'perturbation_manifest.csv',index=False)
    queue.to_csv(output/'human_review_queue.csv',index=False)
    (output/'preparation.json').write_bytes((public/'preparation.json').read_bytes())
    save(output/'restoration.json',{'operation':'explicit_text_rehydration_from_pinned_official_training',
        'public_manifest_sha256':sha256(public/'perturbation_manifest.csv'),
        'split_manifests_regenerated':False,'human_decisions_generated':False,
        'test_text_read':False,'test_evaluated':False,'evidence':evidence})
    seal(output)


def token_signature(tokenizer, text):
    full=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
    effective=tokenizer(text,add_special_tokens=True,truncation=True,max_length=128)['input_ids']
    return {'full_sha256':text_hash(json.dumps(full)),'effective_sha256':text_hash(json.dumps(effective)),
            'full_count':len(full),'effective_count':len(effective),'truncated':len(full)>128}


def prepare(args, config):
    queue,manifest,baseline,evidence=verify_preparation(args.preparation,args.baseline,config)
    reviewed=Path('reports/task05-final-review'); registry(reviewed)
    review=csv(reviewed/'confirmed_evaluation_manifest.csv')
    require(len(review)==404 and int(review.eligible_confirmed.eq('True').sum())==394,'Final review changed')
    lookup=manifest.set_index('variant_id')
    for field in ['source_row_id','original_sha256','perturbed_sha256','rule','intensity']:
        require(review[field].tolist()==lookup.loc[review.variant_id,field].tolist(),f'Review input differs: {field}')
    review_lookup=review.set_index('variant_id')
    confirmed_source_ids=set(review.loc[review.eligible_confirmed.eq('True'),'source_row_id'])
    from .verified_tokenizer import verify
    from transformers import AutoTokenizer
    hashes=verify(args.tokenizer_directory,evidence['checkpoint_sha256'])
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_directory,local_files_only=True,trust_remote_code=False)
    rules=load_rules(); output=fresh(args.output)
    originals=manifest.drop_duplicates('source_row_id').copy()
    inputs=[]
    for row in originals.itertuples():
        inputs.append({'input_id':'clean:'+row.source_row_id,'source_row_id':row.source_row_id,'kind':'clean',
            'text':row.original_text,'input_sha256':row.original_sha256,'rule':'clean','intensity':0,
            'gold_id':int(row.gold_id),'high_risk':row.high_risk_near_duplicate=='True',
            'lexical_negation':row.has_negation=='True','source_human_confirmed':False,
            'source_review_status':'not_assigned','human_negation':'not_assigned'})
    for row in manifest[manifest.applicable.eq('True')].itertuples():
        human=review_lookup.loc[row.variant_id] if row.variant_id in review_lookup.index else None
        inputs.append({'input_id':row.variant_id,'source_row_id':row.source_row_id,'kind':'perturbed',
            'text':row.perturbed_text,'input_sha256':row.perturbed_sha256,'rule':row.rule,'intensity':int(row.intensity),
            'gold_id':int(row.gold_id),'high_risk':row.high_risk_near_duplicate=='True',
            'lexical_negation':row.has_negation=='True',
            'source_human_confirmed':human is not None and human.eligible_confirmed=='True',
            'source_review_status':'unreviewed_provisional' if human is None else
                'eligible_user_declared_human' if human.eligible_confirmed=='True' else 'reviewed_excluded',
            'human_negation':human.negation_present if human is not None else 'not_assigned'})
    require(len(inputs)==12470 and sum(r['source_human_confirmed'] for r in inputs)==394,'Unexpected development population')
    tokens={}; transformed={}; rows=[]; queue_records=[]
    def signature(text):
        if text not in tokens:tokens[text]=token_signature(tokenizer,text)
        return tokens[text]
    with gzip.open(output/'reversible_edit_logs.jsonl.gz','wt',encoding='utf-8') as logs:
        for index,row in enumerate(inputs):
            require(text_hash(row['text'])==row['input_sha256'],'Source text checksum differs')
            raw_tokens=signature(row['text'])
            for strategy in STRATEGIES:
                key=(row['text'],strategy)
                if key not in transformed:
                    result=normalize(row['text'],strategy,rules)
                    require(restore(result.text,result)==row['text'],'Normalization inverse failed')
                    transformed[key]=result
                result=transformed[key]; normalized_tokens=signature(result.text)
                record={**row,'strategy':strategy,'normalized_sha256':result.normalized_sha256,
                    'normalized_text':result.text,'character_changed':result.text!=row['text'],
                    'edits_n':len(result.edits),'protected_spans_n':len(result.protected_spans),
                    'full_tokens_changed':raw_tokens['full_sha256']!=normalized_tokens['full_sha256'],
                    'effective_tokens_changed':raw_tokens['effective_sha256']!=normalized_tokens['effective_sha256'],
                    'raw_full_token_count':raw_tokens['full_count'],'normalized_full_token_count':normalized_tokens['full_count'],
                    'raw_effective_token_sha256':raw_tokens['effective_sha256'],
                    'normalized_effective_token_sha256':normalized_tokens['effective_sha256'],
                    'normalized_full_token_sha256':normalized_tokens['full_sha256'],
                    'normalized_truncated':normalized_tokens['truncated'],'normalization_human_validated':False}
                rows.append(record)
                logs.write(json.dumps({'input_id':row['input_id'],**result.record()},ensure_ascii=False)+'\n')
                if (row['source_human_confirmed'] or (row['kind']=='clean' and row['source_row_id'] in confirmed_source_ids)) and result.edits:
                    queue_records.append({'input_id':row['input_id'],'source_row_id':row['source_row_id'],
                        'strategy':strategy,'kind':row['kind'],'rule':row['rule'],'intensity':row['intensity'],
                        'input_text':row['text'],'normalized_text':result.text,
                        'input_sha256':row['input_sha256'],'normalized_sha256':result.normalized_sha256,
                        'semantic_validity':'','sentiment_preserved':'','reviewer_id':'','review_timestamp':'','linguistic_notes':''})
            if (index+1)%2000==0:print(f'Prepared {index+1}/{len(inputs)} input texts',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'normalized_inputs.csv.gz',index=False,compression='gzip')
    public=frame.drop(columns=['text','normalized_text'])
    public.to_csv(output/'normalization_manifest.csv.gz',index=False,compression='gzip')
    public.groupby(['strategy','kind']).agg(rows=('input_id','size'),characters_changed=('character_changed','sum'),
        full_tokens_changed=('full_tokens_changed','sum'),effective_tokens_changed=('effective_tokens_changed','sum'),
        protected_spans=('protected_spans_n','sum')).reset_index().to_csv(output/'character_token_audit.csv',index=False)
    if queue_records:
        q=pd.DataFrame(queue_records).sort_values('input_id').groupby(['strategy','rule','intensity'],sort=True).head(2)
        q.to_csv(output/'new_normalization_review_queue.csv',index=False,encoding='utf-8-sig')
    else:pd.DataFrame(columns=['input_id','strategy','semantic_validity']).to_csv(output/'new_normalization_review_queue.csv',index=False)
    b3=frame[frame.strategy.eq('B3')].set_index('input_id')
    b4=frame[frame.strategy.eq('B4')].set_index('input_id').loc[b3.index]
    delta=pd.DataFrame({'input_id':b3.index,'source_row_id':b3.source_row_id,'kind':b3.kind,
        'characters_differ':b3.normalized_sha256!=b4.normalized_sha256,
        'effective_tokens_differ':b3.normalized_effective_token_sha256!=b4.normalized_effective_token_sha256}).reset_index(drop=True)
    delta.to_csv(output/'B3_B4_guard_effect.csv',index=False)
    collisions=[]
    for strategy in STRATEGIES:
        clean=frame[frame.strategy.eq(strategy)&frame.kind.eq('clean')]
        for digest,group in clean.groupby('normalized_sha256'):
            if len(group)>1:
                collisions.append({'strategy':strategy,'normalized_sha256':digest,'n':len(group),
                    'gold_conflict':group.gold_id.nunique()>1,'source_row_ids':'|'.join(sorted(group.source_row_id))})
    pd.DataFrame(collisions,columns=['strategy','normalized_sha256','n','gold_conflict','source_row_ids']).to_csv(output/'development_collisions.csv',index=False)
    save(output/'evidence.json',{**evidence,'rules_sha256':sha256(DEFAULT_RULES),'normalization_code_sha256':sha256(Path(__file__).with_name('normalization.py')),
        'task06_code_sha256':sha256(Path(__file__)),'tokenizer_sha256':hashes,
        'review_manifest_sha256':sha256(reviewed/'confirmed_evaluation_manifest.csv'),
        'task05_results_registry_sha256':sha256(Path('reports/task05-robustness/artifact_hashes.json')),
        'normalization_decision_inputs':['raw_text','strategy','fixed_rules'],
        'decisions_use_gold_labels_or_predictions':False,'protocol_tuning_performed':False,
        'development_is_existing_validation':'post-hoc study, not independent final evaluation',
        'character_token_audit_rows':len(frame),'inverse_checks_passed':len(frame)})
    (output/'normalization_rules.json').write_bytes(DEFAULT_RULES.read_bytes())
    save(output/'run_status.json',{'status':'prepared_character_token_audit_complete_model_inference_pending',
        'clean_original_ids':2509,'applicable_perturbed_variants':9961,'source_human_confirmed_variants':394,
        'source_unreviewed_provisional_variants':9557,'source_reviewed_excluded_variants':10,
        'normalization_strategy_input_rows':len(frame),
        'new_normalization_human_validated':False,'training_performed':False,'test_text_read':False,'test_evaluated':False,
        'new_strategy_metrics_computed':False})
    seal(output)


def comparison_tables(predictions, repetitions=2000):
    """Matched original-ID resampling within each rule/intensity, not pooled variants."""
    require(repetitions>=100,'At least 100 paired bootstrap resamples required')
    require(not predictions.duplicated(['input_id','strategy']).any(),'Repeated prediction identity')
    require(set(predictions.strategy)==set(STRATEGIES),'Incomplete strategy predictions')
    clean=predictions[predictions.kind.eq('clean')].set_index(['strategy','source_row_id'])
    changed=predictions[predictions.kind.eq('perturbed')].copy()
    changed['clean_id']=[int(clean.loc[(s,i),'predicted_id']) for s,i in zip(changed.strategy,changed.source_row_id)]
    reports=[]; strategy_deltas=[]
    require(changed.source_review_status.isin(['eligible_user_declared_human','unreviewed_provisional','reviewed_excluded']).all(),
            'Unknown source review provenance')
    require(changed.source_human_confirmed.equals(changed.source_review_status.eq('eligible_user_declared_human')),
            'Source eligibility flag contradicts review provenance')
    for semantic,eligible in [('source_human_confirmed',changed[changed.source_review_status.eq('eligible_user_declared_human')]),
                             ('source_unreviewed_provisional',changed[changed.source_review_status.eq('unreviewed_provisional')]),
                             ('source_reviewed_excluded',changed[changed.source_review_status.eq('reviewed_excluded')])]:
        for subset,part in [('all_development',eligible),('exclude_high_risk',eligible[~eligible.high_risk])]:
            for (rule,intensity),group in part.groupby(['rule','intensity']):
                strata=[('all','all',group)]
                field='human_negation' if semantic=='source_human_confirmed' else 'lexical_negation'
                strata += [(field,str(value),g) for value,g in group.groupby(field)]
                for field,value,group in strata:
                    for strategy,g in group.groupby('strategy'):
                        common={'strategy':strategy,'semantic_cohort':semantic,'normalization_human_validated':False,
                            'subset':subset,'rule':rule,'intensity':int(intensity),'subgroup':field,'value':value,
                            'n':len(g),'seed':42,'bootstrap_repetitions':repetitions}
                        if semantic=='source_reviewed_excluded':
                            reports.append({**common,'status':'excluded_by_source_human_review'});continue
                        if len(g)<30:reports.append({**common,'status':'insufficient_sample_size'});continue
                        for metric,estimate in paired_bootstrap(g,repetitions).items():
                            reports.append({**common,'status':'computed','metric':metric,**estimate})
                    base=group[group.strategy.eq('B0')].set_index('input_id').sort_index()
                    if semantic=='source_reviewed_excluded':continue
                    for strategy,g in group.groupby('strategy'):
                        if strategy=='B0':continue
                        g=g.set_index('input_id').loc[base.index]
                        common={'strategy':strategy,'reference':'B0','semantic_cohort':semantic,'subset':subset,
                            'rule':rule,'intensity':int(intensity),'subgroup':field,'value':value,'n':len(g),'seed':42,'bootstrap_repetitions':repetitions}
                        if len(g)<30:strategy_deltas.append({**common,'status':'insufficient_sample_size'});continue
                        # Same resampled IDs for both pipelines and both conditions.
                        arrays=np.vstack([base.gold_id,base.clean_id,base.predicted_id,g.clean_id,g.predicted_id]).astype(int)
                        from .task04 import paired_scores
                        def values(a):
                            b=paired_scores(a[0],a[1],a[2]); n=paired_scores(a[0],a[3],a[4])
                            return {'perturbed_macro_f1_gain':n['perturbed_macro_f1']-b['perturbed_macro_f1'],
                                'clean_macro_f1_gain':n['clean_macro_f1']-b['clean_macro_f1'],
                                'macro_f1_drop_reduction':b['macro_f1_drop']-n['macro_f1_drop'],
                                'flip_rate_reduction':b['flip_rate']-n['flip_rate']}
                        point=values(arrays); rng=np.random.default_rng(42); draws={key:[] for key in point}
                        for _ in range(repetitions):
                            result=values(arrays[:,rng.integers(0,len(g),len(g))])
                            for key,v in result.items():draws[key].append(v)
                        for key,v in point.items():strategy_deltas.append({**common,'status':'computed','metric':key,
                            'estimate':v,'ci_low':float(np.quantile(draws[key],.025)),'ci_high':float(np.quantile(draws[key],.975))})
    return pd.DataFrame(reports),pd.DataFrame(strategy_deltas)


def run(args, config):
    output=fresh(args.output); start=time.perf_counter()
    status={'status':'initializing','training_performed':False,'test_text_read':False,'test_evaluated':False,'metrics_computed':False}
    try:
        preparation=Path(args.prepared);registry(preparation);evidence=read(preparation/'evidence.json')
        require(evidence['rules_sha256']==sha256(DEFAULT_RULES) and evidence['normalization_code_sha256']==sha256(Path(__file__).with_name('normalization.py')),
                'Prepared normalization protocol changed')
        require(evidence['task06_code_sha256']==sha256(Path(__file__)),'Prepared runner code changed')
        require(evidence['review_manifest_sha256']==sha256('reports/task05-final-review/confirmed_evaluation_manifest.csv'), 'Human cohort changed')
        validation,_,current=validation_inputs(config);verify_external_reports(args.baseline)
        # Missing selected weights fail here, never download another model.
        saved,hashes,_=verify_run(args.baseline,args.checkpoint,validation,current,config)
        require(hashes==evidence['checkpoint_sha256'] and current['manifest_sha256']==evidence['manifest_sha256'],'Prepared model/data identity differs')
        import torch
        require(torch.cuda.is_available() and torch.version.cuda is not None,'CUDA GPU unavailable; use the prepared Colab notebook')
        free,total=torch.cuda.mem_get_info();require(free>=2*1024**3,'Less than 2 GiB free GPU memory before checkpoint loading')
        require(args.batch_size>=1,'Inference batch size must be positive')
        from .marbert import seed_all,encoded_dataset,evaluate
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        seed_all(42,4)
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModelForSequenceClassification.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False,output_loading_info=True)
        require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), 'Incomplete selected checkpoint load; random fallback prohibited')
        require(model.config.num_labels==3 and model.config.label2id==LABEL2ID,'Checkpoint label mapping changed')
        model.to('cuda').eval();torch.cuda.reset_peak_memory_stats()
        frame=pd.read_csv(preparation/'normalized_inputs.csv.gz',keep_default_na=False)
        require(len(frame)==62350 and not frame.duplicated(['input_id','strategy']).any(),'Prepared row coverage differs')
        require(frame.strategy.value_counts().to_dict()=={s:12470 for s in STRATEGIES},'Prepared strategies incomplete')
        require(all(text_hash(t)==h for t,h in zip(frame.normalized_text,frame.normalized_sha256)),'Prepared normalized text hash differs')
        unique=frame.drop_duplicates('normalized_sha256').copy().rename(columns={'normalized_text':'tweet'})
        unique['sentiment']=unique.gold_id.map({v:k for k,v in LABEL2ID.items()});unique['partition']='validation'
        loader=torch.utils.data.DataLoader(encoded_dataset(unique,tokenizer,128),batch_size=args.batch_size,shuffle=False)
        probabilities,_=evaluate(model,loader,'cuda')
        require(probabilities.shape==(len(unique),3) and np.isfinite(probabilities).all(),'Invalid inference probabilities')
        prob_lookup=dict(zip(unique.normalized_sha256,probabilities))
        pred=frame.drop(columns=['text','normalized_text']).copy()
        probs=np.stack([prob_lookup[h] for h in frame.normalized_sha256])
        pred['predicted_id']=probs.argmax(axis=1);pred['confidence']=probs.max(axis=1)
        for label,index in LABEL2ID.items():pred['probability_'+label]=probs[:,index]
        raw=pred[pred.strategy.eq('B0')&pred.kind.eq('clean')].set_index('source_row_id').loc[saved.source_row_id]
        require(raw.predicted_id.tolist()==saved.predicted_id.tolist() and np.allclose(raw[[f'probability_{x}' for x in LABEL2ID]],saved[[f'probability_{x}' for x in LABEL2ID]],atol=1e-5,rtol=1e-5),
                'Reloaded checkpoint B0 clean predictions differ from Task 03B')
        pred.to_csv(output/'development_predictions.csv.gz',index=False,compression='gzip')
        metrics,deltas=comparison_tables(pred,args.bootstrap)
        metrics.to_csv(output/'comparison_metrics.csv',index=False);deltas.to_csv(output/'paired_strategy_differences.csv',index=False)
        status.update(status='completed',metrics_computed=True,gpu=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,
            total_vram_bytes=total,free_vram_before_model_bytes=free,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            batch_size=args.batch_size,checkpoint_reload_verified=True,unique_inference_inputs=len(unique),seed=42,
            versions=library_versions(),duration_seconds=time.perf_counter()-start,
            prepared_registry_sha256=sha256(preparation/'artifact_hashes.json'),checkpoint_sha256=hashes,
            bootstrap_repetitions=args.bootstrap,normalization_human_validated=False,
            inference_precision='float32',automatic_strategy_selection=False)
        save(output/'loading_info.json',loading)
    except Exception as exc:
        status.update(status='blocked_or_failed',reason=f'{type(exc).__name__}: {exc}',duration_seconds=time.perf_counter()-start)
        save(output/'run_status.json',status);seal(output);raise
    save(output/'run_status.json',status);seal(output)
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/marbert_v1.json')
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('restore-task04');r.add_argument('--output',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--preparation',required=True);prep.add_argument('--baseline',default=BASELINE)
    prep.add_argument('--tokenizer-directory',required=True);prep.add_argument('--output',required=True)
    run_parser=sub.add_parser('run');run_parser.add_argument('--prepared',required=True);run_parser.add_argument('--checkpoint',required=True)
    run_parser.add_argument('--baseline',default=BASELINE);run_parser.add_argument('--output',required=True)
    run_parser.add_argument('--batch-size',type=int,default=8);run_parser.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args(argv)
    try:
        config=read(args.config)
        if args.command=='restore-task04':restore_task04(args.output,config)
        elif args.command=='prepare':prepare(args,config)
        else:print(json.dumps(run(args,config),indent=2))
    except Exception as exc:p.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
