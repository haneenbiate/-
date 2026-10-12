"""Task 07 diagnostics and frozen v2 development evaluation; no model training."""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import require,LABEL2ID,library_versions
from .normalization import normalize as v1_normalize,restore,text_hash,negation_guard,load_rules as v1_rules
from .normalization_v2 import normalize as v2_normalize,explain,load_rules,STRATEGIES,SIMPLE,RULES
from .task06 import token_signature,comparison_tables as v1_tables
from .task04 import validation_inputs,verify_external_reports,verify_run,read,paired_scores
from .task05 import registry,fresh,seal,save

BASELINE='reports/task03b/marbert-v1-20261009T213143Z-be8b8811'
METHOD_MAP=dict(zip(['B0','B1','B2','B3','B4'],STRATEGIES))


def v1_inputs(directory):
    directory=Path(directory);registry(directory)
    evidence=read(directory/'evidence.json');known=read('reports/task06b/evidence.json')
    for field in ['manifest_sha256','source_commit','source_training_sha256','rules_sha256',
                  'normalization_code_sha256','task06_code_sha256','review_manifest_sha256','checkpoint_sha256']:
        require(evidence[field]==known[field],f'V1 preparation differs: {field}')
    frame=pd.read_csv(directory/'normalized_inputs.csv.gz',keep_default_na=False)
    public=pd.read_csv('reports/task06b/normalization_manifest.csv.gz',keep_default_na=False)
    require(frame.drop(columns=['text','normalized_text']).equals(public),'V1 registered input metadata differs')
    require(all(text_hash(t)==h for t,h in zip(frame.text,frame.input_sha256)) and
            all(text_hash(t)==h for t,h in zip(frame.normalized_text,frame.normalized_sha256)), 'Private v1 text hashes differ')
    return frame,evidence


def selected_tokenizer(directory,evidence):
    from .verified_tokenizer import verify
    from transformers import AutoTokenizer
    verify(directory,evidence['checkpoint_sha256'])
    return AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)


def verified_v1_predictions(intake):
    intake=Path(intake);registry(intake);registry(intake/'evaluation')
    require(read(intake/'intake_status.json')['status']=='verified_completed_task06_inference', 'Unverified Task 06 intake')
    from .task07_io import FILES
    hashes=read(intake/'report_checksums.json')
    require(set(hashes)==FILES,'Verified archive coverage changed')
    for name,digest in hashes.items():require(sha256(intake/name)==digest,f'Imported Task 06 bytes changed: {name}')
    return pd.read_csv(intake/'evaluation/development_predictions.csv.gz',keep_default_na=False)


def diagnose(args):
    frame,evidence=v1_inputs(args.v1_prepared);tokenizer=selected_tokenizer(args.tokenizer_directory,evidence)
    pred=verified_v1_predictions(args.task06_intake)
    public=frame.drop(columns=['text','normalized_text'])
    require(pred[public.columns].equals(public),'Task 06 predictions differ from private verified inputs')
    output=fresh(args.output);tokens={};results={};rules=v1_rules()
    for row in frame.itertuples():
        key=(row.text,row.strategy)
        if key not in results:results[key]=v1_normalize(row.text,row.strategy,rules)
        result=results[key]
        require(result.text==row.normalized_text and restore(result.text,result)==row.text,'V1 character replay failed')
        if row.normalized_text not in tokens:tokens[row.normalized_text]=token_signature(tokenizer,row.normalized_text)
        actual=tokens[row.normalized_text]
        require(actual['effective_sha256']==row.normalized_effective_token_sha256 and
                actual['full_sha256']==row.normalized_full_token_sha256,'V1 selected-tokenizer replay failed')
    base=frame[frame.strategy.eq('B0')].set_index('input_id')
    clean=base[base.kind.eq('clean')].set_index('source_row_id')
    a=frame[frame.strategy.eq('B3')].set_index('input_id')
    b=frame[frame.strategy.eq('B4')].set_index('input_id').loc[a.index]
    probabilities=pred.set_index(['input_id','strategy'])
    differences=[]
    for input_id in a.index[a.normalized_sha256.ne(b.normalized_sha256)]:
        left,right=a.loc[input_id],b.loc[input_id]
        p,q=probabilities.loc[(input_id,'B3')],probabilities.loc[(input_id,'B4')]
        extra=[e for e in results[(left.text,'B3')].edits if (e.start,e.end,e.target_span) not in
               {(e.start,e.end,e.target_span) for e in results[(right.text,'B4')].edits}]
        markers=negation_guard(left.text,{**rules,'scope_following_arabic_tokens':0})
        typography=bool(extra) and all(e.source_span.replace('ـ','')==e.target_span and
            not any(e.start<end and start<e.end for start,end in markers) for e in extra)
        differences.append({'input_id':input_id,'source_row_id':left.source_row_id,'kind':left.kind,'rule':left.rule,
            'intensity':int(left.intensity),'source_review_status':left.source_review_status,
            'original_text':clean.loc[left.source_row_id,'text'],'input_text':left.text,
            'B3_normalized_text':left.normalized_text,'B4_normalized_text':right.normalized_text,
            'B3_edits':json.dumps([e.__dict__ for e in results[(left.text,'B3')].edits],ensure_ascii=False),
            'B4_edits':json.dumps([e.__dict__ for e in results[(right.text,'B4')].edits],ensure_ascii=False),
            'B4_guard_spans':json.dumps(negation_guard(left.text,rules)),
            'typography_only_extra_edits_outside_possible_marker':typography,'human_validated_safe':False,
            'effective_tokens_differ':left.normalized_effective_token_sha256!=right.normalized_effective_token_sha256,
            'B3_prediction':int(p.predicted_id),'B4_prediction':int(q.predicted_id),'gold_id':int(left.gold_id),
            'B3_confidence':float(p.confidence),'B4_confidence':float(q.confidence),
            **{method+'_probability_'+label:float(pr['probability_'+label]) for method,pr in [('B3',p),('B4',q)] for label in LABEL2ID},
            'B3_correct':int(p.predicted_id)==int(left.gold_id),'B4_correct':int(q.predicted_id)==int(left.gold_id),
            'B3_sha256':left.normalized_sha256,'B4_sha256':right.normalized_sha256})
    full=pd.DataFrame(differences)
    full.to_csv(output/'all_B3_B4_differences_private.csv',index=False,encoding='utf-8-sig')
    public_cols=[c for c in full if c not in ['original_text','input_text','B3_normalized_text','B4_normalized_text','B3_edits','B4_edits','B4_guard_spans']]
    full[public_cols].to_csv(output/'all_B3_B4_differences.csv',index=False)
    full.groupby(['kind','rule','intensity']).agg(n=('input_id','size'),
        typography_candidates=('typography_only_extra_edits_outside_possible_marker','sum'),
        B3_correct=('B3_correct','sum'),B4_correct=('B4_correct','sum')).reset_index().to_csv(output/'guard_difference_summary.csv',index=False)
    rows=[];exceptions=[]
    for (kind,rule,intensity),group in pred.groupby(['kind','rule','intensity']):
        ids=group[group.strategy.eq('B0')].input_id
        chars=group.pivot(index='input_id',columns='strategy',values='normalized_sha256').loc[ids]
        effective=group.pivot(index='input_id',columns='strategy',values='normalized_effective_token_sha256').loc[ids]
        predictions=group.pivot(index='input_id',columns='strategy',values='predicted_id').loc[ids]
        methods=['B1','B2','B3','B4']
        row={'kind':kind,'rule':rule,'intensity':int(intensity),'input_ids':len(ids),
            'B1_B4_identical_texts':int(chars[methods].nunique(axis=1).eq(1).sum()),
            'B1_B4_identical_effective_inputs':int(effective[methods].nunique(axis=1).eq(1).sum()),
            'B1_B4_identical_predictions':int(predictions[methods].nunique(axis=1).eq(1).sum())}
        rows.append(row)
        if rule=='elongation':
            for i in effective.index[effective[methods].nunique(axis=1).gt(1)]:
                exceptions.append({'input_id':i,'intensity':int(intensity),
                    **{m+'_token_sha256':effective.at[i,m] for m in methods},
                    **{m+'_prediction':int(predictions.at[i,m]) for m in methods}})
    pd.DataFrame(rows).to_csv(output/'pipeline_equality.csv',index=False)
    pd.DataFrame(exceptions).to_csv(output/'all_elongation_token_exceptions.csv',index=False)
    # Original descriptive annotations are not authenticated independent humans.
    registry('reports/task06a')
    provenance=read('reports/task06a/source_provenance.json')
    require(provenance['independence_status']=='unverified','Reassess independently supplied provenance before new IAA')
    save(output/'annotation_quality_status.json',{'status':'no_new_independent_human_agreement_computed',
        'reason':'Original A/B records exist but authenticated independent people/submissions/blinding are unverified',
        'original_source_registry_sha256':sha256('reports/task06a/artifact_hashes.json'),
        'prior_descriptive_statistics':'reports/task06a/agreement.csv; not verified human reliability',
        'human_independence_verified':False,'human_confirmation_claim':'user-declared eligibility only'})
    save(output/'evidence.json',{'v1_private_registry_sha256':sha256(Path(args.v1_prepared)/'artifact_hashes.json'),
        'verified_task06_intake_status':read(Path(args.task06_intake)/'intake_status.json'),
        'v1_replayed_strategy_input_rows':len(frame),'B3_B4_differences':len(full),
        'selected_checkpoint_weight_bytes_available_locally':False,'new_model_inference':False,
        'test_text_read':False,'training_performed':False,'diagnostic_code_sha256':sha256(Path(__file__))})
    seal(output)
    print('Replayed v1 characters/tokens and actual Task 06 predictions:',output)


def prepare(args):
    frame,evidence=v1_inputs(args.v1_prepared);tokenizer=selected_tokenizer(args.tokenizer_directory,evidence)
    inputs=frame[frame.strategy.eq('B0')].drop(columns=['strategy','normalized_text','normalized_sha256']).copy()
    output=fresh(args.output);rules=load_rules();records=[];review=[];tokens={};cache={}
    def signature(text):
        if text not in tokens:tokens[text]=token_signature(tokenizer,text)
        return tokens[text]
    with gzip.open(output/'reversible_v2_logs.jsonl.gz','wt',encoding='utf-8') as logs:
        for i,row in enumerate(inputs.to_dict('records')):
            for strategy in STRATEGIES:
                key=(row['text'],strategy)
                if key not in cache:
                    result,decisions=explain(row['text'],strategy,rules)
                    require(restore(result.text,result)==row['text'],'V2 inverse failed');cache[key]=(result,decisions)
                result,decisions=cache[key];sig=signature(result.text)
                item={**row,'strategy':strategy,'normalized_text':result.text,'normalized_sha256':result.normalized_sha256,
                    'character_changed':result.text!=row['text'],'edits_n':len(result.edits),
                    'protected_spans_n':len(result.protected_spans),
                    'normalized_full_token_sha256':sig['full_sha256'],'normalized_effective_token_sha256':sig['effective_sha256'],
                    'normalized_full_token_count':sig['full_count'],'normalized_truncated':sig['truncated'],
                    'full_tokens_changed':sig['full_sha256']!=signature(row['text'])['full_sha256'],
                    'effective_tokens_changed':sig['effective_sha256']!=signature(row['text'])['effective_sha256'],
                    'repeat_edits':sum('repeat_run_ge3_cap2' in e.rule for e in result.edits),
                    'repeat_scope_abstentions':sum(d.get('repetition_decision')=='abstain_negation_context' for d in decisions),
                    'normalization_human_validated':False}
                records.append(item)
                logs.write(json.dumps({'input_id':row['input_id'],**result.record(),'token_decisions':decisions},ensure_ascii=False)+'\n')
                if result.edits:
                    variant='nv2_'+text_hash(f'{rules["version"]}|{strategy}|{row["input_id"]}|{result.normalized_sha256}')
                    review.append({'variant_id':variant,'source_row_id':row['source_row_id'],
                        'original_sha256':result.original_sha256,'perturbed_sha256':result.normalized_sha256,
                        'original_text':row['text'],'perturbed_text':result.text,'rule':strategy,'intensity':len(result.edits),
                        'source_input_id':row['input_id'],'kind':row['kind'],'repeat_edits':item['repeat_edits']})
            if (i+1)%2000==0:print(f'V2 prepared {i+1}/{len(inputs)} source inputs',flush=True)
    result=pd.DataFrame(records)
    result.to_csv(output/'normalized_inputs.csv.gz',index=False,compression='gzip')
    result.drop(columns=['text','normalized_text']).to_csv(output/'normalization_manifest.csv.gz',index=False,compression='gzip')
    result.groupby(['strategy','kind']).agg(rows=('input_id','size'),changed=('character_changed','sum'),
        effective_tokens_changed=('effective_tokens_changed','sum'),repeat_edits=('repeat_edits','sum'),
        context_abstentions=('repeat_scope_abstentions','sum')).reset_index().to_csv(output/'character_token_audit.csv',index=False)
    # Explicitly sampled provisional review; no labels/predictions/rationale auto-filled.
    q=pd.DataFrame(review).sort_values('variant_id').groupby(['rule','kind','repeat_edits']).head(12).copy()
    from .task05 import HUMAN
    for field in HUMAN:q[field]=''
    q.to_csv(output/'new_human_review_queue.csv',index=False,encoding='utf-8-sig')
    # Reuse the existing strict two-reviewer/adjudication workflow on NEW outputs.
    from .task05 import VISIBLE,html_review
    package=output/'review-package';package.mkdir()
    template=q[VISIBLE+HUMAN]
    for reviewer in ['a','b']:
        template.to_csv(package/f'reviewer_{reviewer}.csv',index=False,encoding='utf-8-sig')
        html_review(template,package/f'reviewer_{reviewer}.html')
    source_meta=pd.read_csv('reports/task04/perturbation_manifest.csv',dtype=str,keep_default_na=False).drop_duplicates('source_row_id').set_index('source_row_id')
    meta=q.drop(columns=['original_text','perturbed_text']+HUMAN).copy()
    for field in ['gold_label','dialect','sarcasm','has_negation']:
        meta[field]=meta.source_row_id.map(source_meta[field])
    meta.to_csv(package/'queue_registry.csv',index=False)
    (package/'protocol.md').write_text('''# Independent NEW normalization review

Two different reviewers complete reviewer_a/reviewer_b independently without
seeing each other's answers, model predictions or gold sentiment. Record actual
reviewer identifiers, timezone-aware timestamps, rationale, cue and scope. No
identity or answer is generated by software. Review the input versus normalized
text, including emphasis, spelling/morphology, negation and intended sentiment.
Use changed/ambiguous/uncertain rather than assuming normalization is harmless.
Source perturbation eligibility does not approve this new transformation.

Keep separate original submissions and collect documented independence/blinding
and any AI-assistance disclosures. Distinct spreadsheet IDs alone do not establish
independence. Use an actual third reviewer for categorical/cue/scope conflicts.
Any claimed independently validated meaning-preserving analysis additionally
requires trustworthy provenance. Never let an LLM automatically adjudicate.

Import the completed original CSVs using nasn.task05 import-reviews, with this
directory as --review-package; provide explicit adjudication if requested. Keep
all human decisions separate from saved predictions. Only exact new variant IDs
can become eligible. Both normalized clean and perturbed members of a model
comparison require validation; do not propagate sampled approval to a rule.
''',encoding='utf-8')
    seal(package)
    import zipfile
    for reviewer in ['a','b']:
        with zipfile.ZipFile(output/f'new_human_reviewer_{reviewer}.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in [f'reviewer_{reviewer}.csv',f'reviewer_{reviewer}.html','protocol.md']:
                bundle.write(package/name,arcname=name)
    save(output/'evidence.json',{**evidence,'v1_source_registry_sha256':sha256(Path(args.v1_prepared)/'artifact_hashes.json'),
        'v2_rules_sha256':sha256(RULES),'v2_normalization_code_sha256':sha256(Path(__file__).with_name('normalization_v2.py')),
        'runner_code_sha256':sha256(Path(__file__)),'frozen_protocol_sha256':sha256('docs/task07_protocol.md'),
        'rows':len(result),'inverse_checks_passed':len(result),'normalization_decision_inputs':['text','strategy','fixed_rules'],
        'model_selection_or_training':False,'test_text_read':False})
    (output/'normalization_rules.json').write_bytes(RULES.read_bytes())
    save(output/'run_status.json',{'status':'v2_prepared_no_model_metrics','strategy_input_rows':len(result),
        'source_eligible_variants':394,'source_unreviewed_variants':9557,'source_excluded_variants':10,
        'new_human_validated':False,'new_review_queue_n':len(q),'training_performed':False,'test_evaluated':False})
    seal(output)


def cluster_metrics(frame,repetitions=2000):
    """All variants of a sampled source ID move together; paired across methods."""
    require(repetitions>=100,'At least 100 paired cluster draws required')
    require(set(frame.strategy)==set(STRATEGIES),'Missing v2 prediction strategies')
    clean=frame[frame.kind.eq('clean')].set_index(['strategy','source_row_id'])
    changed=frame[frame.kind.eq('perturbed')].copy()
    changed['clean_id']=[int(clean.loc[(s,i),'predicted_id']) for s,i in zip(changed.strategy,changed.source_row_id)]
    rows=[];differences=[]
    for cohort,label in [('source_human_confirmed','eligible_user_declared_human'),('source_unreviewed_provisional','unreviewed_provisional')]:
        part=changed[changed.source_review_status.eq(label)]
        if part.empty:continue
        methods={s:g.set_index('input_id').sort_index() for s,g in part.groupby('strategy')}
        base=methods['RAW'];ids=base.index
        require(all(set(g.index)==set(ids) for g in methods.values()),'Cohort pairing differs')
        methods={s:g.loc[ids] for s,g in methods.items()}
        clusters=[np.flatnonzero(base.source_row_id.to_numpy()==i) for i in sorted(base.source_row_id.unique())]
        common={'cohort':cohort,'n_variants':len(base),'n_source_ids':len(clusters),'seed':42,'bootstrap_repetitions':repetitions,
                'new_normalization_human_validated':False}
        if len(clusters)<30:rows.append({**common,'status':'insufficient_source_ids'});continue
        def values(index):
            results={s:paired_scores(g.gold_id.to_numpy()[index],g.clean_id.to_numpy()[index],g.predicted_id.to_numpy()[index]) for s,g in methods.items()}
            contrasts={}
            for reference in SIMPLE+('RAW',):
                for metric in ['clean_macro_f1','perturbed_macro_f1','flip_rate','macro_f1_drop']:
                    contrasts[(reference,metric)]=results['NASN2'][metric]-results[reference][metric]
            return results,contrasts
        point,delta=values(np.arange(len(base)));draws={(s,m):[] for s,v in point.items() for m in v};dd={key:[] for key in delta}
        rng=np.random.default_rng(42)
        for _ in range(repetitions):
            ix=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
            v,d=values(ix)
            for s,metrics in v.items():
                for metric,value in metrics.items():draws[(s,metric)].append(value)
            for key,value in d.items():dd[key].append(value)
        for s,metrics in point.items():
            for metric,value in metrics.items():rows.append({**common,'status':'computed','strategy':s,'metric':metric,'estimate':value,
                'ci_low':float(np.quantile(draws[(s,metric)],.025)),'ci_high':float(np.quantile(draws[(s,metric)],.975))})
        for (reference,metric),value in delta.items():differences.append({**common,'status':'computed','method':'NASN2','reference':reference,
            'metric':metric,'direction':'NASN2_minus_reference','estimate':value,
            'ci_low':float(np.quantile(dd[(reference,metric)],.025)),'ci_high':float(np.quantile(dd[(reference,metric)],.975))})
    return pd.DataFrame(rows),pd.DataFrame(differences)


def scientific_decision(pooled,contrasts,clean_metrics):
    primary=pooled[pooled.cohort.eq('source_human_confirmed')&pooled.metric.eq('perturbed_macro_f1')].set_index('strategy')
    require(set(STRATEGIES)<=set(primary.index),'Incomplete primary metrics; decision undetermined')
    best=max(SIMPLE,key=lambda s:(float(primary.at[s,'estimate']),-SIMPLE.index(s)))
    checks=[]
    for baseline in SIMPLE:
        part=contrasts[contrasts.cohort.eq('source_human_confirmed')&contrasts.reference.eq(baseline)&contrasts.metric.eq('perturbed_macro_f1')]
        require(len(part)==1,'Incomplete paired primary contrast')
        clean_gain=float(clean_metrics.loc['NASN2','macro_f1']-clean_metrics.loc[baseline,'macro_f1'])
        checks.append({'baseline':baseline,'paired_gain':float(part.iloc[0].estimate),
            'paired_ci_low':float(part.iloc[0].ci_low),'clean_all_development_macro_f1_gain':clean_gain,
            'supports':bool(part.iloc[0].ci_low>0 and clean_gain>=0)})
    passed=all(c['supports'] for c in checks)
    return {'decision':'provisional_performance_gate_passed_semantic_and_provenance_validation_pending' if passed else
        'abandon_improvement_claim_or_reframe_as_preprocessing_audit', 'strongest_simple_baseline':best,'checks':checks,
        'human_semantic_validation_complete':False,'reviewer_independence_verified':False,
        'novelty_or_statistical_significance_claim':False,'post_hoc_rule_additions_recommended':False}


def run(args):
    output=fresh(args.output);started=time.perf_counter()
    status={'status':'initializing','training_performed':False,'test_text_read':False,'test_evaluated':False,'metrics_computed':False}
    try:
        prepared=Path(args.prepared);registry(prepared);evidence=read(prepared/'evidence.json')
        require(evidence['v2_rules_sha256']==sha256(RULES) and evidence['v2_normalization_code_sha256']==sha256(Path(__file__).with_name('normalization_v2.py'))
                and evidence['runner_code_sha256']==sha256(Path(__file__)),'Frozen v2 code/rules changed')
        require(evidence['frozen_protocol_sha256']==sha256('docs/task07_protocol.md'),'Frozen decision protocol changed')
        config=read('configs/marbert_v1.json');validation,_,current=validation_inputs(config)
        verify_external_reports(BASELINE);baseline,hashes,_=verify_run(BASELINE,args.checkpoint,validation,current,config)
        require(hashes==evidence['checkpoint_sha256'],'Different selected model')
        import torch
        require(torch.cuda.is_available() and torch.version.cuda is not None,'CUDA GPU unavailable')
        free,total=torch.cuda.mem_get_info();require(free>=2*1024**3 and args.batch_size>=1,'Insufficient free VRAM or invalid inference batch')
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        from .marbert import seed_all,encoded_dataset,evaluate
        seed_all(42,4);tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModelForSequenceClassification.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False,output_loading_info=True)
        require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']) and model.config.label2id==LABEL2ID,'Incomplete or wrong selected classifier')
        model.to('cuda').eval();torch.cuda.reset_peak_memory_stats()
        frame=pd.read_csv(prepared/'normalized_inputs.csv.gz',keep_default_na=False)
        require(len(frame)==62350 and not frame.duplicated(['strategy','input_id']).any(),'V2 input coverage differs')
        require(all(text_hash(t)==h for t,h in zip(frame.normalized_text,frame.normalized_sha256)),'Normalized text hashes differ')
        unique=frame.drop_duplicates('normalized_sha256').rename(columns={'normalized_text':'tweet'}).copy()
        unique['sentiment']=unique.gold_id.map({v:k for k,v in LABEL2ID.items()});unique['partition']='validation'
        loader=torch.utils.data.DataLoader(encoded_dataset(unique,tokenizer,128),batch_size=args.batch_size,shuffle=False)
        probs,_=evaluate(model,loader,'cuda');lookup=dict(zip(unique.normalized_sha256,probs))
        pred=frame.drop(columns=['text','normalized_text']);p=np.stack([lookup[h] for h in frame.normalized_sha256])
        pred['predicted_id']=p.argmax(axis=1);pred['confidence']=p.max(axis=1)
        for label,index in LABEL2ID.items():pred['probability_'+label]=p[:,index]
        raw=pred[pred.strategy.eq('RAW')&pred.kind.eq('clean')].set_index('source_row_id').loc[baseline.source_row_id]
        require(raw.predicted_id.tolist()==baseline.predicted_id.tolist() and np.allclose(raw[[f'probability_{s}' for s in LABEL2ID]],baseline[[f'probability_{s}' for s in LABEL2ID]],atol=1e-5,rtol=1e-5),'Reloaded RAW differs from Task 03B baseline')
        pred.to_csv(output/'development_predictions.csv.gz',index=False,compression='gzip')
        renamed=pred.copy();renamed.strategy=renamed.strategy.map({v:k for k,v in METHOD_MAP.items()})
        metrics,deltas=v1_tables(renamed,2000)
        metrics.strategy=metrics.strategy.map(METHOD_MAP);deltas.strategy=deltas.strategy.map(METHOD_MAP);deltas.reference='RAW'
        metrics.to_csv(output/'rule_intensity_metrics.csv',index=False);deltas.to_csv(output/'rule_intensity_vs_RAW.csv',index=False)
        pooled,contrasts=cluster_metrics(pred,2000)
        pooled.to_csv(output/'pooled_cluster_metrics.csv',index=False);contrasts.to_csv(output/'NASN2_vs_simple_paired.csv',index=False)
        from .metrics import calculate
        clean=[]
        for s,g in pred[pred.kind.eq('clean')].groupby('strategy'):
            values,_,_=calculate(g.gold_id,g.predicted_id);clean.append({'strategy':s,**values})
        clean=pd.DataFrame(clean).set_index('strategy');clean.to_csv(output/'all_clean_metrics.csv')
        from .task04 import paired_bootstrap
        clean_rows=[]
        raw_clean=pred[pred.kind.eq('clean')&pred.strategy.eq('RAW')].set_index('source_row_id')
        for s,g in pred[pred.kind.eq('clean')].groupby('strategy'):
            pairs=g.copy();pairs['clean_id']=pairs.source_row_id.map(raw_clean.predicted_id)
            for metric,value in paired_bootstrap(pairs,2000).items():
                clean_rows.append({'strategy':s,'reference':'RAW_clean','condition':'normalized_clean_not_perturbed',
                    'n_source_ids':len(g),'metric':metric,**value})
        pd.DataFrame(clean_rows).to_csv(output/'all_clean_paired_intervals.csv',index=False)
        save(output/'scientific_decision.json',scientific_decision(pooled,contrasts,clean))
        status.update(status='completed',metrics_computed=True,checkpoint_reload_verified=True,checkpoint_sha256=hashes,
            gpu=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,free_vram_before_model=free,total_vram=total,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),inference_batch_size=args.batch_size,
            precision='float32',seed=42,max_length=128,bootstrap_repetitions=2000,versions=library_versions(),
            prepared_registry_sha256=sha256(prepared/'artifact_hashes.json'),duration_seconds=time.perf_counter()-started,
            new_normalization_human_validated=False,reviewer_independence_verified=False)
        save(output/'loading_info.json',loading)
    except Exception as exc:
        status.update(status='blocked_or_failed',reason=f'{type(exc).__name__}: {exc}',duration_seconds=time.perf_counter()-started)
        save(output/'scientific_decision.json',{'decision':'undetermined_missing_v2_model_evaluation',
            'do_not_claim_improvement':True,'do_not_add_post_hoc_rules':True})
        save(output/'run_status.json',status);seal(output);raise
    save(output/'run_status.json',status);seal(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ['diagnose','prepare']:
        q=sub.add_parser(command);q.add_argument('--v1-prepared',required=True);q.add_argument('--tokenizer-directory',required=True);q.add_argument('--output',required=True)
        if command=='diagnose':q.add_argument('--task06-intake',required=True)
    q=sub.add_parser('run');q.add_argument('--prepared',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--output',required=True);q.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args(argv)
    try:globals()[args.command](args)
    except Exception as exc:p.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
