"""Audit submitted annotations; workbook names never establish human provenance."""
import argparse
import datetime as dt
import json
import zipfile
import hashlib
from pathlib import Path

import openpyxl
import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import require, library_versions
from .task05 import (DECISIONS, HUMAN, VISIBLE, agreement, csv, fresh, html_review,
                     registry, review_template, save, seal, validate_decisions)

SCOPE = ['negation_cue','original_negation_scope','perturbed_negation_scope']


def read_workbook(path):
    """Read literal cell values only. Reject formulas rather than use cached results."""
    path=Path(path)
    require(path.is_file() and path.suffix.lower()=='.xlsx', 'Expected an existing XLSX file')
    require(path.stat().st_size<=10*1024**2, 'Review workbook exceeds size limit')
    with zipfile.ZipFile(path) as bundle:
        require(sum(i.file_size for i in bundle.infolist())<=50*1024**2, 'Expanded workbook too large')
        require(not any('vbaProject' in i.filename for i in bundle.infolist()), 'Macros are not supported')
    workbook=openpyxl.load_workbook(path,read_only=True,data_only=False,keep_links=False)
    try:
        require(len(workbook.sheetnames)==1, 'Exactly one annotation worksheet required; do not silently select a sheet')
        sheet=workbook.active
        require(sheet.max_row<=1000 and sheet.max_column==len(VISIBLE+HUMAN), 'Unexpected worksheet dimensions')
        for row in sheet.iter_rows():
            require(not any(cell.data_type=='f' for cell in row), 'Formula cells are prohibited in review inputs')
        metadata={'sheet_name':sheet.title,'creator':workbook.properties.creator,
                  'last_modified_by':workbook.properties.lastModifiedBy,
                  'formula_cells':0,'workbook_sha256':sha256(path)}
    finally:
        workbook.close()
    frame=pd.read_excel(path,sheet_name=metadata['sheet_name'],dtype=str,keep_default_na=False,engine='openpyxl')
    require(not any(str(c).startswith('Unnamed:') for c in frame.columns), 'Missing or duplicate worksheet header')
    return frame,metadata


def disagreements(a,b):
    pair=a.merge(b,on='variant_id',suffixes=('_a','_b'),validate='one_to_one')
    result=pair[['variant_id']].copy()
    for field in DECISIONS+SCOPE:
        result[field+'_differs']=pair[field+'_a']!=pair[field+'_b']
    result['categorical_disagreement']=result[[x+'_differs' for x in DECISIONS]].any(axis=1)
    result['scope_text_difference']=result[[x+'_differs' for x in SCOPE]].any(axis=1)
    result['requires_inspection']=result.categorical_disagreement|result.scope_text_difference
    # String differences identify inspection needs, not linguistic disagreement proof.
    return result


def pair_reports(frames):
    reports,matrices=[],[]
    for left,right in [('a','b'),('a','llm'),('b','llm')]:
        pair,rows,_=agreement(frames[left],frames[right],404)
        for row in rows:
            reports.append({'comparison':left+'_vs_'+right,'interpretation':'agreement_between_submitted_annotations_not_validated_human_IAA',**row})
        for field in DECISIONS:
            counts=pair.groupby([field+'_a',field+'_b']).size()
            for (x,y),n in counts.items():
                matrices.append({'comparison':left+'_vs_'+right,'field':field,'left_value':x,'right_value':y,'n':int(n)})
    return pd.DataFrame(reports),pd.DataFrame(matrices)


def distribution_tables(frames,metadata):
    counts=[]
    for role,frame in frames.items():
        for field in DECISIONS:
            for value,n in frame[field].value_counts().items():
                counts.append({'source':role,'field':field,'value':value,'n':int(n),'denominator':len(frame)})
    coverage=metadata[['variant_id','rule','intensity','gold_label','dialect','sarcasm','has_negation']].copy()
    for role in frames:
        coverage[role+'_submitted']=coverage.variant_id.isin(frames[role].variant_id).astype(int)
    coverage=coverage.groupby(['rule','intensity','gold_label','dialect','sarcasm','has_negation']).agg(
        assigned_n=('variant_id','size'), a_submitted=('a_submitted','sum'),
        b_submitted=('b_submitted','sum'), llm_submitted=('llm_submitted','sum')).reset_index()
    coverage['human_provenance_verified']=False
    return pd.DataFrame(counts),coverage


def note_flags(frame):
    """Evidence flags never classify a person or infer the truth of a review."""
    return {
        'explicit_ai_assistance_disclaimer_n':int(frame.linguistic_notes.str.contains('وسم أولي بمساعدة نموذج',regex=False).sum()),
        'explicit_not_independent_human_disclaimer_n':int(frame.linguistic_notes.str.contains('ليس تحكيمًا بشريًا مستقلًا',regex=False).sum()),
        'unique_timestamp_n':int(frame.review_timestamp.nunique()),
        'reviewer_identifier_n':int(frame.reviewer_id.nunique()),
        'negation_uncertain_n':int(frame.negation_present.eq('uncertain').sum()),
        'negation_effect_uncertain_n':int(frame.negation_affects_sentiment.eq('uncertain').sum())}


def provenance_declaration(path,source_metadata):
    if path is None:return None
    declaration=json.loads(Path(path).read_text())
    require(declaration.get('source_roles')=={'a':'human_user_declared','b':'human_user_declared','llm':'automated_llm'},
            'Source declaration must explicitly distinguish human and LLM roles')
    require(isinstance(declaration.get('declaration_basis'),str) and declaration['declaration_basis'].strip(),
            'Source declaration basis is required')
    require(declaration.get('source_sha256')=={role:meta['workbook_sha256'] for role,meta in source_metadata.items()},
            'Source declaration does not match these exact uploaded workbooks')
    return declaration


def audit(package,a_path,b_path,llm_path,output,provenance=None):
    package=Path(package)
    require({'reviewer_a.csv','queue_registry.csv','evidence.json'}<=set(registry(package)), 'Incomplete review package')
    template=csv(package/'reviewer_a.csv')
    require(len(template)==404 and template[HUMAN].eq('').all().all(), 'Expected untouched blank 404-item assignment template')
    frames,source_metadata={},{}
    # Validate every input before creating the result directory.
    raw={}
    for role,path in [('a',a_path),('b',b_path),('llm',llm_path)]:
        frame,meta=read_workbook(path)
        require(len(frame)==404,'Each submitted file must contain all 404 assignments')
        require(set(frame.columns)==set(template.columns),'Review worksheet columns changed')
        require(not frame.variant_id.duplicated().any() and set(frame.variant_id)==set(template.variant_id), 'Review assignment IDs changed')
        frame=frame.set_index('variant_id').loc[template.variant_id].reset_index()
        for field in VISIBLE:
            require(frame[field].tolist()==template[field].tolist(),f'Altered original assignment field: {field}')
        frames[role]=frame;source_metadata[role]=meta;raw[role]=Path(path)
    declaration=provenance_declaration(provenance,source_metadata)
    output=fresh(output)
    for role,frame in frames.items():
        path=output/f'submitted_{role}.csv'
        frame.to_csv(path,index=False,encoding='utf-8-sig')
        require(len(validate_decisions(path,template))==404,'Incomplete submitted judgments')
    agreement_table,matrix=pair_reports(frames)
    if declaration:
        agreement_table.loc[agreement_table.comparison.eq('a_vs_b'),'interpretation']='agreement_between_user_declared_human_reviewers_independence_not_independently_verified'
        agreement_table.loc[~agreement_table.comparison.eq('a_vs_b'),'interpretation']='LLM_agreement_with_user_declared_reviews_not_accuracy_against_adjudicated_gold'
    differences=disagreements(frames['a'],frames['b'])
    metadata=csv(package/'queue_registry.csv')
    distributions,coverage=distribution_tables(frames,metadata)
    summary=[]
    for field in DECISIONS+SCOPE:
        summary.append({'field':field,'different_n':int(differences[field+'_differs'].sum()),'paired_n':404,
                        'interpretation':'categorical disagreement' if field in DECISIONS else 'literal scope wording difference; needs inspection'})
    # Lexical cues are an existing stratification heuristic, not linguistic ground truth.
    lexical=[]
    for role,frame in frames.items():
        joined=metadata[['variant_id','has_negation']].merge(frame[['variant_id','negation_present','negation_affects_sentiment']],on='variant_id',validate='one_to_one')
        for (flag,presence,effect),n in joined.groupby(['has_negation','negation_present','negation_affects_sentiment']).size().items():
            lexical.append({'source':role,'task04_lexical_negation_flag':flag,'submitted_negation_present':presence,
                            'submitted_negation_affects_sentiment':effect,'n':int(n),'interpretation':'comparison_to_heuristic_not_accuracy'})
    agreement_table.to_csv(output/'submitted_annotation_agreement.csv',index=False)
    (output/'submitted_annotation_agreement.tex').write_text(agreement_table.drop(columns='interpretation').to_latex(index=False,float_format='%.4f'))
    matrix.to_csv(output/'agreement_contingency.csv',index=False)
    differences.to_csv(output/'pairwise_inspection_flags.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'disagreement_summary.csv',index=False)
    distributions.to_csv(output/'decision_distributions.csv',index=False)
    coverage.to_csv(output/'submitted_coverage.csv',index=False)
    pd.DataFrame(lexical).to_csv(output/'negation_heuristic_comparison.csv',index=False)
    inspect_ids=differences.loc[differences.requires_inspection,'variant_id']
    blank=template[template.variant_id.isin(inspect_ids)].copy()
    blank.to_csv(output/'inspection_queue.csv',index=False,encoding='utf-8-sig')
    html_review(blank,output/'inspection_queue.html')
    for role in ['a','b']:
        frames[role][frames[role].variant_id.isin(inspect_ids)].to_csv(output/f'inspection_source_{role}.csv',index=False,encoding='utf-8-sig')
    for role,path in raw.items():
        # Preserve exact uploaded bytes privately beside lossless string CSV conversion.
        (output/f'original_{role}.xlsx').write_bytes(path.read_bytes())
    flags={role:note_flags(frame) for role,frame in frames.items()}
    status={'status':'submitted_annotations_audited_human_provenance_pending','queue_n':404,
            'submitted_a_n':404,'submitted_b_n':404,'submitted_llm_n':404,
            'categorical_disagreement_n':int(differences.categorical_disagreement.sum()),
            'literal_scope_difference_n':int(differences.scope_text_difference.sum()),
            'inspection_queue_n':len(inspect_ids),'fully_matching_records_n':404-len(inspect_ids),
            'human_provenance_verified':False,'human_confirmed_n':0,'human_review_completed':False,
            'llm_used_for_adjudication':False,'robustness_metrics_computed':False,
            'checkpoint_loaded':False,'training_performed':False,'test_text_read':False,'test_evaluated':False,
            'remaining':['Clarify independent human involvement in A and B; file metadata cannot prove it',
                         'Resolve disagreements through explicit qualified third-human adjudication',
                         'Supply LLM model/version/prompt/settings provenance',
                         'Supply completed Task 04 perturbed predictions and verified tokenizer access']}
    if declaration:
        from .task05 import resolve_reviews
        _,_,_,human_coverage=resolve_reviews(template,frames['a'],frames['b'])
        status.update(status='user_declared_human_reviews_audited_pending_adjudication',
                      human_source_user_declared=True,
                      human_consensus_eligible_n=human_coverage['confirmed_n'],
                      human_confirmed_n=human_coverage['confirmed_n'],
                      source_provenance_basis='user_declaration_not_independent_identity_verification',
                      remaining=['Complete explicit third-human adjudication for all flagged cases',
                                 'Supply LLM model/version/prompt/settings provenance',
                                 'Supply completed Task 04 perturbed predictions and verified tokenizer access'])
    evidence={'source_workbooks':source_metadata,'source_flags':flags,
              'review_package_registry_sha256':sha256(package/'artifact_hashes.json'),
              'task05_preparation_evidence':json.loads((package/'evidence.json').read_text()),
              'llm_declared_identifier':frames['llm'].reviewer_id.unique().tolist(),
              'llm_model_and_revision_verified':False,'llm_prompt_available':False,'llm_sampling_settings_available':False,
              'recorded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'versions':{**library_versions(),'openpyxl':openpyxl.__version__},
              'code_sha256':{'review_workbooks.py':sha256(Path(__file__)),'task05.py':sha256(Path(__file__).with_name('task05.py'))}}
    evidence['source_declaration']=declaration
    evidence['source_declaration_sha256']=sha256(provenance) if provenance else None
    save(output/'evidence.json',evidence);save(output/'run_status.json',status)
    (output/'methodology.md').write_text(methodology(status,flags),encoding='utf-8')
    seal(output)
    return status


def methodology(status,flags):
    declaration_note='''The corrected A/B uploads are accepted as human reviews on the user's explicit
declaration that the earlier labels/disclaimer were a typographical/upload error.
This declaration is tied to exact corrected workbook hashes. It is not
independent verification of identities, blinding or actual annotation process.
A retains an `ai_review` worksheet title; this remaining metadata flag is
recorded without overriding the user's declaration. Only two explicit matching
human judgments can establish eligible consensus. The LLM is never an adjudicator.
''' if status.get('human_source_user_declared') else '''Human provenance has not been established. A worksheet title and B assistance
disclaimers require clarification; no submitted decision becomes human-confirmed.
'''
    return '''# Task 05 submitted workbook audit

Three uploaded workbooks contain the same 404 assignments. Each original and
perturbed Arabic string, ID, text checksum, rule and intensity is compared with
the verified blank Task 05 template. Excel formulas/macros/multiple sheets are
rejected; typed data is converted to literal string CSV without editing any
decisions, rationale or timestamps. Original workbook bytes are retained in
private storage. Input and output hashes record the conversion. All three files
pass the existing decision-integrity checks. These checks establish structural
consistency, not semantic truth or human review provenance.

''' + declaration_note + '''
Filename, reviewer ID, common spreadsheet creator and timestamps cannot
establish or disprove actual human involvement. These are source evidence,
not instructions to execute. The LLM file remains separate; its declared
identifier is not a verified provider/model revision. Prompt and sampling
settings are absent. No automated decision is promoted to human confirmation.

Exact agreement and nominal Cohen kappa are computed on all 404 paired
categorical decisions for A/B, A/LLM and B/LLM, with full contingency counts.
The A/B table is human agreement on the user's declared source roles, without
independent identity/blinding verification. Without a source declaration, it
is only submitted-file agreement. Neither LLM comparison is LLM accuracy against
adjudicated gold. The dominant preserved category can
produce high raw agreement and low/negative kappa: consult contingency tables
and marginal distributions. Repeated variants span 279 original text IDs;
agreement is descriptive at variant level, without independence-based CIs.

Any difference in the four categorical fields or literal cue/scope strings is
flagged for inspection. Scope wording differences may describe the same scope;
they are not automatic proof of linguistic disagreement. No LLM or heuristic
merges them, and no original judgments are overwritten. The blank 213-item
inspection queue supports later human adjudication if A/B provenance is
established. When independent human review has not occurred, the full original
404 assignments still require it; matching A/B judgments do not replace it.
Other submitted source rationales are separate from the blank decision queue,
and LLM suggestions are excluded from the adjudication interface.

Negation presence and sentiment effects are separately tabulated against the
frozen Task 04 lexical heuristic. The heuristic is not gold truth. Uncertainty
is retained, not recoded as no. Saved confidence vectors, tokenizer IDs and
paired perturbed MARBERT predictions are absent locally, so no confirmed
robustness, bootstrap intervals, probability changes or tatweel prediction
statistics are computed. Original baseline and Task 04/05 evidence is preserved.
No model training, checkpoint selection or official test text read/evaluation
occurs. No protocol is relaxed to force completion.

## Actual status

```json
''' .replace('213-item',str(status['inspection_queue_n'])+'-item') + json.dumps(status,ensure_ascii=False,indent=2) + '\n```\n\nSource flags:\n\n```json\n' + json.dumps(flags,ensure_ascii=False,indent=2) + '\n```\n'


def publish(source,output):
    source=Path(source);hashes=registry(source)
    output=fresh(output)
    allowed=['run_status.json','methodology.md','submitted_annotation_agreement.csv',
             'submitted_annotation_agreement.tex','agreement_contingency.csv','pairwise_inspection_flags.csv',
             'disagreement_summary.csv','decision_distributions.csv','submitted_coverage.csv','negation_heuristic_comparison.csv']
    for name in allowed:
        require(name in hashes,f'Unregistered report: {name}')
        (output/name).write_bytes((source/name).read_bytes())
    assignments=csv(source/'inspection_queue.csv')
    assignments.drop(columns=['original_text','perturbed_text']+HUMAN).to_csv(output/'adjudication_assignment.csv',index=False)
    evidence=json.loads((source/'evidence.json').read_text())
    public={'source_sha256':{role:meta['workbook_sha256'] for role,meta in evidence['source_workbooks'].items()},
            'source_flags':evidence['source_flags'],
            'worksheet_a_ai_review_title_flag':'ai_review' in evidence['source_workbooks']['a']['sheet_name'].lower(),
            'all_workbooks_same_creator_metadata':len({m['creator'] for m in evidence['source_workbooks'].values()})==1,
            'llm_declared_identifier':evidence['llm_declared_identifier'],'llm_model_and_revision_verified':False,
            'llm_prompt_available':False,'llm_sampling_settings_available':False,
            'private_audit_artifact_sha256':hashes,'review_package_registry_sha256':evidence['review_package_registry_sha256'],
            'versions':evidence['versions'],'code_sha256':evidence['code_sha256'],
            'source_declaration':evidence['source_declaration'],
            'source_declaration_sha256':evidence['source_declaration_sha256'],
            'human_provenance_verified':False,'test_text_read':False,'test_evaluated':False}
    public['human_confirmed_n']=json.loads((source/'run_status.json').read_text())['human_confirmed_n']
    save(output/'evidence.json',public)
    seal(output)
    return {'status':'published_text_free_annotation_audit','source':str(source),'output':str(output)}


def export_adjudication(preparation,assignments,output):
    """Recover exact private text from registered candidates, not public raw tweets."""
    preparation,assignments=Path(preparation),Path(assignments)
    require(assignments.name in registry(assignments.parent),'Unregistered adjudication assignments')
    require({'perturbation_manifest.csv','human_review_queue.csv'} <= set(registry(preparation)),
            'Task 04 full-text preparation evidence incomplete')
    needed=csv(assignments)
    expected_columns=[c for c in VISIBLE if c not in ['original_text','perturbed_text']]
    require(set(needed.columns)==set(expected_columns) and not needed.variant_id.duplicated().any(),
            'Invalid adjudication assignment schema or duplicate IDs')
    original_queue=csv(preparation/'human_review_queue.csv')
    require(set(needed.variant_id)<=set(original_queue.variant_id),'Adjudication ID outside original human queue')
    manifest=csv(preparation/'perturbation_manifest.csv').set_index('variant_id')
    full=manifest.loc[needed.variant_id].reset_index()
    require(full.partition.eq('validation').all(),'Adjudication must use validation only')
    for field in expected_columns:
        require(full[field].tolist()==needed[field].tolist(),f'Adjudication identity mismatch: {field}')
    for text,digest in [('original_text','original_sha256'),('perturbed_text','perturbed_sha256')]:
        require(full[text].map(lambda x:hashlib.sha256(x.encode('utf-8')).hexdigest()).tolist()==full[digest].tolist(),
                f'Adjudication text hash mismatch: {text}')
    template=review_template(full)
    output=fresh(output)
    template.to_csv(output/'adjudication.csv',index=False,encoding='utf-8-sig')
    html_review(template,output/'adjudication.html')
    protocol=Path('docs/task05_review_protocol_ar.md').read_bytes()
    (output/'review_protocol_ar.md').write_bytes(protocol)
    checks={name:sha256(output/name) for name in ['adjudication.csv','adjudication.html','review_protocol_ar.md']}
    save(output/'package_checksums.json',checks)
    with zipfile.ZipFile(output/'adjudication.zip','w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(list(checks)+['package_checksums.json']):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o600<<16
            bundle.writestr(info,(output/name).read_bytes())
    status={'status':'adjudication_download_prepared','assigned_n':len(template),
            'human_decisions_generated':False,'model_predictions_in_package':False,
            'zip_sha256':sha256(output/'adjudication.zip'),'assignments_sha256':sha256(assignments),
            'source_manifest_sha256':sha256(preparation/'perturbation_manifest.csv'),
            'test_text_read':False,'test_evaluated':False}
    save(output/'run_status.json',status);seal(output)
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    ingest=sub.add_parser('audit')
    for name in ['review-package','reviewer-a','reviewer-b','llm','output']:
        ingest.add_argument('--'+name,required=True)
    ingest.add_argument('--source-provenance')
    public=sub.add_parser('publish')
    for name in ['source','output']:public.add_argument('--'+name,required=True)
    export=sub.add_parser('export-adjudication')
    for name in ['preparation','assignments','output']:export.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='audit':result=audit(args.review_package,args.reviewer_a,args.reviewer_b,args.llm,args.output,args.source_provenance)
        elif args.command=='publish':result=publish(args.source,args.output)
        else:result=export_adjudication(args.preparation,args.assignments,args.output)
        print(json.dumps(result,indent=2))
    except Exception as exc:
        parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
