"""Lossless adjudication intake with explicit, audited Excel formatting repairs."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from arsarcasm_audit.audit import sha256
from .preflight import require
from .task05 import (VISIBLE,HUMAN,DECISIONS,csv,registry,save,seal,fresh,validate_decisions,
                     resolve_reviews,import_reviews)
from .review_workbooks import read_workbook


def canonicalize(frame,template,timezone=None,restore_edge_spaces=False):
    require(set(frame.columns)-{'diffs'}==set(template.columns),'Adjudication columns changed')
    require(not frame.variant_id.duplicated().any() and set(frame.variant_id)==set(template.variant_id),
            'Adjudication assignments changed or duplicated')
    frame=frame.set_index('variant_id').loc[template.variant_id].reset_index()
    result=frame[template.columns].copy()
    changes=[]
    for field in VISIBLE:
        for index in result.index[result[field].ne(template[field])]:
            before,expected=result.at[index,field],template.at[index,field]
            require(field in ['original_text','perturbed_text'] and restore_edge_spaces and
                    before==expected.strip(' '),f'Adjudication protected field differs: {field}, {result.at[index,"variant_id"]}')
            digest_field='original_sha256' if field=='original_text' else 'perturbed_sha256'
            require(hashlib.sha256(expected.encode()).hexdigest()==template.at[index,digest_field],
                    'Trusted template text checksum differs')
            result.at[index,field]=expected
            changes.append({'variant_id':result.at[index,'variant_id'],'field':field,
                'repair':'restore_registered_ASCII_edge_spaces',
                'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                'after_sha256':hashlib.sha256(expected.encode()).hexdigest(),
                'added_leading_spaces':len(expected)-len(expected.lstrip(' ')),
                'added_trailing_spaces':len(expected)-len(expected.rstrip(' '))})
    zone=ZoneInfo(timezone) if timezone else None
    for index in result.index:
        original=result.at[index,'review_timestamp']
        try: timestamp=dt.datetime.fromisoformat(original.replace('Z','+00:00'))
        except ValueError as exc:raise ValueError('Invalid adjudicator timestamp') from exc
        if timestamp.tzinfo is None:
            require(zone is not None,'Naive Excel timestamps: explicit reviewer timezone required')
            timestamp=timestamp.replace(tzinfo=zone)
            result.at[index,'review_timestamp']=timestamp.isoformat()
            changes.append({'variant_id':result.at[index,'variant_id'],'field':'review_timestamp',
                'repair':'attach_explicitly_declared_timezone','before':original,
                'after':timestamp.isoformat(),'timezone':timezone})
    # Decisions, scope notes, rationale and reviewer ID must remain verbatim.
    for field in HUMAN:
        if field!='review_timestamp':require(result[field].tolist()==frame[field].tolist(),f'Human decision changed: {field}')
    return result,pd.DataFrame(changes)


def intake(review_package,independent_review,workbook,output,timezone=None,restore_edge_spaces=False):
    package,independent=Path(review_package),Path(independent_review)
    require({'reviewer_a.csv'}<=set(registry(package)),'Review package template missing')
    require({'independent_reviewer_a.csv','independent_reviewer_b.csv','adjudication_queue.csv'}<=set(registry(independent)),
            'Independent human review registry incomplete')
    template=csv(package/'reviewer_a.csv')
    a=validate_decisions(independent/'independent_reviewer_a.csv',template)
    b=validate_decisions(independent/'independent_reviewer_b.csv',template)
    _,expected,_,_=resolve_reviews(template,a,b)
    require(expected.equals(csv(independent/'adjudication_queue.csv')),'Adjudication queue differs from independent judgments')
    raw,metadata=read_workbook(workbook,allowed_extra=('diffs',))
    canonical,changes=canonicalize(raw,expected,timezone,restore_edge_spaces)
    output=fresh(output)
    (output/'original_adjudication.xlsx').write_bytes(Path(workbook).read_bytes())
    raw.to_csv(output/'original_cell_values.csv',index=False,encoding='utf-8-sig')
    canonical.to_csv(output/'canonical_adjudication.csv',index=False,encoding='utf-8-sig')
    changes.to_csv(output/'formatting_repairs.csv',index=False)
    if 'diffs' in raw:
        raw[['variant_id','diffs']].to_csv(output/'auxiliary_diffs.csv',index=False,encoding='utf-8-sig')
    validate_decisions(output/'canonical_adjudication.csv',expected)
    result=import_reviews(package,independent/'independent_reviewer_a.csv',
        independent/'independent_reviewer_b.csv',output/'canonical_adjudication.csv',output/'decisions')
    resolved=csv(output/'decisions'/'resolved_decisions.csv')
    human_evidence={'adjudicator_source':'human_user_declared','identity_and_independence_independently_verified':False,
        'workbook_metadata':metadata,'source_workbook_sha256':sha256(workbook),
        'independent_registry_sha256':sha256(independent/'artifact_hashes.json'),
        'review_package_registry_sha256':sha256(package/'artifact_hashes.json'),
        'resolved_decisions_registry_sha256':sha256(output/'decisions'/'artifact_hashes.json'),
        'timestamp_timezone':timezone,'edge_space_restoration_explicitly_enabled':restore_edge_spaces,
        'auxiliary_columns_kept_separate':['diffs'] if 'diffs' in raw else [],
        'formatting_repair_cells_n':len(changes),'human_decisions_modified':False,
        'model_predictions_used_for_adjudication':False,'training_performed':False,'test_text_read':False,
        'code_sha256':sha256(Path(__file__))}
    save(output/'intake_evidence.json',human_evidence)
    distributions=[]
    for field in DECISIONS:
        for value,n in resolved[field].value_counts().items():distributions.append({'field':field,'value':value,'n':int(n)})
    pd.DataFrame(distributions).to_csv(output/'final_decision_distributions.csv',index=False)
    for field in ['negation_present','negation_affects_sentiment','eligible_confirmed']:
        table=resolved.groupby([field,'eligible_confirmed']).size() if field!='eligible_confirmed' else resolved.groupby(field).size()
        table.rename('n').reset_index().to_csv(output/f'final_{field}_coverage.csv',index=False)
    status={**result,'intake_status':'completed','robustness_metrics_computed':False,
            'test_evaluated':False,'training_performed':False}
    save(output/'run_status.json',status);seal(output)
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['review-package','independent-review','workbook','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--timestamp-timezone')
    parser.add_argument('--restore-edge-spaces',action='store_true')
    args=parser.parse_args(argv)
    try:print(json.dumps(intake(args.review_package,args.independent_review,args.workbook,args.output,args.timestamp_timezone,args.restore_edge_spaces),indent=2))
    except Exception as exc:parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
