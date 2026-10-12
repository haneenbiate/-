"""Private complete text/rule/prediction examples for all v1 token exceptions."""
import argparse
import json
from pathlib import Path

import pandas as pd

from nasn.task07 import v1_inputs,verified_v1_predictions
from nasn.normalization import normalize,load_rules
from nasn.task05 import fresh,registry,save,seal
from arsarcasm_audit.audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['v1-prepared','intake','diagnosis','output']:p.add_argument('--'+n,required=True)
    args=p.parse_args();registry(args.diagnosis)
    frame,_=v1_inputs(args.v1_prepared);pred=verified_v1_predictions(args.intake).set_index(['input_id','strategy'])
    ids=pd.read_csv(Path(args.diagnosis)/'all_elongation_token_exceptions.csv').input_id
    raw=frame[frame.strategy.eq('B0')].set_index('input_id')
    clean=raw[raw.kind.eq('clean')].set_index('source_row_id')
    rows=[];rules=load_rules()
    for input_id in ids:
        source=raw.loc[input_id]
        record={'input_id':input_id,'source_row_id':source.source_row_id,'rule':source.rule,'intensity':int(source.intensity),
            'source_review_status':source.source_review_status,'original_text':clean.loc[source.source_row_id,'text'],
            'perturbed_text':source.text,'gold_id':int(source.gold_id)}
        for method in ['B0','B1','B2','B3','B4']:
            result=normalize(source.text,method,rules);model=pred.loc[(input_id,method)]
            record[method+'_normalized_text']=result.text
            record[method+'_rules']=json.dumps([e.__dict__ for e in result.edits],ensure_ascii=False)
            record[method+'_prediction']=int(model.predicted_id)
            record[method+'_confidence']=float(model.confidence)
            for label in ['NEG','NEU','POS']:record[method+'_probability_'+label]=float(model['probability_'+label])
        rows.append(record)
    output=fresh(args.output)
    pd.DataFrame(rows).to_csv(output/'all_elongation_exceptions_private.csv',index=False,encoding='utf-8-sig')
    save(output/'evidence.json',{'n_examples':len(rows),'input_ids_registry_sha256':sha256(Path(args.diagnosis)/'all_elongation_token_exceptions.csv'),
        'source_intake_status_sha256':sha256(Path(args.intake)/'intake_status.json'),
        'exporter_sha256':sha256(Path(__file__)),'new_model_inference':False,'human_validation_generated':False,
        'full_text_export_is_private':True})
    seal(output)


if __name__=='__main__':main()
