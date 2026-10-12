"""A dose comparison requires matched original IDs, not raw cohort score differences."""
import pandas as pd

from nasn.task05 import paired_intensity_tables,verify_exported_token_flags
import pytest
from arsarcasm_audit.audit import DataValidationError


def test_matched_confirmed_intensities_use_intersection_and_report_small_cohort():
    rows=[]
    for i in range(40):
        for intensity in [1,2]:
            rows.append({'variant_id':f'v{i}-{intensity}','source_row_id':f's{i}',
                'rule':'tatweel','intensity':intensity,'applicable':True,'gold_id':i%3,
                'predicted_id':i%3,'high_risk_near_duplicate':i<10})
    changed=pd.DataFrame(rows)
    decisions=pd.DataFrame({'variant_id':changed.variant_id,'eligible_confirmed':[not v.startswith('v39-') for v in changed.variant_id]})
    metrics,ids=paired_intensity_tables(changed,decisions,100)
    selected=metrics[(metrics.cohort=='confirmed_both_intensities')&(metrics.metric=='macro_f1_drop')]
    assert selected.n_original_ids.tolist()==[39] and selected.estimate.tolist()==[0]
    assert selected.ci_low.tolist()==selected.ci_high.tolist()==[0]
    sensitive=metrics[metrics.cohort.eq('confirmed_both_exclude_high_risk')]
    assert sensitive.n_original_ids.tolist()==[29]
    assert sensitive.status.tolist()==['insufficient_sample_size']
    assert ids.both_human_confirmed.sum()==39


def test_recomputed_token_flags_reject_false_exported_flags():
    def tokenizer(text,**kwargs):
        ids=[ord(c) for c in text]
        return {'input_ids':ids[:128] if kwargs.get('truncation') else ids}
    manifest=pd.DataFrame({'variant_id':['v'],'original_text':['a'*130], 'perturbed_text':['a'*131]})
    frame=pd.DataFrame({'variant_id':['v'],'token_ids_changed':['True'],'effective_token_ids_changed':['False'],
        'clean_truncated':['True'],'perturbed_truncated':['True'],'clean_token_count':['130'],
        'perturbed_token_count':['131'],'token_count_delta':['1']})
    assert verify_exported_token_flags(frame,manifest,tokenizer)['variant_rows']==1
    frame.loc[0,'effective_token_ids_changed']='True'
    with pytest.raises(DataValidationError,match='token flag differs'):
        verify_exported_token_flags(frame,manifest,tokenizer)
