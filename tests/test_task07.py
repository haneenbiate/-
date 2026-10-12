import pandas as pd
import pytest

from nasn.task07 import STRATEGIES,SIMPLE,cluster_metrics,scientific_decision
from nasn.task07_io import compare_report,import_bundle


def synthetic_predictions():
    rows=[]
    for s in STRATEGIES:
        for i in range(30):
            common={'strategy':s,'source_row_id':str(i),'gold_id':i%3,'source_review_status':'eligible_user_declared_human'}
            rows.append({**common,'input_id':'c'+str(i),'kind':'clean','predicted_id':i%3})
            for j in range(2):
                rows.append({**common,'input_id':f'p{i}:{j}','kind':'perturbed',
                    'predicted_id':i%3 if s=='NASN2' else (i+1)%3})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_repeated_originals_and_joint_contrast():
    metrics,delta=cluster_metrics(synthetic_predictions(),100)
    assert metrics.n_variants.eq(60).all() and metrics.n_source_ids.eq(30).all()
    gains=delta[delta.metric.eq('perturbed_macro_f1')]
    assert gains.estimate.eq(1).all()
    clean=pd.DataFrame({'macro_f1':[1]*5},index=STRATEGIES)
    decision=scientific_decision(metrics,delta,clean)
    assert decision['strongest_simple_baseline']=='TAT'
    assert decision['decision'].startswith('provisional_performance_gate_passed')
    assert decision['human_semantic_validation_complete'] is False


def test_no_gain_requires_abandon_or_reframe_not_new_rules():
    frame=synthetic_predictions();frame['predicted_id']=frame.gold_id
    metrics,delta=cluster_metrics(frame,100)
    clean=pd.DataFrame({'macro_f1':[1]*5},index=STRATEGIES)
    decision=scientific_decision(metrics,delta,clean)
    assert decision['decision']=='abandon_improvement_claim_or_reframe_as_preprocessing_audit'
    assert decision['post_hoc_rule_additions_recommended'] is False


def test_metric_intake_does_not_accept_changed_scores():
    actual=pd.DataFrame([{'status':'computed','estimate':.5}])
    compare_report(actual.copy(),actual)
    with pytest.raises(ValueError,match='differs'):compare_report(actual.assign(estimate=.6),actual)


def test_unsafe_zip_is_rejected_before_extraction(tmp_path):
    import zipfile
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape','untrusted instructions are not executed')
    with pytest.raises(ValueError,match='unsafe'):import_bundle(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()
