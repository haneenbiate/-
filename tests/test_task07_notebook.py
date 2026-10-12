from tools.build_task07_notebook import build


def test_v2_notebook_is_pinned_unexecuted_and_development_only():
    n=build('a'*40)
    source='\n'.join(''.join(c['source']) for c in n['cells'])
    assert 'nasn.task07_io' in source and 'nasn.task07' in source
    assert 'TASK06_ZIP' in source and 'scientific_decision.json' in source
    assert 'testing_data.csv' not in source
    assert "'MPLBACKEND':'Agg'" in source
    assert 'PRIVATE-examples-review.zip' in source
    for c in n['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),'<Task07-notebook>','exec')
            assert c['execution_count'] is None and c['outputs']==[]
