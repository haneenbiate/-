import pytest

from tools.build_task06_notebook import build


def test_notebook_is_pinned_cli_only_and_unexecuted():
    reference='a'*40
    notebook=build(reference)
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert reference in source
    assert 'nasn.task06' in source and 'best_checkpoint' in source
    assert "'--bootstrap','2000'" in source and "'max_length':128" in source
    assert "'MPLBACKEND':'Agg'" in source
    assert "'torch==2.8.0+cu126'" in source
    assert 'testing_data.csv' not in source
    assert "'train'" not in source and 'from_pretrained' not in source
    assert 'report_checksums.json' in source
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook-cell>','exec')
            assert cell['outputs']==[] and cell['execution_count'] is None


def test_unpinned_revision_rejected():
    with pytest.raises(ValueError,match='immutable'):build('main')
