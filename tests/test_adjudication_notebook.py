"""The adjudication downloader must not label cases, train or load a model."""
import ast

import nbformat


def test_adjudication_notebook_pins_verified_inputs_and_preserves_blank_decisions():
    notebook=nbformat.read('notebooks/task05_adjudication_colab.ipynb',as_version=4)
    nbformat.validate(notebook)
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert 'd3658b4da959913530f77a26f6b3aa115b849493' in source
    assert "'openpyxl==3.1.5','et_xmlfile==2.0.0'" in source
    assert "'export-adjudication'" in source and 'adjudication_assignment.csv' in source
    assert "status['assigned_n']==213" in source
    assert "status['human_decisions_generated'] is False" in source
    assert "files.download(str(downloads/'adjudication.zip'))" in source
    assert 'actual[columns].equals(original[columns])' in source
    assert "'MPLBACKEND':'Agg'" in source
    for forbidden in ['testing_data.csv','AutoModel',"'train'","'evaluate'",'llm-annotator.xlsx']:
        assert forbidden not in source
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is None and not cell.outputs
            ast.parse(cell.source)
