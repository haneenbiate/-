"""Verify the download notebook remains review-only and safely pinned."""
import ast

import nbformat


def test_review_download_notebook_is_blinded_cpu_only_and_pinned():
    notebook=nbformat.read('notebooks/task05_review_download.ipynb',as_version=4)
    nbformat.validate(notebook)
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert '8bc384eb8dab6b0266825b3cae3f1d8c7aa96ebb' in source
    assert "'MPLBACKEND':'Agg'" in source
    assert 'actual[columns].equals(original[columns])' in source
    assert "'nasn.task05','prepare'" in source and "'tools.package_task05_review'" in source
    assert "files.download(str(downloads/'reviewer_a.zip'))" in source
    assert "files.download(str(downloads/'reviewer_b.zip'))" in source
    for forbidden in ['testing_data.csv', 'AutoModel', 'nasn.marbert', "'evaluate'", "'train'", 'cuda.is_available']:
        assert forbidden not in source
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is None and not cell.outputs
            ast.parse(cell.source)
