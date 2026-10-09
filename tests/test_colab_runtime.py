"""Offline integrity/storage tests; no Colab or MARBERT experiment results."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('colab_runtime',ROOT / 'tools/colab_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fixture_snapshot(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    files = {}
    for name in runtime.REQUIRED_FILES:
        payload = (name + ': synthetic unit-test bytes').encode()
        (source / name).write_bytes(payload)
        files[name] = {'sha256':hashlib.sha256(payload).hexdigest(),'size':len(payload)}
    monkeypatch.setattr(runtime,'WEIGHTS_SHA256',files['pytorch_model.bin']['sha256'])
    return source, {'repo_id':runtime.MODEL,'revision':runtime.REVISION,'files':files}


def test_snapshot_hashes_and_cache_staging(tmp_path,monkeypatch):
    source, lock = fixture_snapshot(tmp_path,monkeypatch)
    verified = runtime.verify_snapshot(source,lock)
    target = runtime.cache_snapshot(source,tmp_path / 'cache',lock)
    assert target.name == runtime.REVISION
    assert runtime.verify_snapshot(target,lock) == verified
    assert runtime.cache_snapshot(source,tmp_path / 'cache',lock) == target
    (source / 'vocab.txt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='mismatch'):
        runtime.verify_snapshot(source,lock)


def test_local_declaration_and_conflicting_cache_fail(tmp_path,monkeypatch):
    source, lock = fixture_snapshot(tmp_path,monkeypatch)
    runtime.save(source / 'snapshot_revision.json',{'repo_id':runtime.MODEL,'revision':'wrong'})
    with pytest.raises(ValueError,match='revision'):
        runtime.acquire_snapshot(tmp_path / 'cache',lock,tmp_path / 'report.json',source)
    assert json.loads((tmp_path / 'report.json').read_text())['weights_verified'] is False
    runtime.save(source / 'snapshot_revision.json',{'repo_id':runtime.MODEL,'revision':runtime.REVISION})
    target = runtime.acquire_snapshot(tmp_path / 'cache',lock,tmp_path / 'good.json',source)
    assert json.loads((tmp_path / 'good.json').read_text())['status'] == 'verified'
    (target / 'vocab.txt').write_bytes(b'other content')
    with pytest.raises(ValueError,match='Conflicting existing'):
        runtime.cache_snapshot(source,tmp_path / 'cache',lock)


def test_lock_not_taken_from_untrusted_snapshot():
    lock = json.loads((ROOT / 'configs/marbert_snapshot_lock.json').read_text())
    runtime.validate_lock(lock)
    assert lock['files']['pytorch_model.bin']['sha256'] == '801862ace5fc9e6b135b19476dbda5c7de1e8673b5b9d55db08d79e2c30b9dca'
    lock['files']['pytorch_model.bin']['sha256'] = 'a' * 64
    with pytest.raises(ValueError,match='Untrusted'):
        runtime.validate_lock(lock)

def test_unverified_alternative_weights_cannot_shadow_snapshot(tmp_path,monkeypatch):
    source,lock=fixture_snapshot(tmp_path,monkeypatch)
    (source / 'model.safetensors').write_bytes(b'unverified alternate weights')
    with pytest.raises(ValueError,match='Unexpected snapshot files'):
        runtime.verify_snapshot(source,lock)


@pytest.mark.parametrize('free,micro',[(16,4),(10,2),(7,1)])
def test_config_preserves_protocol(free,micro):
    original=json.loads((ROOT / 'configs/marbert_v1.json').read_text())
    gpu={'status':'gpu_gate_passed','free_vram_bytes':free * 1024**3}
    config,changes=runtime.make_config(original,gpu,'/tmp/cache')
    assert config['micro_batch_size'] == micro
    assert config['micro_batch_size'] * config['gradient_accumulation_steps'] == 16
    assert config['device'] == 'cuda'
    assert set(changes) <= {'micro_batch_size','gradient_accumulation_steps','device','cache_directory'}
    for key,value in original.items():
        if key not in changes: assert config[key] == value


@pytest.mark.parametrize('micro',[0,3,8,16,True])
def test_invalid_microbatch_fails(micro):
    original=json.loads((ROOT / 'configs/marbert_v1.json').read_text())
    with pytest.raises(ValueError,match='Allowed'):
        runtime.make_config(original,{'status':'gpu_gate_passed','free_vram_bytes':16*1024**3},'/tmp/cache',micro)


def test_no_cpu_fallback():
    original=json.loads((ROOT / 'configs/marbert_v1.json').read_text())
    with pytest.raises(ValueError,match='GPU gate'):
        runtime.make_config(original,{'status':'blocked'},'/tmp/cache')


def test_report_export_excludes_weights_and_raw_data(tmp_path):
    source=tmp_path / 'run'; source.mkdir()
    (source / 'run_status.json').write_text('{"status":"blocked"}')
    (source / 'validation_predictions.csv').write_text('source_row_id,gold_id\n')
    (source / 'pytorch_model.bin').write_bytes(b'weights')
    (source / 'training_data.csv').write_text('raw data')
    (source / 'execution.log').write_text('arbitrary log')
    (source / 'best_checkpoint').mkdir()
    (source / 'best_checkpoint/model.safetensors').write_bytes(b'weights')
    runtime.export_reports(source,tmp_path / 'export')
    assert set(p.name for p in (tmp_path / 'export').iterdir()) == {'run_status.json','validation_predictions.csv','report_checksums.json'}
    with pytest.raises(ValueError,match='already exists'):
        runtime.export_reports(source,tmp_path / 'export')


def test_incomplete_run_cannot_claim_success(tmp_path):
    with pytest.raises(ValueError,match='missing'):
        runtime.verify_completed_run(tmp_path)


def test_diagnostics_redact_signed_queries():
    message=runtime.safe_error(Exception('403 host us.aws.cdn.hf.co /weight?Signature=sensitive&Policy=private'))
    assert '403' in message and 'us.aws.cdn.hf.co' in message
    assert 'sensitive' not in message and 'private' not in message


def test_notebook_schema_and_executable_cells():
    notebook=nbformat.read(ROOT / 'notebooks/marbert_baseline_colab.ipynb',as_version=4)
    nbformat.validate(notebook)
    codes=[]
    helper_source=None
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type != 'code': continue
        assert cell.execution_count is None and not cell.outputs
        compile(cell.source,f'cell-{i}','exec')
        tree=ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node,ast.Assign):
                if any(isinstance(t,ast.Name) and t.id=='HELPER_SOURCE' for t in node.targets):
                    helper_source=ast.literal_eval(node.value)
        codes.append(cell.source)
    assert helper_source == (ROOT / 'tools/colab_runtime.py').read_text()
    all_code='\n'.join(codes)
    assert runtime.REFERENCE in all_code and 'torch==2.8.0+cu126' in all_code
    assert '"-m","nasn.marbert","train"' in all_code
    assert '"-m","arsarcasm_audit","split"' not in all_code
    assert 'drive.mount' in all_code and 'HF_HUB_OFFLINE' in all_code
    assert not any('def train_epoch' in code for code in codes)


def test_generated_json_writes_are_real_newlines():
    notebook=nbformat.read(ROOT / 'notebooks/marbert_baseline_colab.ipynb',as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for node in ast.walk(ast.parse(cell.source)):
                assert not (isinstance(node,ast.Constant) and node.value=='\\n')
