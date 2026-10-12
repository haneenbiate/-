"""Synthetic bundles test integrity handling, not experiment completion claims."""
import hashlib
import json
import stat
import zipfile

import pytest

from arsarcasm_audit.audit import DataValidationError
from nasn.task04_bundle import import_bundle


def bundle(path,mutation=None):
    status={'status':'completed','training_performed':False,'test_text_read':False,'test_evaluated':False}
    if mutation=='test':status['test_evaluated']=True
    files={'inference/run_status.json':json.dumps(status).encode(),
           'runtime_provenance.json':json.dumps({'status':status}).encode(),
           'inference/evidence.json':b'{}','inference/clean_predictions.csv':b'fixture\n',
           'inference/perturbed_predictions.csv':b'fixture\n','inference/robustness_metrics.csv':b'fixture\n'}
    hashes={n:hashlib.sha256(d).hexdigest() for n,d in files.items()}
    if mutation=='checksum':hashes['inference/clean_predictions.csv']='0'*64
    if mutation=='coverage':hashes.pop('inference/evidence.json')
    files['artifact_hashes.json']=json.dumps(hashes).encode()
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('inference/',b'')
        for n,d in files.items():
            info=zipfile.ZipInfo(n)
            if mutation=='symlink' and n=='inference/evidence.json':info.external_attr=(stat.S_IFLNK|0o777)<<16
            z.writestr(info,d)
        if mutation=='path':z.writestr('../outside',b'not an instruction')
    return files,hashes


def test_nested_registry_import_preserves_all_original_bytes(tmp_path):
    path=tmp_path/'source.zip';files,hashes=bundle(path)
    out=tmp_path/'out';result=import_bundle(path,out)
    for name,data in files.items():assert (out/name).read_bytes()==data
    derived=json.loads((out/'inference/artifact_hashes.json').read_text())
    assert derived=={n.removeprefix('inference/'):h for n,h in hashes.items() if n.startswith('inference/')}
    assert result['archive_contains_checkpoint_weights'] is False


@pytest.mark.parametrize('mutation',['checksum','coverage','symlink','path','test'])
def test_invalid_archive_fails_before_extracting(tmp_path,mutation):
    path=tmp_path/'source.zip';bundle(path,mutation)
    with pytest.raises(DataValidationError):import_bundle(path,tmp_path/'out')
    assert not (tmp_path/'out').exists()
