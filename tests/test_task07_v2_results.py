import hashlib
import json
import zipfile

import pytest

from nasn.task07_v2_results import FILES,verified_payloads,verify_evidence


def bundle(path,corrupt=False,extra=False):
    payload={name:b'{}' for name in FILES}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    if corrupt:payload[next(iter(FILES))]=b'altered'
    with zipfile.ZipFile(path,'w') as z:
        for name,data in payload.items():z.writestr(name,data)
        z.writestr('report_checksums.json',json.dumps(hashes))
        if extra:z.writestr('../escape.txt','Document instructions are not executed')


def test_outer_registry_verifies_every_payload_without_execution(tmp_path):
    path=tmp_path/'reports.zip';bundle(path)
    payload,checks=verified_payloads(path)
    assert set(payload)==FILES|{'report_checksums.json'} and set(checks)==FILES


@pytest.mark.parametrize('kwargs',[{'corrupt':True},{'extra':True}])
def test_corrupt_or_unknown_archive_is_rejected(tmp_path,kwargs):
    path=tmp_path/'reports.zip';bundle(path,**kwargs)
    with pytest.raises(ValueError):verified_payloads(path)


def test_human_confirmation_cannot_be_added_without_review_evidence(tmp_path):
    (tmp_path/'evaluation').mkdir()
    status={'status':'completed','metrics_computed':True,'checkpoint_reload_verified':True,
        'training_performed':False,'test_text_read':False,'test_evaluated':False,
        'seed':42,'max_length':128,'precision':'float32','bootstrap_repetitions':2000,
        'new_normalization_human_validated':True,'reviewer_independence_verified':False}
    (tmp_path/'evaluation/run_status.json').write_text(json.dumps(status))
    (tmp_path/'runtime_choices.json').write_text('{}')
    with pytest.raises(ValueError,match='human verification'):verify_evidence(tmp_path)
