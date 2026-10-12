"""Verify exact file identity; no pretrained/classifier model is loaded."""
import hashlib

import pytest

from arsarcasm_audit.audit import DataValidationError
from nasn.verified_tokenizer import FILES,verify


def test_selected_tokenizer_hash_verification(tmp_path):
    hashes={}
    for name in FILES:
        data=('synthetic '+name).encode();(tmp_path/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    assert verify(tmp_path,hashes)==hashes
    (tmp_path/'tokenizer.json').write_text('tampered')
    with pytest.raises(DataValidationError,match='checksum mismatch'):verify(tmp_path,hashes)


def test_unregistered_config_cannot_override_selected_tokenizer(tmp_path):
    hashes={}
    for name in FILES:
        data=name.encode();(tmp_path/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(DataValidationError,match='override prohibited'):verify(tmp_path,hashes)
