"""Recover only byte-identical selected tokenizer files; never load model weights."""
import argparse
import json
from pathlib import Path

from arsarcasm_audit.audit import sha256
from .preflight import require,library_versions
from .task05 import fresh,save

FILES={'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'}
REVISION='88e1fa192dd723cf0b3563500aec46209762eb22'


def verify(directory,checkpoint_hashes):
    directory=Path(directory)
    require(FILES<=set(checkpoint_hashes),'Selected checkpoint tokenizer registry incomplete')
    require({f.name for f in directory.iterdir() if f.is_file()}<=FILES|{'verification.json'},
            'Unexpected tokenizer files; model/config override prohibited')
    for name in FILES:
        require((directory/name).is_file() and sha256(directory/name)==checkpoint_hashes[name],
                f'Selected tokenizer checksum mismatch: {name}')
    versions=library_versions()
    require(versions['transformers']=='4.57.3' and versions['tokenizers']=='0.22.1',
            'Tokenizer runtime differs from baseline')
    return {name:sha256(directory/name) for name in sorted(FILES)}


def recover(source,lock_file,checkpoint_registry,output):
    source=Path(source);lock=json.loads(Path(lock_file).read_text())
    require(lock['repo_id']=='UBC-NLP/MARBERT' and lock['revision']==REVISION,'Official tokenizer revision differs')
    source_files=['config.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']
    for name in source_files:
        require(sha256(source/name)==lock['files'][name]['sha256'],f'Pinned official file changed: {name}')
    target=json.loads(Path(checkpoint_registry).read_text())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
    # Original nasn.marbert.encoded_dataset sets this backend serialization state.
    # A synthetic probe restores that state only; no classifier is instantiated.
    tokenizer('هذا نص للتحقق',max_length=128,truncation=True,padding='max_length')
    output=fresh(output);tokenizer.save_pretrained(output)
    hashes=verify(output,target)  # Fail on any difference, including tokenizer.json.
    result={'status':'all_selected_tokenizer_bytes_identical','model_revision':REVISION,
        'official_source_sha256':{n:sha256(source/n) for n in source_files},'tokenizer_sha256':hashes,
        'checkpoint_registry_sha256':sha256(checkpoint_registry),
        'serialization_state':'original baseline truncation=128 and max_length padding',
        'synthetic_probe_is_experimental_data':False,'model_weights_loaded':False,
        'selected_checkpoint_weight_bytes_verified_locally':False,'versions':library_versions(),
        'code_sha256':sha256(Path(__file__))}
    save(output/'verification.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','lock','checkpoint-registry','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    try:print(json.dumps(recover(args.source,args.lock,args.checkpoint_registry,args.output),indent=2))
    except Exception as exc:parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
