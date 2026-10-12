"""Verified import of the existing text-free Task 04 Colab report ZIP."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path,PurePosixPath

from arsarcasm_audit.audit import sha256
from .preflight import require
from .task05 import fresh,save

ROOT_FILES={'perturbation_manifest.csv','human_review_queue.csv','preparation.json','applicability.csv',
 'failure_cases.csv','baseline_metrics.json','baseline_per_class.csv','baseline_per_class.tex','baseline_confusion.csv',
 'baseline_subgroups.json','baseline_subgroups.csv','baseline_sensitivity.json','baseline_confidence_intervals.json',
 'baseline_error_analysis.md','error_patterns.csv','evidence.json','validation_confusion.pdf','validation_confusion.svg',
 'analysis_status.json','publication.json','artifact_hashes.json','gpu_runtime.json','dependency_check.txt',
 'dependency_freeze.txt','runtime_provenance.json','checkpoint_access.json','execution_command.json'}
INFERENCE_FILES={'run_status.json','evidence.json','methodology.md','robustness_metrics.csv','robustness_table.tex',
 'tokenization_associations.csv','perturbed_predictions.csv','perturbation_manifest.csv','clean_predictions.csv',
 'robustness_provisional.pdf','robustness_provisional.svg','robustness_provisional_population.pdf',
 'robustness_provisional_population.svg','robustness_confirmed.pdf','robustness_confirmed.svg'}


def import_bundle(archive,output):
    allowed=ROOT_FILES|{'inference/'+x for x in INFERENCE_FILES}|{'inference/'}
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        require(len(names)==len(set(names)) and set(names)<=allowed,'Unknown or duplicated ZIP members')
        for member in bundle.infolist():
            path=PurePosixPath(member.filename)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in member.filename,'Unsafe ZIP path')
            require(not stat.S_ISLNK(member.external_attr>>16),'ZIP symlinks prohibited')
        files=[i for i in bundle.infolist() if not i.is_dir()]
        require(sum(i.file_size for i in files)<=150*1024**2 and all(i.file_size<=75*1024**2 for i in files),'ZIP size limit exceeded')
        require('artifact_hashes.json' in names,'Original export registry missing')
        checks=json.loads(bundle.read('artifact_hashes.json'))
        require(set(checks)=={i.filename for i in files}-{'artifact_hashes.json'},'Incomplete export checksum coverage')
        data={i.filename:bundle.read(i.filename) for i in files}
        for name,digest in checks.items():
            require(hashlib.sha256(data[name]).hexdigest()==digest,f'Archive checksum mismatch: {name}')
        required={'inference/run_status.json','inference/evidence.json','inference/clean_predictions.csv',
                  'inference/perturbed_predictions.csv','inference/robustness_metrics.csv','runtime_provenance.json'}
        require(required<=set(data),'Required inference evidence missing')
        status=json.loads(data['inference/run_status.json'])
        runtime=json.loads(data['runtime_provenance.json'])
        require(status['status']=='completed' and status['training_performed'] is False and
                status['test_evaluated'] is False and status['test_text_read'] is False,'Task 04 inference incomplete or protocol violation')
        require(runtime['status']==status,'Runtime provenance differs from inference status')
    output=fresh(output)
    for name,content in data.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    # The notebook supplies one root registry with nested keys. Derive a flat
    # inference-only registry from those exact verified hashes; CSVs stay intact.
    inference_checks={name.removeprefix('inference/'):value for name,value in checks.items() if name.startswith('inference/')}
    save(output/'inference'/'artifact_hashes.json',inference_checks)
    provenance={'source_archive_name':Path(archive).name,'source_archive_sha256':sha256(archive),
        'original_root_registry_sha256':hashlib.sha256(data['artifact_hashes.json']).hexdigest(),
        'verified_original_files':len(data),'original_file_bytes_preserved':True,
        'derived_inference_registry_sha256':sha256(output/'inference'/'artifact_hashes.json'),
        'archive_contains_checkpoint_weights':False,'runtime':runtime,'code_sha256':sha256(Path(__file__))}
    save(output/'import_provenance.json',provenance)
    return provenance


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:print(json.dumps(import_bundle(args.archive,args.output),indent=2))
    except Exception as exc:parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
