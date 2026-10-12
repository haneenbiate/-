"""Colab development-only v2 inference using the unchanged verified classifier."""
import argparse
import json
from pathlib import Path

from tools.build_task06_notebook import build as task06_build

RUN='nasn-prototype-v1-20261012T111339Z-4e08cdbf'


def build(reference):
    notebook=task06_build(reference)
    for cell in notebook['cells']:
        source=''.join(cell['source']).replace('/content/nasn-task06','/content/nasn-task07')
        source=source.replace("'nasn-prototype-v1-'","'nasn-v2-controlled-'")
        source=source.replace("Path('/content/drive/MyDrive/NASN/task06')/RUN_ID","Path('/content/drive/MyDrive/NASN/task07')/RUN_ID")
        cell['source']=source.splitlines(keepends=True)
    def replace(index,source):notebook['cells'][index]['source']=source.splitlines(keepends=True)
    replace(0,'''# Task 07 — verified v1 diagnosis and a frozen v2 comparison

This notebook is **unexecuted preparation**, not v2 performance evidence. Choose
GPU and run cells in order. It loads the SAME selected Task 03B epoch-2 classifier
from Drive, never retrains, changes selection, substitutes another model or reads
official test text. All new outputs go directly to a new durable Task 07 folder.

The original Task 06 result ZIP is independently verified and its tables recomputed.
RAW, TAT, REP2, NO_CONTEXT and NASN2 follow the already frozen Task 07 protocol.
NO_CONTEXT is the simple combined baseline/context ablation. Primary interpretation
requires paired superiority over every simple baseline and clean performance,
plus semantic/provenance evidence; failed performance gates recommend abandoning
the improvement claim or reframing rather than adding rules.
''')
    source=''.join(notebook['cells'][2]['source'])
    source+=f"\nTASK06_ZIP=Path('/content/drive/MyDrive/NASN/task06/{RUN}/{RUN}-reports.zip')\nassert TASK06_ZIP.is_file(), 'Provide the original completed Task 06 report ZIP path on Drive; never replace it with invented results'\n"
    replace(2,source)
    source=''.join(notebook['cells'][7]['source'])
    source=source.replace("'--output',ROOT/'preparation'","'--output',ROOT/'v1-preparation'")
    source+='''
call([PY,'-m','nasn.task07_io','--archive',TASK06_ZIP,'--output',ROOT/'task06-verified'],cwd=REPO,env=ENV)
call([PY,'-m','nasn.task07','diagnose','--v1-prepared',ROOT/'v1-preparation','--tokenizer-directory',TOKENIZER,'--task06-intake',ROOT/'task06-verified','--output',ROOT/'v1-diagnosis'],cwd=REPO,env=ENV)
call([PY,'-m','tools.export_task07_examples','--v1-prepared',ROOT/'v1-preparation','--intake',ROOT/'task06-verified','--diagnosis',ROOT/'v1-diagnosis','--output',ROOT/'extra-examples'],cwd=REPO,env=ENV)
call([PY,'-m','nasn.task07','prepare','--v1-prepared',ROOT/'v1-preparation','--tokenizer-directory',TOKENIZER,'--output',ROOT/'preparation'],cwd=REPO,env=ENV)
'''
    replace(7,source)
    replace(8,'''## V2 inference and controlled scientific decision

Rules and the decision gate are frozen in `configs/nasn_v2.json` and
`docs/task07_protocol.md` before new v2 effects are inspected. Recognized negation
markers remain byte-identical. Run caps and context guards are heuristics; new
meaning-sensitive output is provisional. The 394 source-eligible cases have
user-declared reviews with unverified independence; 9,557 remain unreviewed and
10 source exclusions receive no robustness metrics. New review packages are blank.

The same model/tokenizer/config hashes and loading checks apply. RAW clean
predictions must reproduce Task 03B. Per-rule metrics and pooled original-ID
clustered paired intervals are computed without fitting or tuning. No automatic
winner becomes a verified semantic-robustness claim. On OOM preserve the failure,
reduce only inference batch size, and choose a fresh output directory.
''')
    replace(9,'''EVALUATION=ROOT/'evaluation'
call([PY,'-m','nasn.task07','run','--prepared',ROOT/'preparation','--checkpoint',CHECKPOINT,'--batch-size',BATCH_SIZE,'--output',EVALUATION],cwd=REPO,env=ENV)
status=json.loads((EVALUATION/'run_status.json').read_text())
assert status['status']=='completed' and status['metrics_computed'] and status['checkpoint_reload_verified']
assert status['training_performed'] is False and status['test_text_read'] is False and status['test_evaluated'] is False
print(json.dumps(status,indent=2))
print((EVALUATION/'scientific_decision.json').read_text())
''')
    replace(10,'''## Durable reports and private review/examples

The report ZIP contains numeric model outputs, metrics, hashes, decision status,
runtime and text-free diagnostic tables. Return this ZIP for independent intake.
Do not publish weights or raw tweet/reviewer text to GitHub. A separate **private**
ZIP contains all v1 guard-difference examples and new blank human review packages;
retain it privately on Drive and send ONLY `new_human_reviewer_a.zip` and
`new_human_reviewer_b.zip` to the respective independent
reviewers. Review semantics/negation/emphasis rather than trusting normalization.
No annotation is automatically marked human-confirmed. Downloading reports does
not complete human validation or establish reviewer independence.
''')
    replace(11,'''VERIFY_SCRIPT="""import sys
from nasn.task05 import registry
for path in sys.argv[1:]:registry(path)
print('Durable Drive registries verified')
"""
call([PY,'-c',VERIFY_SCRIPT,ROOT/'preparation',ROOT/'preparation/review-package',EVALUATION],cwd=REPO,env=ENV)
import zipfile
REPORT_ZIP=ROOT/(RUN_ID+'-reports.zip')
names=['evaluation/'+p.name for p in EVALUATION.iterdir() if p.is_file()]
names+=['preparation/'+name for name in ['evidence.json','run_status.json','normalization_manifest.csv.gz','normalization_rules.json','character_token_audit.csv']]
names+=['v1-diagnosis/'+name for name in ['pipeline_equality.csv','all_B3_B4_differences.csv','all_elongation_token_exceptions.csv','guard_difference_summary.csv','annotation_quality_status.json','evidence.json']]
names+=['gpu_runtime.json','runtime_choices.json','dependency_check.txt','dependency_freeze.txt']
checks={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
with zipfile.ZipFile(REPORT_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(ROOT/name,arcname=name)
    z.writestr('report_checksums.json',json.dumps(checks,indent=2)+'\\n')
with zipfile.ZipFile(REPORT_ZIP) as z:
    for name,digest in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
PRIVATE_ZIP=ROOT/(RUN_ID+'-PRIVATE-examples-review.zip')
with zipfile.ZipFile(PRIVATE_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
    z.write(ROOT/'v1-diagnosis/all_B3_B4_differences_private.csv',arcname='all_B3_B4_differences_private.csv')
    z.write(ROOT/'extra-examples/all_elongation_exceptions_private.csv',arcname='all_elongation_exceptions_private.csv')
    z.write(ROOT/'preparation/new_human_review_queue.csv',arcname='new_human_review_queue.csv')
    for reviewer in ['a','b']:
        z.write(ROOT/f'preparation/new_human_reviewer_{reviewer}.zip',arcname=f'new_human_reviewer_{reviewer}.zip')
    for p in (ROOT/'preparation/review-package').iterdir():
        if p.is_file():z.write(p,arcname='review-package/'+p.name)
print('Durable lightweight report:',REPORT_ZIP)
print('Private examples/review archive:',PRIVATE_ZIP)
from google.colab import files
files.download(str(REPORT_ZIP))
''')
    notebook['metadata']['colab']['name']='task07_nasn_v2_colab.ipynb'
    for i,cell in enumerate(notebook['cells']):cell['id']=f'task07-{i:02d}'
    return notebook


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-ref',required=True)
    p.add_argument('--output',default='notebooks/task07_nasn_v2_colab.ipynb');args=p.parse_args()
    Path(args.output).write_text(json.dumps(build(args.code_ref),indent=1,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
