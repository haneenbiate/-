"""Build a CPU-only Colab download workflow; no model or test text access."""
import json
from pathlib import Path

cells=[]
def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text):cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),'outputs':[],'execution_count':None})

md('''# تنزيل حزمتَي المراجعة البشرية — NASN Task 05

اختر جلسة CPU عادية وشغّل الخلايا بالترتيب. لا يحتاج هذا الدفتر GPU أو checkpoint.
يربط Google Drive، ويتحقق من مصدر التدريب وتقسيمات Task 02 الموجودة، ويعيد
تجهيز مرشحات Task 04 بصورة معلنة ليقارنها بالقائمة الأصلية الثابتة ذات 404 حالة؛
لا يعيد إنشاء تقسيمات البيانات. كل القرارات البشرية تبقى فارغة. لا يتم تشغيل
نموذج أو تدريب أو قراءة نصوص الاختبار الرسمي. لم ينفَّذ هذا الدفتر داخل Colab هنا.

في النهاية اضغط زر تنزيل كل حزمة. احفظ الملفات الخاصة على Drive، ولا تنشر نصوصها
أو هويات المراجعين على GitHub. تتضمن كل حزمة HTML عربي وCSV فارغًا وتعليمات عربية.
''')
code('''from pathlib import Path
import os, sys, uuid, datetime, subprocess, json, hashlib, urllib.request
from google.colab import drive
drive.mount('/content/drive')
RUN_ID='review-download-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
REVIEW_ROOT=Path('/content/drive/MyDrive/NASN/task05')/RUN_ID
REVIEW_ROOT.mkdir(parents=True,exist_ok=False)
REVIEW_REPO=Path('/content')/('nasn-'+RUN_ID)
REVIEW_CODE_REF='8bc384eb8dab6b0266825b3cae3f1d8c7aa96ebb'
def review_call(args, **kwargs):
    result=subprocess.run([str(x) for x in args],capture_output=True,text=True,**kwargs)
    if result.stdout:print(result.stdout,end='')
    if result.stderr:print(result.stderr,end='')
    if result.returncode:
        (REVIEW_ROOT/('failure-'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps({'command':[str(x) for x in args],'stdout':result.stdout,'stderr':result.stderr},indent=2))
        result.check_returncode()
    return result
review_call(['git','clone','--no-checkout','https://github.com/haneenbiate/-.git',REVIEW_REPO])
review_call(['git','-C',REVIEW_REPO,'checkout','--detach',REVIEW_CODE_REF])
assert subprocess.check_output(['git','-C',str(REVIEW_REPO),'rev-parse','HEAD'],text=True).strip()==REVIEW_CODE_REF
print('سيتم حفظ المراجعة في:',REVIEW_ROOT)
''')
code('''# Minimal Python environment: no torch, transformers, GPU or model download.
bootstrap=Path('/content')/('bootstrap-'+RUN_ID)
review_call([sys.executable,'-m','pip','install','--target',bootstrap,'uv==0.13.0'])
uv=bootstrap/'bin/uv'
bootenv={**os.environ,'UV_PYTHON_INSTALL_DIR':'/content/nasn-review-python','UV_CACHE_DIR':'/content/nasn-review-cache','UV_PYTHON_BIN_DIR':'/content/nasn-review-bin'}
review_call([uv,'python','install','--system-certs','3.12.14'],env=bootenv)
managed=subprocess.check_output([str(uv),'python','find','--managed-python','3.12.14'],env=bootenv,text=True).strip()
venv=Path('/content')/('venv-'+RUN_ID)
review_call([managed,'-m','venv',venv])
REVIEW_PY=venv/'bin/python'
assert subprocess.check_output([str(REVIEW_PY),'-c','import platform;print(platform.python_version())'],text=True).strip()=='3.12.14'
review_call([REVIEW_PY,'-m','pip','install','-r',REVIEW_REPO/'requirements.txt',
    'Jinja2==3.1.6','MarkupSafe==3.0.3',
    'matplotlib==3.10.8','contourpy==1.3.3','cycler==0.12.1','fonttools==4.61.1',
    'kiwisolver==1.4.9','pillow==12.3.0','pyparsing==3.3.2'])
check=review_call([REVIEW_PY,'-m','pip','check'])
(REVIEW_ROOT/'dependency_check.txt').write_text(check.stdout+check.stderr)
(REVIEW_ROOT/'dependency_freeze.txt').write_text(subprocess.check_output([str(REVIEW_PY),'-m','pip','freeze'],text=True))
REVIEW_ENV={**os.environ,'PYTHONPATH':str(REVIEW_REPO),'MPLBACKEND':'Agg','MPLCONFIGDIR':'/tmp/nasn-review-matplotlib','XDG_CACHE_HOME':'/tmp/nasn-review-cache'}
''')
code('''# Fetch only pinned official training bytes; never fetch official test text.
train=REVIEW_REPO/'data/raw/training_data.csv';train.parent.mkdir(parents=True,exist_ok=True)
url='https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/eb85513e133cfc67e30aca4968fadbe2594fcb42/ArSarcasm-v2/training_data.csv'
train.write_bytes(urllib.request.urlopen(url).read())
assert hashlib.sha256(train.read_bytes()).hexdigest()=='1da727eb763459f436c4e61f52c6a0bc63d5300af8fcc99b5675ae4616dae04a'
# Explicit candidate preparation only, using existing checksum-verified splits.
preparation=REVIEW_ROOT/'task04-preparation'
review_call([REVIEW_PY,'-m','nasn.task04','prepare','--output',preparation],cwd=REVIEW_REPO,env=REVIEW_ENV)
check_code="""import pandas as pd,sys
from pathlib import Path
root=Path(sys.argv[1]); repo=Path(sys.argv[2])
columns=['variant_id','source_row_id','original_sha256','perturbed_sha256','rule','intensity']
actual=pd.read_csv(root/'human_review_queue.csv',dtype=str,keep_default_na=False)
original=pd.read_csv(repo/'reports/task04/human_review_queue.csv',dtype=str,keep_default_na=False)
assert len(actual)==404 and actual[columns].equals(original[columns]), 'Original review queue changed'
print('Verified original 404 assignments; no human decisions generated')
"""
review_call([REVIEW_PY,'-c',check_code,preparation,REVIEW_REPO],cwd=REVIEW_REPO,env=REVIEW_ENV)
''')
code('''package=REVIEW_ROOT/'review-package'
review_call([REVIEW_PY,'-m','nasn.task05','prepare','--preparation',preparation,
    '--baseline',REVIEW_REPO/'reports/task03b/marbert-v1-20261009T213143Z-be8b8811',
    '--output',package],cwd=REVIEW_REPO,env=REVIEW_ENV)
downloads=REVIEW_ROOT/'downloads'
review_call([REVIEW_PY,'-m','tools.package_task05_review','--review-package',package,
    '--output',downloads],cwd=REVIEW_REPO,env=REVIEW_ENV)
checks=json.loads((downloads/'archive_sha256.json').read_text())
for name,digest in checks.items():assert hashlib.sha256((downloads/name).read_bytes()).hexdigest()==digest
print('الحزمتان محفوظتان على Drive:',downloads)
''')
md('''## تنزيل الملفات

اضغط زر المراجع الأول ثم زر المراجع الثاني. إذا لم يبدأ التنزيل، انتظر اكتمال
الأول ثم اضغط الثاني، أو نزّل الملفين مباشرة من مجلد `downloads` على Drive
بالمسار المطبوع أعلاه. بعد فك الضغط افتح HTML، وصدّر CSV باستمرار.
أعطِ كل شخص حزمته فقط، ولا تشارك الإجابات قبل انتهاء المراجعتين.
''')
code('''from google.colab import files
import ipywidgets as widgets
from IPython.display import display
button_a=widgets.Button(description='تنزيل المراجع الأول',layout=widgets.Layout(width='230px'))
button_b=widgets.Button(description='تنزيل المراجع الثاني',layout=widgets.Layout(width='230px'))
button_a.on_click(lambda _:files.download(str(downloads/'reviewer_a.zip')))
button_b.on_click(lambda _:files.download(str(downloads/'reviewer_b.zip')))
display(button_a,button_b)
''')
notebook={'cells':cells,'metadata':{'colab':{'name':'task05_review_download.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'task05-review-{i:02d}'
Path('notebooks/task05_review_download.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
