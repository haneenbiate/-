"""Reuse the verified CPU-only download setup for fixed adjudication assignments."""
import argparse
import copy
import json
import re
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-ref',required=True)
args=parser.parse_args()
if not re.fullmatch('[0-9a-f]{40}',args.code_ref):parser.error('Full immutable code reference required')
source=json.loads(Path('notebooks/task05_review_download.ipynb').read_text())
cells=[]
def md(value):cells.append({'cell_type':'markdown','metadata':{},'source':value.splitlines(keepends=True)})
def code(value):cells.append({'cell_type':'code','metadata':{},'source':value.splitlines(keepends=True),'outputs':[],'execution_count':None})
md('''# تنزيل حزمة التحكيم البشري — NASN Task 05

هذه حزمة 213 حالة تختلف فيها المراجعتان المصححتان. 150 تختلف في القرارات
الأساسية، و63 إضافية تختلف فقط في عبارة النفي أو وصف نطاقه. اختلاف الصياغة
لا يثبت اختلاف المعنى؛ يُراجع بشريًا ولا يُدمج تلقائيًا.

اختر CPU وشغّل الخلايا بالترتيب، واربط Drive ثم اضغط زر تنزيل حزمة التحكيم.
لا تحتاج GPU أو checkpoint. القرارات الجديدة فارغة؛ لا تُجرى مراجعة أو تحكيم
آلي، ولا يُستخدم LLM لحسم الخلاف. لا يتم تدريب نموذج أو قراءة الاختبار الرسمي.
لم يُنفَّذ هذا الدفتر في Colab هنا؛ تم اختبار التوليد محليًا.

أعطِ الحزمة لشخص ثالث مختلف عن R1 وR2. زوّده أيضًا بملفي A وB المصحّحين
ليقرأ تعليل المراجعين عند التحكيم، دون ملف LLM أو تنبؤات MARBERT. يُسجّل
قراره وتعليله وهوية المراجع ووقت المراجعة. يعود إليك ملف CSV المكتمل لجميع
الحالات الـ213. لا تتغيّر قرارات المراجعتين الأصليتين.
''')
for index in [1,2,3]:
    cell=copy.deepcopy(source['cells'][index])
    value=''.join(cell['source']).replace('8bc384eb8dab6b0266825b3cae3f1d8c7aa96ebb',args.code_ref)
    value=value.replace("RUN_ID='review-download-'","RUN_ID='adjudication-download-'")
    if index==2:
        value=value.replace("'Jinja2==3.1.6','MarkupSafe==3.0.3',","'Jinja2==3.1.6','MarkupSafe==3.0.3','openpyxl==3.1.5','et_xmlfile==2.0.0',")
    cell['source']=value.splitlines(keepends=True);cells.append(cell)
code('''downloads=REVIEW_ROOT/'adjudication-download'
review_call([REVIEW_PY,'-m','nasn.review_workbooks','export-adjudication',
    '--preparation',preparation,
    '--assignments',REVIEW_REPO/'reports/task05-reviewed/adjudication_assignment.csv',
    '--output',downloads],cwd=REVIEW_REPO,env=REVIEW_ENV)
status=json.loads((downloads/'run_status.json').read_text())
assert status['assigned_n']==213 and status['human_decisions_generated'] is False
assert hashlib.sha256((downloads/'adjudication.zip').read_bytes()).hexdigest()==status['zip_sha256']
print('حزمة التحكيم محفوظة على Drive:',downloads)
''')
md('''## تنزيل الحزمة

اضغط الزر. إذا لم يبدأ التنزيل، افتح Drive ونزّل `adjudication.zip` من المسار
المطبوع أعلاه. بعد فك الضغط افتح `adjudication.html`، وصدّر CSV باستمرار.
لا يصبح العمل مكتملًا حتى تُحسم الخلافات بشريًا وتتوفر تنبؤات Task 04 للتحليل.
''')
code('''from google.colab import files
import ipywidgets as widgets
from IPython.display import display
button=widgets.Button(description='تنزيل حزمة التحكيم',layout=widgets.Layout(width='240px'))
button.on_click(lambda _:files.download(str(downloads/'adjudication.zip')))
display(button)
''')
notebook={'cells':cells,'metadata':{'colab':{'name':'task05_adjudication_colab.ipynb'},
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'task05-adjudication-{i:02d}'
Path('notebooks/task05_adjudication_colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
