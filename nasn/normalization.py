"""Fixed, label-blind Arabic normalization with exact reversible edit records.

B4's bounded lexical scope guard is a heuristic, not a syntactic scope parser.
No transformation is automatically certified as meaning preserving.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from .orthography import TOKEN, PROTECTED, MARKS, NEGATORS, lexical_negation, overlaps
from .preflight import require

VERSION = 'nasn-prototype-v1'
STRATEGIES = ('B0','B1','B2','B3','B4')
DEFAULT_RULES = Path(__file__).resolve().parents[1]/'configs/nasn_prototype_v1.json'
REPEATS = re.compile(r'([\u0621-\u063a\u0641-\u064a])\1{2,}')


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_rules(path=DEFAULT_RULES):
    rules=json.loads(Path(path).read_text())
    require(rules['version']==VERSION and rules['strategies']==list(STRATEGIES), 'Unknown normalization protocol')
    require(rules['seed']==42 and rules['unicode_normalization']=='NFC' and rules['repeat_minimum']==3
            and rules['repeat_target']==1 and rules['scope_following_arabic_tokens']==3,
            'Prototype parameters changed; create a separately versioned protocol')
    require(not any(rules[k] for k in ['alef_mapping','remove_diacritics','stemming','stopword_removal']),
            'Meaning-sensitive character deletion/mapping prohibited')
    return rules


def word_key(word):
    """Inspection key only; never used as output text or a semantic classifier."""
    return unicodedata.normalize('NFC', MARKS.sub('',word))


def inspection_forms(word):
    key=word_key(word)
    folded=REPEATS.sub(r'\1',key)
    forms={key,folded}
    forms |= {v[1:] for v in list(forms) if v.startswith(('و','ف'))}
    forms |= {v[1:] for v in list(forms) if v.startswith(('و','ف'))}
    return forms


def negation_guard(text, rules):
    """Cue/possible attached cue + next three Arabic tokens before punctuation.

    Broad prefixes deliberately overprotect possible dialect cues. These spans
    are lexical proxies; false positives/negatives and discontinuous scope remain.
    """
    tokens=list(TOKEN.finditer(text)); guards=[]
    cues={a for a,_,_ in lexical_negation(text)}
    prefixes=('ما','مو','مش','مب','ليس','لم','لن','لا','غير','بدون')
    for i,token in enumerate(tokens):
        forms=inspection_forms(token.group())
        if token.start() not in cues and not any(v in NEGATORS or v.startswith(prefixes)
               or (v.startswith('ما') and v.endswith('ش') and len(v)>3) for v in forms): continue
        end=token.end()
        for next_token in tokens[i+1:i+1+rules['scope_following_arabic_tokens']]:
            if any(c in text[end:next_token.start()] for c in rules['clause_boundaries']):break
            end=next_token.end()
        guards.append((token.start(),end))
    return guards


def protected_spans(text, strategy, rules):
    tokens=list(TOKEN.finditer(text))
    protected=[m.span() for m in PROTECTED.finditer(text)]
    # A shared exact lexical safety list is used in B3/B4; only B4 infers guards.
    words=set(rules['protected_sentiment_words'])|set(NEGATORS)
    for token in tokens:
        if inspection_forms(token.group())&words:protected.append(token.span())
    for phrase in rules['protected_expressions']:
        desired=phrase.split()
        for i in range(len(tokens)-len(desired)+1):
            section=tokens[i:i+len(desired)]
            if any(not text[a.end():b.start()].isspace() for a,b in zip(section,section[1:])):continue
            if all(target in inspection_forms(t.group()) for t,target in zip(section,desired)):
                protected.append((section[0].start(),section[-1].end()))
    if strategy=='B4':protected += negation_guard(text,rules)
    return protected


@dataclass(frozen=True)
class Edit:
    start:int
    end:int
    target_start:int
    target_end:int
    source_span:str
    target_span:str
    rule:str


@dataclass(frozen=True)
class Result:
    text:str
    original_sha256:str
    normalized_sha256:str
    strategy:str
    edits:tuple
    protected_spans:tuple
    semantic_status:str='provisional_normalization_not_human_validated'

    def record(self):
        return asdict(self)


def apply_edits(text, decisions, strategy, protected):
    decisions=sorted(decisions)
    parts=[]; edits=[]; cursor=0; delta=0
    for start,end,target,rule in decisions:
        require(cursor<=start<=end<=len(text),'Overlapping/invalid normalization edits')
        source=text[start:end]
        if source==target:continue
        parts.extend([text[cursor:start],target])
        edits.append(Edit(start,end,start+delta,start+delta+len(target),source,target,rule))
        cursor=end;delta+=len(target)-len(source)
    parts.append(text[cursor:]); result=''.join(parts)
    # Negator spellings can lose presentation tatweel/NFC only, never lexical cues.
    before=[word_key(cue) for _,_,cue in lexical_negation(text)]
    after=[word_key(cue) for _,_,cue in lexical_negation(result)]
    require(before==after,'Normalization changed the original lexical negation sequence')
    return Result(result,text_hash(text),text_hash(result),strategy,tuple(edits),tuple(protected))


def normalize(text, strategy='B4', rules=None):
    """Decisions depend only on text, strategy and fixed rules; no data metadata."""
    require(isinstance(text,str),'Normalization expects a string')
    require(strategy in STRATEGIES,'Unknown normalization strategy')
    rules=load_rules() if rules is None else rules
    if strategy=='B0':return apply_edits(text,[],strategy,[])
    if strategy in {'B1','B2'}:
        target=text.replace('ـ','')
        if strategy=='B2':target=re.sub(r'\s+',' ',unicodedata.normalize('NFC',target)).strip()
        return apply_edits(text,[(0,len(text),target,'tatweel_only' if strategy=='B1' else 'NFC_tatweel_whitespace')],strategy,[])
    protected=protected_spans(text,strategy,rules); decisions=[]
    for token in TOKEN.finditer(text):
        a,b=token.span()
        if any(overlaps((a,b),p) for p in protected):continue
        if (a and (text[a-1].isalnum() or text[a-1]=='_')) or (b<len(text) and (text[b].isalnum() or text[b]=='_')):continue
        word=token.group(); target=unicodedata.normalize('NFC',word.replace('ـ',''))
        # Do not remove vowel marks. Only explicitly allowlisted neutral outcomes.
        collapsed=REPEATS.sub(r'\1',target)
        if collapsed in rules['neutral_repeat_allowlist']:target=collapsed
        if target!=word:decisions.append((a,b,target,'selective_NFC_tatweel_allowlisted_repetition'))
    return apply_edits(text,decisions,strategy,protected)


def restore(normalized, result):
    """Exact inverse requires the original edit log, including deleted characters."""
    require(isinstance(result,Result) and text_hash(normalized)==result.normalized_sha256,
            'Cannot restore altered output or unverified edit record')
    restored=normalized
    for edit in reversed(result.edits):
        require(restored[edit.target_start:edit.target_end]==edit.target_span,'Reversal target span differs')
        restored=restored[:edit.target_start]+edit.source_span+restored[edit.target_end:]
    require(text_hash(restored)==result.original_sha256,'Reversal original checksum differs')
    return restored
