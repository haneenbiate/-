"""Frozen v2: conservative run caps, marker preservation and context ablation."""
import json
import re
from pathlib import Path

from .normalization import apply_edits, negation_guard, protected_spans
from .orthography import TOKEN, PROTECTED, overlaps
from .preflight import require

VERSION='nasn-v2-controlled-1'
STRATEGIES=('RAW','TAT','REP2','NO_CONTEXT','NASN2')
SIMPLE=('TAT','REP2','NO_CONTEXT')
RULES=Path(__file__).resolve().parents[1]/'configs/nasn_v2.json'
REPEATS=re.compile(r'([بتثجحخسشصضطظعغفقكلمنه])\1{2,}')
DIACRITICS=re.compile(r'[\u064b-\u065f\u0670]')


def load_rules():
    rules=json.loads(RULES.read_text())
    require(rules['version']==VERSION and rules['strategies']==list(STRATEGIES), 'Unknown v2 protocol')
    require(rules['repeat_minimum']==3 and rules['repeat_target']==2 and
            rules['repeatable_characters']=='بتثجحخسشصضطظعغفقكلمنه' and
            rules['scope_following_arabic_tokens']==3 and rules['seed']==42,
            'Versioned repetition/context parameters changed')
    return rules


def explain(text,strategy='NASN2',rules=None):
    """No labels, predictions, row IDs, corpus statistics or human choices accepted."""
    require(isinstance(text,str) and strategy in STRATEGIES,'Invalid v2 text/strategy')
    rules=load_rules() if rules is None else rules
    if strategy=='RAW':return apply_edits(text,[],strategy,[]),[]
    marker_rules={**rules,'scope_following_arabic_tokens':0}
    markers=negation_guard(text,marker_rules)
    identifiers=[m.span() for m in PROTECTED.finditer(text)]
    critical=protected_spans(text,'B3',rules)
    context=negation_guard(text,rules) if strategy=='NASN2' else []
    edits=[];decisions=[]
    for token in TOKEN.finditer(text):
        start,end=token.span();word=token.group(); target=word;applied=[]
        attached=(start and (text[start-1].isalnum() or text[start-1]=='_')) or (end<len(text) and (text[end].isalnum() or text[end]=='_'))
        marker=any(overlaps(token.span(),span) for span in markers)
        identifier=any(overlaps(token.span(),span) for span in identifiers)
        sensitive=any(overlaps(token.span(),span) for span in critical)
        in_context=any(overlaps(token.span(),span) for span in context)
        if attached or marker or identifier:
            decisions.append({'start':start,'end':end,'source':word,'target':word,'decision':'preserve_marker_identifier_or_attached_token','marker':marker})
            continue
        if strategy in {'TAT','NO_CONTEXT','NASN2'} and 'ـ' in target:
            target=target.replace('ـ','');applied.append('tatweel_outside_marker')
        repetition_candidate=bool(REPEATS.search(target))
        repeat_reason='no_excessive_eligible_run'
        if strategy in {'REP2','NO_CONTEXT','NASN2'} and repetition_candidate:
            if DIACRITICS.search(word):repeat_reason='abstain_marked_word'
            elif sensitive:repeat_reason='abstain_critical_word_or_expression'
            elif in_context:repeat_reason='abstain_negation_context'
            else:
                target=REPEATS.sub(lambda m:m.group(1)*2,target)
                applied.append('repeat_run_ge3_cap2');repeat_reason='applied'
        decisions.append({'start':start,'end':end,'source':word,'target':target,'decision':'|'.join(applied) or 'unchanged',
            'repetition_candidate':repetition_candidate,'repetition_decision':repeat_reason,
            'in_negation_context':in_context,'sentiment_lexicon_protected':sensitive})
        if target!=word:edits.append((start,end,target,'|'.join(applied)))
    result=apply_edits(text,edits,strategy,markers+identifiers)
    # Verify literal marker bytes at shifted source offsets, not just lexical keys.
    for start,end in markers:
        shift=sum(len(e.target_span)-len(e.source_span) for e in result.edits if e.end<=start)
        require(result.text[start+shift:end+shift]==text[start:end],'Recognized negation marker bytes changed')
    return result,decisions


def normalize(text,strategy='NASN2',rules=None):
    return explain(text,strategy,rules)[0]
