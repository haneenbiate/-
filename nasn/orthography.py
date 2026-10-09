"""Provisional orthographic candidates, not automated semantic validation."""
import hashlib
import re
from dataclasses import dataclass, asdict

VERSION = 'orthography-v1'
SEED = 42
# Lexical heuristic only: misses dialect spellings and does not infer scope.
NEGATORS = frozenset('لا لم لن ليس ليست لست لسنا ليسوا ما مو مش مب مهو موش مافي مافيه ماني بدون غير'.split())
TOKEN = re.compile(r'[\u0621-\u063a\u0641-\u064a\u064b-\u065f\u0670\u0640]+')
PROTECTED = re.compile(r'https?://\S+|www\.\S+|[@#]\S+')
MARKS = re.compile(r'[\u064b-\u065f\u0670\u0640]')
# Only explicitly listed words. Never blanket map hamza, ya/alef maqsura, ta marbuta.
ALEF_VARIANTS = {'إلى':'الى', 'أيضا':'ايضا', 'أيضاً':'ايضاً', 'أحيانا':'احيانا', 'أحياناً':'احياناً'}
REPEATABLE = frozenset('بتثجحخسشصضطظعغفقكلمنه')
JOINING = REPEATABLE | frozenset('ي')
RULES = [('elongation',1), ('elongation',2), ('tatweel',1), ('tatweel',2), ('alef_allowlist',1)]


def lexical_negation(text):
    """Return original cue spans; tolerate marks/tatweel, common conjunctions and ma...sh."""
    spans = []
    for token in TOKEN.finditer(text):
        word = MARKS.sub('', token.group())
        forms = {word, word[1:] if word.startswith(('و','ف')) else word}
        forms |= {v[1:] for v in list(forms) if v.startswith(('و','ف'))}
        if forms & NEGATORS or any(v.startswith('ما') and v.endswith('ش') and len(v) > 3 for v in forms):
            spans.append((token.start(), token.end(), token.group()))
    return spans


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


@dataclass(frozen=True)
class Edit:
    rule: str
    intensity: int
    start: int
    end: int
    source_span: str
    target_span: str
    ambiguity: str

    def record(self):
        return asdict(self)


def candidates(text, rule, intensity):
    if (rule,intensity) not in RULES:
        raise ValueError('Unknown perturbation rule/intensity')
    tokens = list(TOKEN.finditer(text))
    negation = lexical_negation(text)
    protected = [(m.start(),m.end()) for m in PROTECTED.finditer(text)] + [(a,b) for a,b,_ in negation]
    # Overprotect possible attached dialect cues, including ma... without final sh.
    # This guard is broader than the reported lexical subgroup heuristic.
    for token in tokens:
        word=MARKS.sub('',token.group())
        forms=[word,word[1:] if word.startswith(('و','ف')) else word]
        if any(v.startswith(('ما','مو','مش','مب','ليس','لم','لن','لا','غير','بدون')) for v in forms):
            protected.append(token.span())
    # Protect following token too: multiword cues such as ما زال / ليس إلا.
    for a,b,_ in negation:
        following = next((m for m in tokens if m.start() >= b), None)
        if following is not None:
            protected.append((following.start(),following.end()))
    edits = []
    for token in tokens:
        a,b = token.span(); word=token.group()
        if any(overlaps((a,b),p) for p in protected):
            continue
        # Do not edit Arabic substrings attached to Latin/numeric word characters.
        if (a and (text[a-1].isalnum() or text[a-1]=='_')) or (b<len(text) and (text[b].isalnum() or text[b]=='_')):
            continue
        if rule == 'alef_allowlist':
            if word in ALEF_VARIANTS:
                edits.append(Edit(rule,intensity,a,b,word,ALEF_VARIANTS[word],
                    'Hamza omission may be ambiguous; word/context requires human review'))
        elif len(word)>=3 and not MARKS.search(word):
            for offset,char in enumerate(word):
                if rule == 'elongation' and char in REPEATABLE:
                    edits.append(Edit(rule,intensity,a+offset,a+offset+1,char,char*(intensity+1),
                        'Repetition may change emphasis or lexical interpretation'))
                if rule == 'tatweel' and char in JOINING and offset+1<len(word) and '\u0621' <= word[offset+1] <= '\u064a':
                    edits.append(Edit(rule,intensity,a+offset+1,a+offset+1,'','ـ'*intensity,
                        'Typography candidate; semantic equivalence still requires human review'))
    return edits


def perturb(text, row_id, rule, intensity, seed=SEED):
    """One logged edit, nested intensity for a fixed rule/row; stable across row order."""
    if not isinstance(text,str) or not text:
        raise ValueError('Text must be a nonempty string')
    edits = candidates(text,rule,intensity)
    if not edits:
        return text, None
    digest = hashlib.sha256(f'{VERSION}|{seed}|{row_id}|{rule}'.encode()).digest()
    edit = edits[int.from_bytes(digest[:8],'big') % len(edits)]
    changed = text[:edit.start] + edit.target_span + text[edit.end:]
    if changed == text:
        raise ValueError('Perturbation unexpectedly unchanged')
    # Original cue strings must remain at their offset-adjusted positions.
    delta = len(edit.target_span) - len(edit.source_span)
    for a,b,cue in lexical_negation(text):
        if overlaps((a,b),(edit.start,edit.end)) or a < edit.start < b:
            raise ValueError('Negation cue edited')
        shift = delta if a >= edit.end else 0
        if changed[a+shift:b+shift] != cue:
            raise ValueError('Negation cue altered')
    if [MARKS.sub('',s[2]) for s in lexical_negation(changed)] != [MARKS.sub('',s[2]) for s in lexical_negation(text)]:
        # Reject candidates that accidentally introduce a lexical negation cue.
        return text, None
    return changed, edit


def failure_reason(text,rule,intensity):
    """Descriptive applicability categories; never infer semantic validity."""
    tokens=list(TOKEN.finditer(text))
    if not tokens:return 'no_arabic_tokens'
    if candidates(text,rule,intensity):return 'negation_sequence_guard_rejected'
    if rule=='alef_allowlist':
        return 'allowlisted_word_protected' if any(t.group() in ALEF_VARIANTS for t in tokens) else 'allowlisted_alef_word_absent'
    eligible=False
    for token in tokens:
        word=token.group()
        if len(word)<3 or MARKS.search(word):continue
        if rule=='elongation' and any(c in REPEATABLE for c in word):eligible=True
        if rule=='tatweel' and any(c in JOINING and '\u0621'<=word[i+1]<='\u064a' for i,c in enumerate(word[:-1])):eligible=True
    return 'candidate_tokens_protected_or_attached' if eligible else 'no_unmarked_eligible_token'
