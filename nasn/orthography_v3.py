"""Study v3 orthographic noise: realistic families, density and negation-cue targeting.

Candidates are deterministic and logged. Meaning preservation is a hypothesis that
requires human review; `typo` is a deliberately meaning-damaging control.
"""
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict

VERSION = 'orthography-v3'
SEED = 42
TOKEN = re.compile(r'[ء-غف-يً-ٰٟـ]+')
PROTECTED = re.compile(r'https?://\S+|www\.\S+|[@#]\S+')
MARKS = re.compile(r'[ً-ٰٟـ]')
# Unambiguous cues only. «ما» is excluded: relative/interrogative/exclamative uses dominate.
CUES = frozenset('لا لم لن ليس ليست لست لسنا ليسوا مش مو مب موب ماكو مافي مفيش بدون'.split())
# Letters that connect to the following letter, so tatweel after them is typographically valid.
JOINING = frozenset('بتثجحخسشصضطظعغفقكلمنهيئ')
LONG_VOWELS = frozenset('اوي')
REPEATABLE = LONG_VOWELS | frozenset('بتثجحخدذرزسشصضطظعغفقكلمنه')
FOLD = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا'}
FOLD_FINAL = {'ة': 'ه', 'ى': 'ي'}
# Physical neighbours on the standard Arabic (101) keyboard; used only for the typo control.
KEYBOARD = ['ضصثقفغعهخحجد', 'شسيبلاتنمكط', 'ئءؤرىةوزظ']
FAMILIES = {'tatweel', 'elongation', 'fold', 'typo'}
DENSITIES = {'single', 'dense'}
TARGETS = {'non_cue', 'cue'}
DENSE_FRACTION = .3


@dataclass(frozen=True)
class Condition:
    family: str
    density: str
    target: str
    repeat: int = 0

    def __post_init__(self):
        if self.family not in FAMILIES or self.density not in DENSITIES or self.target not in TARGETS:
            raise ValueError(f'Unknown condition: {self}')
        if (self.family == 'elongation') != (self.repeat > 0):
            raise ValueError('repeat is required for elongation only')
        if self.target == 'cue' and self.family not in {'tatweel', 'elongation'}:
            raise ValueError('Only presumed meaning-preserving families may target negation cues')

    @property
    def name(self):
        repeat = f'x{self.repeat}' if self.repeat else ''
        return f'{self.family}{repeat}/{self.density}/{self.target}'

    @classmethod
    def parse(cls, name):
        family, density, target = name.split('/')
        match = re.fullmatch(r'([a-z]+)(?:x(\d+))?', family)
        return cls(match.group(1), density, target, int(match.group(2) or 0))


def stem(word):
    """Strip one conjunction prefix (و/ف) and diacritics/tatweel for cue lookup."""
    bare = MARKS.sub('', word)
    return bare[1:] if bare[:1] in ('و', 'ف') and bare[1:] in CUES else bare


def is_cue(word):
    return stem(word) in CUES


def neighbours(char):
    for row in KEYBOARD:
        i = row.find(char)
        if i >= 0:
            return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    return []


def rewrite(word, family, repeat):
    """Return the perturbed word or None when this family cannot apply to it."""
    if MARKS.search(word):
        return None
    letters = list(word)
    if family == 'tatweel':
        # Insert after the last joining letter that is followed by another letter.
        spots = [i for i, c in enumerate(letters[:-1]) if c in JOINING]
        if not spots:
            return None
        i = spots[-1]
        return word[:i + 1] + 'ـ' + word[i + 1:]
    if family == 'elongation':
        # Emphatic repetition: last internal long vowel (كثيير), else the final letter (حلوو).
        vowels = [i for i, c in enumerate(letters) if c in LONG_VOWELS and i > 0]
        i = vowels[-1] if vowels else len(letters) - 1
        if letters[i] not in REPEATABLE:
            return None
        return word[:i + 1] + letters[i] * repeat + word[i + 1:]
    if family == 'fold':
        folded = ''.join(FOLD.get(c, c) for c in letters[:-1]) + FOLD_FINAL.get(letters[-1], FOLD.get(letters[-1], letters[-1]))
        return folded if folded != word else None
    if family == 'typo':
        # Substitute one interior letter; the first letter is kept so the word stays recognisable.
        spots = [i for i in range(1, len(letters)) if neighbours(letters[i])]
        if not spots:
            return None
        i = spots[len(spots) // 2]
        return word[:i] + neighbours(letters[i])[0] + word[i + 1:]
    raise ValueError(family)


def eligible(text, condition):
    """Candidate (start, end, word, replacement) tuples for one condition."""
    protected = [m.span() for m in PROTECTED.finditer(text)]
    result = []
    for token in TOKEN.finditer(text):
        a, b = token.span()
        word = token.group()
        if any(a < q and p < b for p, q in protected):
            continue
        if (a and (text[a - 1].isalnum() or text[a - 1] == '_')) or (b < len(text) and (text[b].isalnum() or text[b] == '_')):
            continue
        cue = is_cue(word)
        if cue != (condition.target == 'cue') or (not cue and len(word) < 2):
            continue
        new = rewrite(word, condition.family, condition.repeat)
        if new is not None:
            result.append((a, b, word, new))
    return result


def rank(row_id, condition, word_start):
    return hashlib.sha256(f'{VERSION}|{SEED}|{row_id}|{condition.family}|{condition.target}|{word_start}'.encode()).hexdigest()


def perturb(text, row_id, condition):
    """Apply a condition; returns (text, edits). Empty edits means not applicable.

    Selection depends on family/target, not density or repeat, so `single` edits are
    a subset of `dense` edits and repeat levels edit the same word (nested design).
    """
    if not isinstance(text, str) or not text:
        raise ValueError('Text must be a nonempty string')
    candidates = sorted(eligible(text, condition), key=lambda c: rank(row_id, condition, c[0]))
    if not candidates:
        return text, []
    if condition.target == 'cue' or condition.density == 'single':
        count = 1 if condition.density == 'single' else len(candidates)
    else:
        words = sum(1 for _ in TOKEN.finditer(text))
        count = min(len(candidates), max(1, math.ceil(DENSE_FRACTION * words)))
    chosen = sorted(candidates[:count])
    pieces, cursor, edits, shift = [], 0, [], 0
    for a, b, word, new in chosen:
        pieces += [text[cursor:a], new]
        edits.append({'start': a, 'end': b, 'target_start': a + shift, 'source': word, 'target': new})
        shift += len(new) - len(word)
        cursor = b
    changed = ''.join(pieces) + text[cursor:]
    if changed == text:
        raise ValueError('Perturbation unexpectedly unchanged')
    # Edits must never create or remove an unambiguous negation cue outside the target.
    before = sorted(stem(t.group()) for t in TOKEN.finditer(text) if is_cue(t.group()))
    after = sorted(stem(t.group()) for t in TOKEN.finditer(changed) if is_cue(t.group()))
    if condition.target == 'non_cue' and before != after:
        return text, []
    return changed, edits


def edits_json(edits):
    return json.dumps(edits, ensure_ascii=False)
