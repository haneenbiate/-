"""Study v3: multi-model orthographic-noise robustness audit for Arabic sentiment.

Commands: prepare, prevalence, sample-annotation, tokenizer-audit, train, infer, llm, analyze. Private outputs that
contain tweet text belong under runs/ (ignored by Git); public outputs contain IDs,
hashes and aggregate metrics only.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arsarcasm_audit.audit import COMMIT, load_csv, sha256
from .preflight import LABEL2ID, library_versions, load_partition, require
from .orthography_v3 import VERSION as ORTHOGRAPHY, Condition, TOKEN, edits_json, is_cue, perturb

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/study_v3.json'
SPLITS = ROOT / 'reports/task02'
RAW = {'validation': 'data/raw/training_data.csv', 'train': 'data/raw/training_data.csv', 'test': 'data/raw/testing_data.csv'}
SOURCE = {'validation': 'official_train', 'train': 'official_train', 'test': 'official_test'}
LABELS = list(LABEL2ID)
# Fixed gzip header time so identical tables have identical bytes and hashes.
GZIP = {'method': 'gzip', 'mtime': 0}
REPEAT_RUN = re.compile(r'([ء-ي])\1{2,}')


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def content_sha256(path):
    """Hash of the decompressed table: independent of gzip/zlib build and platform."""
    import gzip
    return hashlib.sha256(gzip.open(path).read()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def fresh(path):
    path = Path(path)
    require(not path.exists() or not any(path.iterdir()), f'Output exists; choose a new directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    require(config['version'] == 'study-v3-1' and config['orthography_version'] == ORTHOGRAPHY, 'Unknown study configuration')
    for name in config['conditions']:
        require(Condition.parse(name).name == name, f'Invalid condition: {name}')
    return config


def load_split(partition, config, raw_directory=None, dataset='arsarcasm'):
    """Registered rows joined to raw text; every test partition needs an explicit unlock."""
    require(partition in RAW, f'Unknown partition: {partition}')
    require(partition != 'test' or config['test_unlocked'] is True, 'Official test is locked: ' + config['test_unlock_rule'])
    if dataset == 'arsas':
        from . import arsas
        return arsas.load_partition(Path(raw_directory or ROOT / 'data/arsas') / arsas.MEMBER, partition)
    require(dataset == 'arsarcasm', f'Unknown dataset: {dataset}')
    meta = json.loads((SPLITS / 'metadata.json').read_text(encoding='utf-8'))
    require(meta['source_commit'] == COMMIT, 'Task 02 source revision changed')
    manifest_name = f'{partition}_manifest.csv'
    require(sha256(SPLITS / manifest_name) == meta['artifact_sha256'][manifest_name], f'{manifest_name} changed')
    raw_path = Path(raw_directory) / Path(RAW[partition]).name if raw_directory else ROOT / RAW[partition]
    require(sha256(raw_path) == meta['source_sha256'][SOURCE[partition]], f'Raw CSV hash mismatch: {raw_path}')
    manifest = pd.read_csv(SPLITS / manifest_name, dtype=str, keep_default_na=False)
    return load_partition(load_csv(raw_path), manifest, partition)


# ---------------------------------------------------------------- prepare

def defend(text, defense):
    """Inference-time normalization. STD is the common literature recipe; v2 strategies are frozen."""
    if defense == 'RAW':
        return text
    if defense == 'STD':
        return REPEAT_RUN.sub(r'\1', text.replace('ـ', ''))
    from .normalization_v2 import normalize
    return normalize(text, defense).text


def build_variants(frame, config):
    records = []
    for row in frame.itertuples(index=False):
        cue = any(is_cue(t.group()) for t in TOKEN.finditer(row.tweet))
        for name in config['conditions']:
            changed, edits = perturb(row.tweet, row.source_row_id, Condition.parse(name))
            records.append({
                'variant_id': 'v3_' + text_hash(f'{row.source_row_id}|{name}|{ORTHOGRAPHY}')[:24],
                'source_row_id': row.source_row_id, 'group_id': row.group_id, 'partition': row.partition,
                'gold_label': row.sentiment, 'dialect': row.dialect, 'sarcasm': row.sarcasm, 'has_cue': cue,
                'condition': name, 'applicable': bool(edits), 'n_edits': len(edits),
                'clean_sha256': text_hash(row.tweet), 'perturbed_sha256': text_hash(changed),
                'text': row.tweet, 'perturbed_text': changed, 'edits': edits_json(edits)})
    return pd.DataFrame(records)


def prepare(args, config):
    output = fresh(args.output)
    frame = load_split(args.partition, config, args.raw_directory, args.dataset)
    variants = build_variants(frame, config)
    variants.to_csv(output / 'variants.csv.gz', index=False, lineterminator='\n', compression=GZIP)
    public = variants.drop(columns=['text', 'perturbed_text', 'edits'])
    summary = public.groupby(['condition', 'has_cue']).agg(rows=('variant_id', 'size'), applicable=('applicable', 'sum'),
                                                         mean_edits=('n_edits', 'mean')).reset_index()
    summary.to_csv(output / 'applicability.csv', index=False, lineterminator='\n')
    save(output / 'evidence.json', {'study': config['version'], 'dataset': args.dataset, 'partition': args.partition, 'rows': len(frame),
         'variants': len(variants), 'applicable': int(variants.applicable.sum()),
         'config_sha256': sha256(args.config), 'orthography_sha256': sha256(ROOT / 'nasn/orthography_v3.py'),
         'variants_sha256': sha256(output / 'variants.csv.gz'),
         'variants_content_sha256': content_sha256(output / 'variants.csv.gz'), 'versions': library_versions()})
    print(summary.to_string(index=False))


NATURAL = {
    'tatweel': lambda word: 'ـ' in word,
    'letter_run_ge3': lambda word: bool(REPEAT_RUN.search(word.replace('ـ', ''))),
    'long_vowel_run_ge3': lambda word: bool(re.search(r'([اوي])\1{2,}', word.replace('ـ', ''))),
}


def prevalence(args, config):
    """How often naturally occurring noise appears in real tweets (train + validation only)."""
    output = fresh(args.output)
    rows = []
    for dataset in ['arsarcasm', 'arsas']:
        for partition in ['train', 'validation']:
            frame = load_split(partition, config, None, dataset)
            for row in frame.itertuples(index=False):
                words = [t.group() for t in TOKEN.finditer(row.tweet)]
                record = {'dataset': dataset, 'partition': partition, 'source_row_id': row.source_row_id}
                for name, test in NATURAL.items():
                    hits = [w for w in words if test(w)]
                    record[name] = bool(hits)
                    record[f'{name}_on_cue'] = any(is_cue(w.replace('ـ', '')) or is_cue(re.sub(r'(.)\1+', r'\1', w)) for w in hits)
                rows.append(record)
    detail = pd.DataFrame(rows)
    columns = [c for c in detail.columns if c not in {'dataset', 'partition', 'source_row_id'}]
    summary = detail.groupby('dataset')[columns].agg(['sum', 'mean'])
    summary.columns = [f'{a}_{"tweets" if b == "sum" else "rate"}' for a, b in summary.columns]
    summary.insert(0, 'tweets', detail.groupby('dataset').size())
    summary.reset_index().to_csv(output / 'natural_noise.csv', index=False, lineterminator='\n')
    save(output / 'evidence.json', {'partitions': ['train', 'validation'], 'test_text_read': False,
                                    'definitions': {'tatweel': 'U+0640 inside an Arabic word',
                                                    'letter_run_ge3': 'same Arabic letter 3+ times in a row',
                                                    'long_vowel_run_ge3': 'ا/و/ي 3+ times in a row',
                                                    '_on_cue': 'the noisy word is an unambiguous negation cue'}})
    print(summary.T.to_string())


def sample_annotation(args, config):
    """Stratified human-review sample: equal items per condition, one variant per tweet.

    Annotator files carry no model output, condition name or gold label. The typo
    condition is a built-in attention check (it is expected to damage meaning).
    """
    output = fresh(args.output)
    variants = read_variants(args.variants)
    pool = variants[variants.applicable].copy()
    pool['_rank'] = pool.variant_id.map(lambda v: text_hash(f"{config['seed']}|annotation|{v}"))
    chosen, used = [], set()
    for name in config['conditions']:
        part = pool[(pool.condition == name) & ~pool.source_row_id.isin(used)].sort_values('_rank').head(args.per_condition)
        require(len(part) == args.per_condition, f'Too few eligible items for {name}')
        used |= set(part.source_row_id)
        chosen.append(part)
    sample = pd.concat(chosen).sort_values('_rank').reset_index(drop=True)
    sample['item'] = [f'item{i + 1:04d}' for i in range(len(sample))]
    key = sample[['item', 'variant_id', 'source_row_id', 'condition', 'clean_sha256', 'perturbed_sha256']]
    key.to_csv(output / 'annotation_key_private.csv', index=False, lineterminator='\n', encoding='utf-8-sig')
    sheet = sample[['item', 'text', 'perturbed_text']].rename(columns={'text': 'original', 'perturbed_text': 'modified'})
    for column in ['meaning_preserved', 'sentiment_preserved', 'modified_sentiment', 'negation_present',
                   'negation_affected', 'notes']:
        sheet[column] = ''
    for annotator in ['A', 'B']:
        sheet.to_csv(output / f'annotation_sheet_{annotator}.csv', index=False, lineterminator='\n', encoding='utf-8-sig')
    save(output / 'evidence.json', {'variants_sha256': sha256(args.variants), 'items': len(sample),
         'per_condition': args.per_condition, 'key_sha256': sha256(output / 'annotation_key_private.csv')})
    print(key.condition.value_counts().to_string())


# ---------------------------------------------------------------- tokenizer audit

def tokenizer_audit(args, config):
    """CPU-only: how each tokenizer represents clean vs perturbed text and the edited words."""
    from transformers import AutoTokenizer
    output = fresh(args.output)
    variants = pd.read_csv(args.variants, keep_default_na=False)
    variants = variants[variants.applicable.astype(str).eq('True')]
    models = {**config['encoders'], **config['llms']}
    rows = []
    for key, spec in models.items():
        tokenizer = AutoTokenizer.from_pretrained(spec['model_name'], revision=spec['model_revision'],
                                                  cache_dir=args.cache_directory, trust_remote_code=False)
        unk = tokenizer.unk_token
        tokenize = lambda text: tokenizer.tokenize(text)
        for row in variants.itertuples(index=False):
            clean, noisy = tokenize(row.text), tokenize(row.perturbed_text)
            edits = json.loads(row.edits)
            source_tokens = sum(len(tokenize(e['source'])) for e in edits)
            target_tokens = [tokenize(e['target']) for e in edits]
            rows.append({'model': key, 'variant_id': row.variant_id, 'condition': row.condition,
                         'tokens_changed': clean != noisy, 'clean_tokens': len(clean), 'perturbed_tokens': len(noisy),
                         'clean_unk': clean.count(unk) if unk else 0, 'perturbed_unk': noisy.count(unk) if unk else 0,
                         'edited_word_unk': sum(unk in t for t in target_tokens) if unk else 0,
                         'edited_source_tokens': source_tokens, 'edited_target_tokens': sum(map(len, target_tokens)),
                         'n_edits': len(edits)})
        print(key, 'done', flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(output / 'tokenizer_detail.csv.gz', index=False, lineterminator='\n', compression=GZIP)
    summary = detail.assign(
        unk_word_rate=detail.edited_word_unk / detail.n_edits,
        fragmentation=detail.edited_target_tokens / detail.edited_source_tokens.clip(lower=1),
        new_unk=detail.perturbed_unk > detail.clean_unk).groupby(['model', 'condition']).agg(
        n=('variant_id', 'size'), tokens_changed=('tokens_changed', 'mean'), new_unk_rate=('new_unk', 'mean'),
        edited_word_unk_rate=('unk_word_rate', 'mean'), mean_fragmentation=('fragmentation', 'mean')).reset_index()
    summary.to_csv(output / 'tokenizer_summary.csv', index=False, lineterminator='\n')
    save(output / 'evidence.json', {'variants_sha256': sha256(args.variants), 'models': models, 'versions': library_versions()})
    print(summary.to_string(index=False))


# ---------------------------------------------------------------- training and inference (GPU)

def train(args, config):
    """Fine-tune one encoder with the frozen MARBERT v1 recipe; only model and seed vary."""
    from . import marbert
    spec = config['encoders'][args.model]
    template = json.loads((ROOT / config['training_template']).read_text(encoding='utf-8'))
    run = {**template, **spec, 'seed': args.seed, 'study': config['version'], 'encoder_key': args.model, 'dataset': args.dataset,
           'cache_directory': args.cache_directory}
    output = fresh(args.output)
    marbert.save_json(output / 'training_config.json', run)
    started = time.perf_counter()
    try:
        training, validation = (load_split(p, config, args.raw_directory, args.dataset) for p in ('train', 'validation'))
        status = marbert.fit(training, validation, run, output)
    except Exception:
        # Keep the full traceback next to the run so a remote failure can be diagnosed from Drive.
        import traceback
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8', newline='\n')
        marbert.save_json(output / 'run_status.json', {'status': 'failed', 'error_file': 'error.txt',
                                                       'versions': library_versions()})
        raise
    status['wall_seconds'] = time.perf_counter() - started
    marbert.save_json(output / 'run_status.json', status)
    print(json.dumps(status, indent=2))


def unique_inputs(variants, defenses):
    """Every distinct (defense, text) pair needed for clean and perturbed predictions."""
    texts = pd.concat([variants[['text']], variants[variants.applicable][['perturbed_text']].rename(columns={'perturbed_text': 'text'})])
    texts = texts.drop_duplicates('text')
    frames = [pd.DataFrame({'defense': d, 'raw_sha256': texts.text.map(text_hash),
                            'input_text': texts.text.map(lambda t: defend(t, d))}) for d in defenses]
    return pd.concat(frames, ignore_index=True)


def read_variants(path):
    variants = pd.read_csv(path, keep_default_na=False)
    variants['applicable'] = variants.applicable.astype(str).eq('True')
    return variants


def infer(args, config):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    output = fresh(args.output)
    inputs = unique_inputs(read_variants(args.variants), args.defenses or config['defenses'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True).to(device).eval()
    require(model.config.label2id == LABEL2ID, 'Checkpoint label mapping differs')
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(inputs), config['inference_batch_size']):
            batch = inputs.input_text.iloc[start:start + config['inference_batch_size']].tolist()
            encoded = tokenizer(batch, max_length=128, truncation=True, padding=True, return_tensors='pt').to(device)
            probabilities.append(model(**encoded).logits.float().softmax(-1).cpu().numpy())
    write_predictions(inputs, np.concatenate(probabilities), output, args, {'checkpoint': str(args.checkpoint),
        'checkpoint_hashes': {p.name: sha256(p) for p in sorted(Path(args.checkpoint).glob('*')) if p.is_file()}})


def write_predictions(inputs, probabilities, output, args, extra):
    require(probabilities.shape == (len(inputs), 3) and np.allclose(probabilities.sum(1), 1, atol=1e-4), 'Invalid probabilities')
    frame = inputs.drop(columns='input_text').assign(model=args.model, seed=args.seed)
    for i, label in enumerate(LABELS):
        frame[f'p_{label}'] = probabilities[:, i]
    frame.to_csv(output / 'predictions.csv.gz', index=False, lineterminator='\n', compression=GZIP)
    save(output / 'evidence.json', {'model': args.model, 'seed': args.seed, 'variants_sha256': sha256(args.variants),
         'predictions_sha256': sha256(output / 'predictions.csv.gz'), 'rows': len(frame), **extra,
         'versions': library_versions()})


PROMPT = ('Classify the sentiment of the following Arabic tweet. Answer with exactly one word: '
          'positive, negative, or neutral.\n\nTweet: {text}')


def llm(args, config):
    """Zero-shot LLM: softmax over the first-token logits of the three label words."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config['llms'][args.model]
    output = fresh(args.output)
    inputs = unique_inputs(read_variants(args.variants), args.defenses or ['RAW'])
    tokenizer = AutoTokenizer.from_pretrained(spec['model_name'], revision=spec['model_revision'], cache_dir=args.cache_directory)
    tokenizer.padding_side = 'left'
    label_ids = [tokenizer.encode(spec['labels'][label], add_special_tokens=False)[0] for label in LABELS]
    require(len(set(label_ids)) == 3, 'Label words must start with distinct tokens')
    model = AutoModelForCausalLM.from_pretrained(spec['model_name'], revision=spec['model_revision'], cache_dir=args.cache_directory,
                                                 torch_dtype=getattr(torch, spec['dtype']), device_map='auto').eval()
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT.format(text=t)}], tokenize=False,
                                             add_generation_prompt=True) for t in inputs.input_text]
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            encoded = tokenizer(prompts[start:start + args.batch_size], return_tensors='pt', padding=True).to(model.device)
            logits = model(**encoded).logits[:, -1, label_ids].float()
            probabilities.append(logits.softmax(-1).cpu().numpy())
            if start % (args.batch_size * 50) == 0:
                print(start, len(prompts), flush=True)
    write_predictions(inputs, np.concatenate(probabilities), output, args, {'llm': spec, 'prompt': PROMPT})


# ---------------------------------------------------------------- analysis

def macro_f1(gold, pred):
    """Vectorised three-class Macro-F1; arrays are (..., n)."""
    scores = []
    for label in range(3):
        tp = ((pred == label) & (gold == label)).sum(-1)
        fp = ((pred == label) & (gold != label)).sum(-1)
        fn = ((pred != label) & (gold == label)).sum(-1)
        denominator = 2 * tp + fp + fn
        scores.append(np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.))
    return np.mean(scores, axis=0)


def cluster_weights(groups, repetitions, seed):
    """(repetitions, n) row multiplicities of a cluster bootstrap over groups.

    Each resample draws len(groups) groups with replacement (duplicate tweets move
    together); a row's weight is how often its group was drawn. Equivalent to
    concatenating the drawn groups' rows, but vectorised.
    """
    codes, uniques = pd.factorize(groups)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength=len(uniques))
                       for _ in range(repetitions)])
    return counts[:, codes].astype(float)


def weighted_macro_f1(gold, pred, weights):
    """Three-class Macro-F1 for each row of a (r, n) weight matrix."""
    scores = []
    for label in range(3):
        tp = weights @ ((pred == label) & (gold == label))
        fp = weights @ ((pred == label) & (gold != label))
        fn = weights @ ((pred != label) & (gold == label))
        denominator = 2 * tp + fp + fn
        scores.append(np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1e-12), 0.))
    return np.mean(scores, axis=0)


def cell_statistics(gold, clean, noisy, groups, repetitions, seed):
    """gold (n,), clean/noisy (seeds, n). Metrics average over training seeds."""
    def stats(weights):
        flip = weights @ (clean != noisy).mean(0) / weights.sum(1)
        f1_clean = np.mean([weighted_macro_f1(gold, c, weights) for c in clean], axis=0)
        f1_noisy = np.mean([weighted_macro_f1(gold, p, weights) for p in noisy], axis=0)
        return np.stack([flip, f1_clean - f1_noisy, f1_clean, f1_noisy], axis=1)
    point = stats(np.ones((1, len(gold))))[0]
    draws = stats(cluster_weights(groups, repetitions, seed))
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    names = ['flip_rate', 'macro_f1_drop', 'clean_macro_f1', 'perturbed_macro_f1']
    return {**{n: float(v) for n, v in zip(names, point)}, **{f'{n}_low': float(v) for n, v in zip(names, low)},
            **{f'{n}_high': float(v) for n, v in zip(names, high)}}


def contrast_statistics(groups, flips_a, flips_b, repetitions, seed):
    """Paired difference in flip rate (a minus b); inputs are per-row seed-averaged rates (n,)."""
    point = float(flips_a.mean() - flips_b.mean())
    weights = cluster_weights(groups, repetitions, seed)
    draws = weights @ (flips_a - flips_b) / weights.sum(1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {'difference': point, 'low': float(low), 'high': float(high)}


def load_predictions(paths):
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    require(not frame.duplicated(['model', 'seed', 'defense', 'raw_sha256']).any(), 'Duplicate predictions')
    frame['pred'] = frame[[f'p_{l}' for l in LABELS]].to_numpy().argmax(1)
    return frame


def natural_noise(variants, predictions, config):
    """Unmodified tweets that already contain noise: RAW vs STD-cleaned predictions of the same text."""
    clean = variants.drop_duplicates('source_row_id')
    words = clean.text.map(lambda t: [m.group() for m in TOKEN.finditer(t)])
    cells = []
    for name, test in NATURAL.items():
        rows = clean[words.map(lambda ws: any(map(test, ws)))]
        for model, part in predictions[predictions.defense.isin(['RAW', 'STD'])].groupby('model'):
            seeds = sorted(set(part[part.defense == 'RAW'].seed) & set(part[part.defense == 'STD'].seed))
            cell = {'model': model, 'noise': name, 'n': len(rows), 'seeds': len(seeds)}
            if len(rows) >= config['minimum_cell_size'] and seeds:
                lookup = {(d, s): part[(part.defense == d) & (part.seed == s)].set_index('raw_sha256').pred for d in ['RAW', 'STD'] for s in seeds}
                raw = np.stack([lookup['RAW', s].reindex(rows.clean_sha256).to_numpy() for s in seeds]).astype(int)
                std = np.stack([lookup['STD', s].reindex(rows.clean_sha256).to_numpy() for s in seeds]).astype(int)
                # Orientation: "clean" = STD-cleaned text, "noisy" = the tweet as written.
                cell.update(cell_statistics(rows.gold_label.map(LABEL2ID).to_numpy(), std, raw, rows.group_id.to_numpy(),
                                            config['bootstrap_repetitions'], config['seed']))
            cells.append(cell)
    return cells


def analyze(args, config):
    output = fresh(args.output)
    variants = read_variants(args.variants)
    predictions = load_predictions(args.predictions)
    reps, seed, minimum = config['bootstrap_repetitions'], config['seed'], config['minimum_cell_size']
    cells, contrasts, flips = [], [], {}
    for (model, defense), part in predictions.groupby(['model', 'defense']):
        seeds = sorted(part.seed.unique())
        lookup = {s: part[part.seed == s].set_index('raw_sha256').pred for s in seeds}

        def preds(column, rows):
            return np.stack([lookup[s].reindex(rows[column]).to_numpy() for s in seeds])

        for name in config['conditions']:
            rows = variants[(variants.condition == name) & variants.applicable]
            if args.cue_only:
                rows = rows[rows.has_cue.astype(str).eq('True')]
            clean, noisy = preds('clean_sha256', rows), preds('perturbed_sha256', rows)
            require(not np.isnan(clean.astype(float)).any() and not np.isnan(noisy.astype(float)).any(),
                    f'Missing predictions: {model}/{defense}/{name}')
            clean, noisy = clean.astype(int), noisy.astype(int)
            # Per-tweet flip rate averaged over training seeds.
            flips[model, defense, name] = pd.Series((clean != noisy).mean(0), index=rows.source_row_id.to_numpy())
            cell = {'model': model, 'defense': defense, 'condition': name, 'n': len(rows), 'seeds': len(seeds)}
            if len(rows) >= minimum:
                cell.update(cell_statistics(rows.gold_label.map(LABEL2ID).to_numpy(), clean, noisy, rows.group_id.to_numpy(), reps, seed))
            cells.append(cell)
    groups = variants.drop_duplicates('source_row_id').set_index('source_row_id').group_id

    def paired(kind, first, second, **labels):
        """Flip-rate difference first minus second on tweets where both apply."""
        if first not in flips or second not in flips:
            return
        shared = flips[first].index.intersection(flips[second].index)
        row = {'kind': kind, **labels, 'n': len(shared)}
        if len(shared) >= minimum:
            row.update(contrast_statistics(groups.reindex(shared).to_numpy(), flips[first].loc[shared].to_numpy(),
                                           flips[second].loc[shared].to_numpy(), reps, seed))
        contrasts.append(row)

    models, defenses = sorted({k[0] for k in flips}), sorted({k[1] for k in flips})
    for model in models:
        for defense in defenses:
            for kind in ['cue_contrasts', 'density_contrasts']:
                for a, b in config[kind]:
                    paired(kind, (model, defense, a), (model, defense, b), model=model, defense=defense, a=a, b=b)
        for name in config['conditions']:
            for a, b in config['defense_contrasts']:
                paired('defense_contrasts', (model, a, name), (model, b, name), model=model, condition=name, a=a, b=b)
    reference = config['reference_model']
    for name in config['model_contrast_conditions']:
        for model in models:
            if model != reference:
                paired('model_contrasts', (model, 'RAW', name), (reference, 'RAW', name), condition=name, a=model, b=reference)
    pd.DataFrame(natural_noise(variants, predictions, config)).to_csv(output / 'natural_noise_cells.csv', index=False, lineterminator='\n')
    pd.DataFrame(cells).to_csv(output / 'robustness_cells.csv', index=False, lineterminator='\n')
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False, lineterminator='\n')
    save(output / 'evidence.json', {'variants_sha256': sha256(args.variants),
         'predictions_sha256': {str(p): sha256(p) for p in args.predictions}, 'bootstrap_repetitions': reps,
         'config_sha256': sha256(args.config), 'versions': library_versions()})
    print(pd.DataFrame(cells).to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--partition', default='validation'); p.add_argument('--raw-directory')
    p.add_argument('--dataset', choices=['arsarcasm', 'arsas'], default='arsarcasm')
    sub.add_parser('prevalence')
    p = sub.add_parser('sample-annotation'); p.add_argument('--variants', required=True)
    p.add_argument('--per-condition', type=int, default=30)
    p = sub.add_parser('tokenizer-audit'); p.add_argument('--variants', required=True)
    p = sub.add_parser('train'); p.add_argument('--model', required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--raw-directory'); p.add_argument('--dataset', choices=['arsarcasm', 'arsas'], default='arsarcasm')
    p = sub.add_parser('infer'); p.add_argument('--variants', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--defenses', nargs='*')
    p = sub.add_parser('llm'); p.add_argument('--variants', required=True); p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--batch-size', type=int, default=16); p.add_argument('--defenses', nargs='*')
    p = sub.add_parser('analyze'); p.add_argument('--variants', required=True); p.add_argument('--predictions', nargs='+', required=True)
    p.add_argument('--cue-only', action='store_true')
    for p in sub.choices.values():
        p.add_argument('--output', required=True); p.add_argument('--cache-directory', default='data/huggingface')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    {'prepare': prepare, 'prevalence': prevalence, 'sample-annotation': sample_annotation, 'tokenizer-audit': tokenizer_audit, 'train': train, 'infer': infer, 'llm': llm,
     'analyze': analyze}[args.command](args, config)


if __name__ == '__main__':
    main()
