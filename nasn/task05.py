"""Blinded semantic review and analysis of saved validation predictions only."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from arsarcasm_audit.audit import sha256
from .preflight import LABEL2ID, require
from .task04 import (read, save, fresh, validation_inputs, build_manifest, review_sample,
                     verify_run, verify_external_reports, paired_bootstrap)

DECISIONS = ['semantic_validity', 'sentiment_preserved', 'negation_present',
             'negation_affects_sentiment']
VALUES = {'semantic_validity': ['preserved', 'changed', 'ambiguous'],
          **{x: ['yes', 'no', 'uncertain'] for x in DECISIONS[1:]}}
HUMAN = DECISIONS + ['linguistic_notes', 'reviewer_id', 'review_timestamp',
                     'negation_cue', 'original_negation_scope', 'perturbed_negation_scope']
IDENTITY = ['variant_id', 'source_row_id', 'original_sha256', 'perturbed_sha256']
VISIBLE = IDENTITY + ['original_text', 'perturbed_text', 'rule', 'intensity']
PROBS = ['probability_' + x for x in LABEL2ID]


def csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def registry(directory, name='artifact_hashes.json'):
    directory = Path(directory)
    hashes = read(directory / name)
    require(isinstance(hashes, dict) and hashes, 'Empty evidence registry')
    for file, digest in hashes.items():
        require(Path(file).name == file and sha256(directory / file) == digest,
                f'Evidence checksum mismatch: {file}')
    return hashes


def seal(directory):
    save(directory / 'artifact_hashes.json', {
        p.name: sha256(p) for p in sorted(directory.iterdir())
        if p.is_file() and p.name != 'artifact_hashes.json'})


def verify_preparation(preparation, baseline, config, checkpoint=None):
    """Verify original full-text queue, not silently recreate a missing artifact."""
    preparation = Path(preparation)
    hashes = registry(preparation)
    require({'preparation.json', 'perturbation_manifest.csv', 'human_review_queue.csv'} <= set(hashes),
            'Preparation registry lacks required evidence')
    validation, _, evidence = validation_inputs(config)
    verify_external_reports(baseline)
    predictions, checkpoint_hashes, status = verify_run(baseline, checkpoint, validation, evidence, config)
    recorded = read(preparation / 'preparation.json')
    for key in ['manifest_sha256', 'source_commit', 'source_training_sha256', 'task02_metadata_sha256']:
        require(recorded[key] == evidence[key], f'Task 04 preparation provenance mismatch: {key}')
    require(recorded['test_text_read'] is False and recorded['test_evaluated'] is False,
            'Task 04 touched official test data')
    manifest = csv(preparation / 'perturbation_manifest.csv')
    # Regeneration is a comparison only: never save or replace Task 04 inputs.
    expected = pd.read_csv(io.StringIO(build_manifest(validation).to_csv(index=False)),
                           dtype=str, keep_default_na=False)
    require(manifest.equals(expected), 'Task 04 original manifest differs from verified deterministic candidates')
    queue = csv(preparation / 'human_review_queue.csv')
    expected_queue = review_sample(build_manifest(validation))
    require(len(queue) == 404 and queue.variant_id.tolist() == expected_queue.variant_id.tolist(),
            'Existing 404-item review queue changed')
    joined = queue.set_index('variant_id').loc[:, manifest.columns.drop('variant_id')]
    require(joined.equals(manifest.set_index('variant_id').loc[queue.variant_id]),
            'Queue text or metadata differs from manifest')
    evidence.update(checkpoint_sha256=checkpoint_hashes,
                    checkpoint_bytes_verified=checkpoint is not None,
                    baseline_run_status=status, baseline_report_registry_sha256=sha256(Path(baseline)/'report_checksums.json'),
                    task04_input_sha256=hashes,
                    task05_code_sha256=sha256(Path(__file__)),
                    interface_code_sha256=sha256(Path(__file__).with_name('review_interface.html')))
    return queue, manifest, predictions, evidence


def review_template(queue):
    """No labels, predictions, confidence, sarcasm or heuristic negation hints."""
    result = queue[VISIBLE].copy()
    for column in HUMAN:
        result[column] = ''
    return result


def html_review(template, destination):
    shell = Path(__file__).with_name('review_interface.html').read_text()
    payload = json.dumps(template.to_dict('records'), ensure_ascii=False).replace('<', '\\u003c')
    Path(destination).write_text(shell.replace('__REVIEW_DATA__', payload), encoding='utf-8')


def prepare(preparation, baseline, output, config):
    queue, manifest, _, evidence = verify_preparation(preparation, baseline, config)
    output = fresh(output)
    template = review_template(queue)
    template.to_csv(output/'reviewer_a.csv', index=False, encoding='utf-8-sig')
    template.to_csv(output/'reviewer_b.csv', index=False, encoding='utf-8-sig')
    html_review(template, output/'reviewer_a.html')
    html_review(template, output/'reviewer_b.html')
    # Analysis-only metadata is deliberately separate from reviewer packages.
    queue.drop(columns=['original_text', 'perturbed_text', 'source_span', 'target_span',
                        'negation_spans', 'reviewer', 'rationale']).to_csv(output/'queue_registry.csv', index=False)
    coverage = queue.groupby(['rule', 'intensity', 'gold_label', 'dialect', 'sarcasm', 'has_negation']).size().rename('queued_n').reset_index()
    coverage['completed_reviews'] = 0
    coverage.to_csv(output/'review_coverage.csv', index=False)
    (output/'review_coverage.tex').write_text(coverage.to_latex(index=False), encoding='utf-8')
    stage = pd.DataFrame([{'queue_n':len(queue),'unique_original_ids':queue.source_row_id.nunique(),
                   'reviewer_a_completed':0,'reviewer_b_completed':0,
                   'confirmed_n':0,'agreement_status':'not_available_no_human_reviews',
                   'robustness_status':'not_available_review_and_inference_pending'}])
    stage.to_csv(output/'review_stage.csv',index=False)
    (output/'review_stage.tex').write_text(stage.to_latex(index=False),encoding='utf-8')
    status = {'status': 'prepared_review', 'queue_n': len(queue), 'candidate_n': len(manifest),
              'applicable_n': int(manifest.applicable.eq('True').sum()), 'reviewer_a_completed': 0,
              'reviewer_b_completed': 0, 'confirmed_n': 0, 'agreement': None,
              'robustness_metrics': None, 'tatweel_comparison': None,
              'human_review_completed': False, 'training_performed': False, 'test_evaluated': False,
              'remaining': ['Two independent human reviews and adjudication',
                            'Completed Task 04 inference evidence and paired predictions',
                            'Verified tokenizer/checkpoint access for intensity-1 versus intensity-2 token comparison']}
    evidence['prepared_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(output/'evidence.json', evidence)
    save(output/'run_status.json', status)
    (output/'methodology.md').write_text(methodology(status), encoding='utf-8')
    seal(output)
    return status


def validate_decisions(path, template):
    frame = csv(path)
    require(set(frame.columns) == set(template.columns), 'Review CSV columns changed or model information added')
    require(not frame.variant_id.duplicated().any() and set(frame.variant_id) == set(template.variant_id),
            'Review CSV must retain all assigned IDs exactly once')
    frame = frame.set_index('variant_id').loc[template.variant_id].reset_index()
    for column in VISIBLE:
        require(frame[column].tolist() == template[column].astype(str).tolist(), f'Review identity/text altered: {column}')
    started = frame[HUMAN].ne('').any(axis=1)
    for row in frame[started].to_dict('records'):
        for field in DECISIONS:
            require(row[field] in VALUES[field], f'Incomplete/invalid decision {field}: {row["variant_id"]}')
        require(row['reviewer_id'].strip() and row['linguistic_notes'].strip(), 'Reviewer identity and rationale required')
        try:
            timestamp = dt.datetime.fromisoformat(row['review_timestamp'].replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Invalid review_timestamp; ISO 8601 with timezone required') from exc
        require(timestamp.tzinfo is not None, 'Review timestamp must include timezone')
        require(not (row['negation_present'] == 'no' and row['negation_affects_sentiment'] == 'yes'),
                'Contradictory negation decisions')
        require(not (row['semantic_validity'] == 'preserved' and row['sentiment_preserved'] == 'no'),
                'Preserved meaning cannot have changed sentiment')
        if row['negation_present'] == 'yes':
            require(row['negation_cue'].strip() and row['original_negation_scope'].strip() and
                    row['perturbed_negation_scope'].strip(), 'Present negation requires explicit cue and scope notes')
    complete = frame[started].copy()
    require(complete.reviewer_id.nunique() <= 1, 'One independent reviewer ID per review file required')
    return complete


def agreement(a, b, queue_n):
    pair = a.merge(b, on='variant_id', suffixes=('_a', '_b'), validate='one_to_one')
    require(not set(a.reviewer_id) & set(b.reviewer_id), 'Independent reviews must have distinct reviewer IDs')
    reports = []
    for field in DECISIONS:
        x, y = pair[field+'_a'], pair[field+'_b']
        exact = float((x == y).mean()) if len(pair) else None
        kappa = None
        if len(pair):
            # Constant identical annotations make chance-adjusted agreement undefined.
            px, py = x.value_counts(normalize=True), y.value_counts(normalize=True)
            chance = sum(px.get(v, 0)*py.get(v, 0) for v in VALUES[field])
            if chance < 1-1e-15:
                kappa = float(cohen_kappa_score(x, y, labels=VALUES[field]))
        reports.append({'field': field, 'paired_n': len(pair), 'exact_agreement': exact,
                        'cohen_kappa': kappa, 'kappa_status': 'computed' if kappa is not None else 'undefined_or_no_pairs'})
    coverage = {'queue_n': queue_n, 'reviewer_a_completed': len(a), 'reviewer_b_completed': len(b),
                'paired_completed': len(pair), 'paired_coverage': len(pair)/queue_n}
    return pair, reports, coverage


def resolve_reviews(template, a, b, adjudication=None):
    pair, reports, coverage = agreement(a, b, len(template))
    disagreement = pd.Series(False, index=pair.index)
    for field in DECISIONS:
        disagreement |= pair[field+'_a'] != pair[field+'_b']
    # Scope wording is free text; both independent interpretations must be checked.
    for field in ['negation_cue', 'original_negation_scope', 'perturbed_negation_scope']:
        disagreement |= pair[field+'_a'] != pair[field+'_b']
    conflicts = set(pair.loc[disagreement, 'variant_id'])
    conflict_queue = template[template.variant_id.isin(conflicts)].copy()
    decisions = []
    adjudicated = None
    if adjudication is not None:
        adjudicated = validate_decisions(adjudication, conflict_queue)
        require(not set(adjudicated.reviewer_id) & (set(a.reviewer_id)|set(b.reviewer_id)),
                'Adjudicator must be independent of both reviewers')
    adjud = {} if adjudicated is None else adjudicated.set_index('variant_id').to_dict('index')
    for row in pair.to_dict('records'):
        variant = row['variant_id']
        if variant in conflicts:
            if variant not in adjud:
                continue
            decision = {k: adjud[variant][k] for k in HUMAN}
            basis = 'explicit_third_reviewer_adjudication'
        else:
            decision = {k: row[k+'_a'] for k in HUMAN}
            decision['reviewer_id'] = row['reviewer_id_a']+'|'+row['reviewer_id_b']
            decision['linguistic_notes'] = row['linguistic_notes_a']+'\n'+row['linguistic_notes_b']
            basis = 'two_explicit_independent_human_decisions'
        confirmed = decision['semantic_validity'] == 'preserved' and decision['sentiment_preserved'] == 'yes'
        # Uncertain scope decisions exclude a case from the trustworthy confirmed subset.
        confirmed &= decision['negation_present'] != 'uncertain' and decision['negation_affects_sentiment'] != 'uncertain'
        decisions.append({'variant_id': variant, **decision, 'decision_basis': basis, 'eligible_confirmed': bool(confirmed)})
    final = pd.DataFrame(decisions, columns=['variant_id']+HUMAN+['decision_basis', 'eligible_confirmed'])
    coverage.update(disagreements=len(conflicts), resolved_n=len(final), confirmed_n=int(final.eligible_confirmed.sum()),
                    unresolved_disagreements=len(conflicts-set(adjud)), human_review_completed=(len(pair)==len(template) and conflicts<=set(adjud)))
    return final, conflict_queue, reports, coverage


def import_reviews(review_package, reviewer_a, reviewer_b, adjudication, output):
    package = Path(review_package)
    require({'reviewer_a.csv','reviewer_b.csv','queue_registry.csv'} <= set(registry(package)),
            'Review package registry lacks required assignments')
    template = csv(package/'reviewer_a.csv')
    require(template[HUMAN].eq('').all().all(), 'Prepared template already contains decisions')
    a, b = validate_decisions(reviewer_a, template), validate_decisions(reviewer_b, template)
    final, conflicts, reports, coverage = resolve_reviews(template, a, b, adjudication)
    output = fresh(output)
    a.to_csv(output/'independent_reviewer_a.csv', index=False, encoding='utf-8-sig')
    b.to_csv(output/'independent_reviewer_b.csv', index=False, encoding='utf-8-sig')
    final.to_csv(output/'resolved_decisions.csv', index=False, encoding='utf-8-sig')
    if adjudication:
        csv(adjudication).to_csv(output/'adjudication_decisions.csv', index=False, encoding='utf-8-sig')
    conflicts.to_csv(output/'adjudication_queue.csv', index=False, encoding='utf-8-sig')
    html_review(conflicts, output/'adjudication.html')
    pd.DataFrame(reports).to_csv(output/'agreement.csv', index=False)
    strata = csv(package/'queue_registry.csv')
    strata['reviewer_a_completed'] = strata.variant_id.isin(a.variant_id).astype(int)
    strata['reviewer_b_completed'] = strata.variant_id.isin(b.variant_id).astype(int)
    strata['resolved_n'] = strata.variant_id.isin(final.variant_id).astype(int)
    strata['confirmed_n'] = strata.variant_id.isin(final.loc[final.eligible_confirmed.eq(True),'variant_id']).astype(int)
    strata.groupby(['rule','intensity','gold_label','dialect','sarcasm','has_negation']).agg(
        queued_n=('variant_id','size'), reviewer_a_completed=('reviewer_a_completed','sum'),
        reviewer_b_completed=('reviewer_b_completed','sum'), resolved_n=('resolved_n','sum'),
        confirmed_n=('confirmed_n','sum')).reset_index().to_csv(output/'coverage_by_stratum.csv',index=False)
    save(output/'review_coverage.json', coverage)
    save(output/'audit.json', {'review_package_registry_sha256': sha256(package/'artifact_hashes.json'),
                             'reviewer_a_sha256': sha256(reviewer_a), 'reviewer_b_sha256': sha256(reviewer_b),
                             'adjudication_sha256': sha256(adjudication) if adjudication else None,
                             'human_decisions_generated': False, 'training_performed': False, 'test_evaluated': False})
    status = {'status': 'human_review_complete' if coverage['human_review_completed'] else 'prepared_review_incomplete', **coverage}
    save(output/'run_status.json', status)
    seal(output)
    return status


def validate_saved_predictions(inference, manifest, baseline_predictions, evidence):
    inference = Path(inference)
    hashes = registry(inference)
    require({'run_status.json', 'evidence.json', 'clean_predictions.csv', 'perturbed_predictions.csv'} <= set(hashes),
            'Inference evidence registry incomplete')
    status, recorded = read(inference/'run_status.json'), read(inference/'evidence.json')
    require(status['status'] == 'completed' and status['training_performed'] is False and
            status['test_evaluated'] is False and status['test_text_read'] is False,
            'Task 04 inference incomplete or protocol violation')
    for key in ['manifest_sha256', 'checkpoint_sha256', 'source_commit', 'source_training_sha256']:
        require(recorded[key] == evidence[key], f'Task 04 inference evidence differs: {key}')
    clean, changed = csv(inference/'clean_predictions.csv'), csv(inference/'perturbed_predictions.csv')
    require(clean.source_row_id.tolist() == baseline_predictions.source_row_id.tolist(), 'Clean IDs/order differ')
    require(not changed.variant_id.duplicated().any(), 'Duplicated perturbed predictions')
    require(set(changed.variant_id) == set(manifest.variant_id),
            'Task 05 requires predictions for all original provisional variants; reviewed Task 04 filtering is not allowed')
    changed = changed.set_index('variant_id').loc[manifest.variant_id].reset_index()
    for key in ['source_row_id', 'original_sha256', 'perturbed_sha256', 'gold_id', 'rule', 'intensity']:
        require(changed[key].tolist() == manifest[key].tolist(), f'Prediction identity mismatch: {key}')
    for frame in [clean, changed]:
        probs = frame[PROBS].astype(float).to_numpy()
        require(np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all() and
                np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0), 'Invalid saved probability vectors')
        require(np.array_equal(probs.argmax(axis=1), frame.predicted_id.astype(int)), 'Prediction not probability argmax')
        if 'predicted_label' in frame:
            require(frame.predicted_label.tolist() == frame.predicted_id.astype(int).map({v:k for k,v in LABEL2ID.items()}).tolist(),
                    'Saved predicted label mapping differs')
        if 'confidence' in frame:
            require(np.allclose(probs.max(axis=1), frame.confidence.astype(float),atol=1e-8,rtol=0),
                    'Saved confidence differs from probabilities')
        require(np.isin(frame.gold_id.astype(int), [0,1,2]).all(), 'Invalid saved gold IDs')
    require(np.array_equal(clean.gold_id.astype(int), baseline_predictions.gold_id) and
            np.array_equal(clean.predicted_id.astype(int), baseline_predictions.predicted_id) and
            np.allclose(clean[PROBS].astype(float), baseline_predictions[PROBS], atol=1e-5, rtol=1e-5),
            'Clean results differ from verified selected baseline')
    clean_index = clean.set_index('source_row_id')
    for row in changed[['source_row_id','clean_id']].itertuples(index=False):
        require(int(row.clean_id) == int(clean_index.loc[row.source_row_id, 'predicted_id']),
                'Saved clean/perturbed pairing changed')
    for field in ['gold_id', 'predicted_id', 'clean_id', 'intensity']:
        changed[field] = changed[field].astype(int)
    changed['high_risk_near_duplicate'] = manifest.high_risk_near_duplicate.eq('True').to_numpy()
    changed['lexical_negation'] = manifest.has_negation.eq('True').to_numpy()
    changed['applicable'] = manifest.applicable.eq('True').to_numpy()
    return clean, changed, hashes


def confirmed_tables(changed, decisions, repetitions=2000, minimum=30):
    require(minimum >= 30, 'Minimum stratum size cannot be below 30')
    human = decisions[decisions.eligible_confirmed.astype(str).eq('True')]
    frame = changed.merge(human[['variant_id','negation_present','negation_affects_sentiment']],
                          on='variant_id', validate='one_to_one')
    require(frame.applicable.all(), 'No-op variants cannot be human-confirmed perturbations')
    reports = []
    for subset, eligible in [('all_reviewed', frame), ('exclude_high_risk', frame[~frame.high_risk_near_duplicate])]:
        for (rule, intensity), group in eligible.groupby(['rule','intensity']):
            strata = [('all', 'all', group)]
            for field in ['negation_present','negation_affects_sentiment','lexical_negation']:
                strata += [(field, str(value), part) for value, part in group.groupby(field)]
            for field, value, part in strata:
                item = {'semantic_status':'human_confirmed', 'subset':subset, 'rule':rule, 'intensity':int(intensity),
                        'subgroup':field, 'value':value, 'n':len(part), 'gold_classes':int(part.gold_id.nunique()),
                        'seed':42, 'bootstrap_repetitions':repetitions}
                if len(part) < minimum:
                    reports.append({**item, 'status':'insufficient_sample_size'})
                    continue
                for metric, values in paired_bootstrap(part, repetitions).items():
                    reports.append({**item, 'status':'computed', 'metric':metric, **values})
    return pd.DataFrame(reports)


def tatweel_comparison(changed, clean, manifest, tokenizer=None, tolerance=1e-6):
    """Same original IDs at both intensities; no token equality inferred from counts."""
    selected = changed[changed.rule.eq('tatweel') & changed.applicable]
    one, two = (selected[selected.intensity.eq(i)].set_index('source_row_id') for i in [1,2])
    require(set(one.index) == set(two.index), 'Tatweel intensity pairing incomplete')
    two = two.loc[one.index]
    clean = clean.set_index('source_row_id').loc[one.index]
    text = manifest.set_index('variant_id')
    records = []
    for index in one.index:
        a, b, c = one.loc[index], two.loc[index], clean.loc[index]
        pc, pa, pb = (r[PROBS].astype(float).to_numpy() for r in [c,a,b])
        ids, full_ids = None, None
        if tokenizer is not None:
            strings = [text.loc[a.variant_id,'original_text'], text.loc[a.variant_id,'perturbed_text'],
                       text.loc[b.variant_id,'perturbed_text']]
            ids = [tokenizer(t, add_special_tokens=True, truncation=True, max_length=128)['input_ids'] for t in strings]
            full_ids = [tokenizer(t, add_special_tokens=True, truncation=False)['input_ids'] for t in strings]
        for comparison, p, q, left, right in [('clean_vs_1',pc,pa,0,1), ('clean_vs_2',pc,pb,0,2), ('1_vs_2',pa,pb,1,2)]:
            final = [int(c.predicted_id), int(a.predicted_id), int(b.predicted_id)]
            record = {'source_row_id':index, 'comparison':comparison, 'probability_vector_changed_exact':not np.array_equal(p,q),
                      'probability_vector_changed_above_1e_6':bool(np.max(np.abs(p-q))>tolerance),
                      'max_absolute_probability_difference':float(np.max(np.abs(p-q))),
                      'prediction_changed':final[left]!=final[right],
                      'effective_token_sequence_changed':ids[left]!=ids[right] if ids is not None else '',
                      'full_token_sequence_changed':full_ids[left]!=full_ids[right] if full_ids is not None else '',
                      'token_status':'verified_tokenizer' if ids is not None else 'unavailable_verified_tokenizer'}
            if ids is not None:
                record['left_token_ids'] = json.dumps(ids[left])
                record['right_token_ids'] = json.dumps(ids[right])
                record['left_full_token_ids'] = json.dumps(full_ids[left])
                record['right_full_token_ids'] = json.dumps(full_ids[right])
            for k, label in enumerate(LABEL2ID):
                record[f'left_probability_{label}'] = float(p[k])
                record[f'right_probability_{label}'] = float(q[k])
                record[f'probability_delta_{label}'] = float(q[k]-p[k])
            records.append(record)
    details = pd.DataFrame(records)
    summary = []
    for comparison, group in details.groupby('comparison'):
        summary.append({'comparison':comparison,'n_original_ids':len(group),
                        'changed_probability_vectors_exact':int(group.probability_vector_changed_exact.sum()),
                        'changed_probability_vectors_above_1e_6':int(group.probability_vector_changed_above_1e_6.sum()),
                        'changed_final_predictions':int(group.prediction_changed.sum()),
                        'changed_effective_token_sequences':int(group.effective_token_sequence_changed.sum()) if tokenizer else None,
                        'changed_full_token_sequences':int(group.full_token_sequence_changed.sum()) if tokenizer else None,
                        'token_status':'verified_tokenizer' if tokenizer else 'unavailable_verified_tokenizer',
                        'interpretation':'provisional candidate instability; descriptive correlation, not causal evidence'})
    return details, pd.DataFrame(summary)


def verify_exported_token_flags(changed,manifest,tokenizer):
    """Recompute every recorded token flag using the byte-verified tokenizer."""
    lookup=manifest.set_index('variant_id');cache={}
    def tokens(text):
        if text not in cache:
            full=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
            effective=tokenizer(text,add_special_tokens=True,truncation=True,max_length=128)['input_ids']
            cache[text]=(full,effective)
        return cache[text]
    for row in changed.itertuples(index=False):
        record=lookup.loc[row.variant_id]
        original,effective=tokens(record.original_text)
        perturbed,changed_effective=tokens(record.perturbed_text)
        expected={'token_ids_changed':original!=perturbed,'effective_token_ids_changed':effective!=changed_effective,
                  'clean_truncated':len(original)>128,'perturbed_truncated':len(perturbed)>128}
        for field,value in expected.items():require(str(getattr(row,field))==str(value),f'Recorded token flag differs: {row.variant_id}, {field}')
        for field,value in [('clean_token_count',len(original)),('perturbed_token_count',len(perturbed)),
                            ('token_count_delta',len(perturbed)-len(original))]:
            require(int(getattr(row,field))==value,f'Recorded token count differs: {row.variant_id}, {field}')
    return {'status':'all_exported_token_flags_reproduced','variant_rows':len(changed),'max_length':128}


def paired_intensity_tables(changed,decisions,repetitions=2000,minimum=30):
    """Compare doses on identical IDs; separate from differently sampled cohorts."""
    selected=changed[changed.rule.eq('tatweel') & changed.applicable]
    one,two=(selected[selected.intensity.eq(i)].set_index('source_row_id') for i in [1,2])
    require(set(one.index)==set(two.index),'Paired intensity IDs differ')
    two=two.loc[one.index]
    require(one.gold_id.tolist()==two.gold_id.tolist(),'Paired intensity gold differs')
    frame=pd.DataFrame({'source_row_id':one.index,'gold_id':one.gold_id.to_numpy(),
                       'clean_id':one.predicted_id.to_numpy(),'predicted_id':two.predicted_id.to_numpy(),
                       'high_risk':one.high_risk_near_duplicate.to_numpy()})
    confirmed=set(decisions.loc[decisions.eligible_confirmed.astype(str).eq('True'),'variant_id'])
    frame['both_human_confirmed']=[a in confirmed and b in confirmed for a,b in zip(one.variant_id,two.variant_id)]
    rows=[]
    for name,part in [('provisional_all_matched',frame),('confirmed_both_intensities',frame[frame.both_human_confirmed]),
                      ('confirmed_both_exclude_high_risk',frame[frame.both_human_confirmed & ~frame.high_risk])]:
        common={'cohort':name,'n_original_ids':len(part),'left':'tatweel_intensity_1','right':'tatweel_intensity_2',
                'seed':42,'bootstrap_repetitions':repetitions}
        if len(part)<minimum:rows.append({**common,'status':'insufficient_sample_size'});continue
        for metric,values in paired_bootstrap(part,repetitions).items():
            # In this table clean_* means intensity 1, perturbed_* intensity 2.
            rows.append({**common,'status':'computed','metric':metric,**values})
    return pd.DataFrame(rows),frame


def analyze(args, config):
    package, decisions = Path(args.review_package), Path(args.decisions)
    package_hashes, decision_hashes = registry(package), registry(decisions)
    coverage = read(decisions/'review_coverage.json')
    require(coverage['human_review_completed'], 'Human review incomplete: stop at prepared-review stage')
    require({'reviewer_a.csv','reviewer_b.csv','queue_registry.csv'} <= set(package_hashes),
            'Review package registry lacks required assignments')
    require({'resolved_decisions.csv','independent_reviewer_a.csv','independent_reviewer_b.csv',
             'review_coverage.json','audit.json','run_status.json'} <= set(decision_hashes),
            'Decision package registry incomplete')
    require(args.bootstrap >= 100, 'At least 100 paired bootstrap repetitions required')
    require(read(decisions/'audit.json')['review_package_registry_sha256'] == sha256(package/'artifact_hashes.json'),
            'Decision package was prepared for different review inputs')
    queue, manifest, baseline_predictions, evidence = verify_preparation(args.preparation, args.baseline, config, args.checkpoint)
    require(csv(package/'reviewer_a.csv').equals(review_template(queue)), 'Review package differs from original queue')
    clean, changed, inference_hashes = validate_saved_predictions(args.inference, manifest, baseline_predictions, evidence)
    final = csv(decisions/'resolved_decisions.csv')
    require(set(final.variant_id) == set(queue.variant_id), 'Resolved review coverage incomplete')
    template = csv(package/'reviewer_a.csv')
    a = validate_decisions(decisions/'independent_reviewer_a.csv', template)
    b = validate_decisions(decisions/'independent_reviewer_b.csv', template)
    adjudication = decisions/'adjudication_decisions.csv'
    require(not adjudication.exists() or adjudication.name in decision_hashes, 'Unregistered adjudication decisions')
    resolved, _, _, verified_coverage = resolve_reviews(template, a, b, adjudication if adjudication.exists() else None)
    expected = pd.read_csv(io.StringIO(resolved.to_csv(index=False)), dtype=str, keep_default_na=False)
    require(final.equals(expected) and coverage == verified_coverage, 'Resolved decisions differ from explicit human records')
    tables = confirmed_tables(changed, final, args.bootstrap)
    tokenizer = None
    tokenizer_hashes = None
    if args.checkpoint:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    elif getattr(args,'tokenizer_directory',None):
        from .verified_tokenizer import verify
        from transformers import AutoTokenizer
        tokenizer_hashes=verify(args.tokenizer_directory,evidence['checkpoint_sha256'])
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_directory,local_files_only=True,trust_remote_code=False)
    details, tatweel = tatweel_comparison(changed, clean, manifest, tokenizer)
    token_reproduction=verify_exported_token_flags(changed,manifest,tokenizer) if tokenizer else None
    intensity_tables,intensity_ids=paired_intensity_tables(changed,final,args.bootstrap)
    output = fresh(args.output)
    tables.to_csv(output/'confirmed_robustness.csv', index=False)
    (output/'confirmed_robustness.tex').write_text(tables.to_latex(index=False, float_format='%.4f'))
    details.to_csv(output/'tatweel_paired_audit.csv', index=False)
    tatweel.to_csv(output/'tatweel_summary.csv', index=False)
    (output/'tatweel_summary.tex').write_text(tatweel.to_latex(index=False))
    intensity_tables.to_csv(output/'tatweel_matched_intensity_metrics.csv',index=False)
    intensity_ids.to_csv(output/'tatweel_matched_intensity_ids.csv',index=False)
    # Original provisional scores stay byte-identical, physically separate.
    source = Path(args.inference)/'robustness_metrics.csv'
    require(source.name in inference_hashes, 'Provisional metrics absent from verified evidence')
    (output/'task04_original_metrics.csv').write_bytes(source.read_bytes())
    save(output/'evidence.json', {**evidence, 'task04_inference_sha256':inference_hashes,
        'decisions_registry_sha256':sha256(decisions/'artifact_hashes.json'), 'seed':42,
        'bootstrap_repetitions':args.bootstrap, 'task05_code_sha256':sha256(Path(__file__)),
        'token_comparison_checkpoint_bytes_verified':args.checkpoint is not None,
        'selected_tokenizer_files_verified':tokenizer_hashes,
        'exported_token_flags_reproduction':token_reproduction,
        'tokenizer_model_weights_loaded':False})
    status = {'status':'completed', **coverage, 'training_performed':False, 'test_evaluated':False,
              'confirmed_metrics':'confirmed_robustness.csv', 'checkpoint_reselection':False}
    save(output/'run_status.json',status)
    (output/'methodology.md').write_text(methodology(status), encoding='utf-8')
    seal(output)
    return status


def methodology(status):
    return '''# Task 05 semantic validation and negation scope

The original Task 04 404-item stratified queue is fixed. Original and perturbed
Arabic strings, stable IDs and SHA-256 checksums are retained exactly. Review
packages omit gold labels, model predictions, confidences and lexical negation
flags. All decision cells start blank. The offline interface presents text in
RTL, saves a UTF-8 BOM CSV and supports restoring a previous CSV. CSVs are the
authoritative records; do not edit identifiers or text. Store full-text packages
and human identities in durable private storage, separate from public reports.

Two Arabic-speaking reviewers independently assess each pair before discussing
decisions. Judge meaning and sentiment polarity, emphasis, entities, irony and
dialect. Identify negation cues and describe their scope and sentiment effects
in both texts; a lexical cue is not proof of scope. Record uncertainty rather
than guessing. Notes and timezone-aware timestamps are required. Present
negation requires cue and both scope descriptions. Disagreements in any
categorical decision or scope description go to a third independent reviewer.
The adjudicator records an explicit decision and rationale after inspecting
both original judgments in the separate adjudication audit files. No model
results are revealed during review. Do not use the model as a semantic judge.

Agreement is measured before adjudication for each categorical field: exact
agreement and nominal Cohen kappa on jointly completed rows. Kappa is unavailable
with no pairs or expected agreement of one. Coverage denominators are always
404 assigned variants, with stratum coverage separately reported. Free-text
scope differences require adjudication but do not receive numerical kappa.
No automated sentence or token heuristic creates human decisions. Eligibility
requires two agreeing explicit human decisions, or third-reviewer adjudication:
meaning preserved, sentiment yes, negation presence and effects not uncertain.
Changed, ambiguous, unresolved and unreviewed cases are excluded from confirmed
meaning-preserving analysis. Even sentiment-stable meaning changes are excluded.

Analysis is gated until all 404 assignments have both independent reviews and
all disagreements are resolved. It reads saved predictions only; no fitting,
checkpoint selection or official test text access occurs. Original Task 04
metrics are copied byte-for-byte into a separate file with their original status.
Confirmed metrics are matched clean/perturbed scores by rule and intensity,
including a sensitivity subset excluding Task 04 high-risk near duplicates.
Negation presence and sentiment effects use human judgments; lexical heuristic
groups are separately labeled. Subgroups below 30 original IDs report sample
size and unavailable metrics. Macro-F1 uses fixed NEG/NEU/POS, absent-class F1=0.
95% percentile CIs use 2,000 paired resamples by original text ID, seed 42;
one variant per ID in each stratum. Intervals condition on this checkpoint,
selected review sample and decisions, exclude training-seed uncertainty and
are not simultaneous. Group comparisons are descriptive; overlapping intervals
or point differences alone do not establish a difference between populations.

Tatweel comparisons pair intensities 1 and 2 on identical original IDs and
report clean-versus-1, clean-versus-2 and 1-versus-2. Probability differences
include exact numeric inequality and maximum absolute difference above 1e-6
to separate tiny roundoff from larger changes. Full token-ID sequences cannot
be inferred from lengths or changed-versus-clean flags. With a byte-verified
selected checkpoint tokenizer, effective IDs (special tokens, truncation at
128) are recomputed without loading or changing the model and saved for audit.
Without tokenizer access that comparison is explicitly unavailable. Counts on
the full candidate population remain provisional instability, not confirmed
semantic robustness. Tokenization associations do not establish causality.

This nonrandom stratified 404-variant sample covers at most those reviewed
examples; approval is never propagated to other variants or whole rules.
Dialect and irony may defeat simple scope judgments. Review identities are
self-reported audit records, not cryptographic proof of independent reviewers.
Checksums detect altered bytes but are not a signature of provenance.

## Actual execution status

```json
''' + json.dumps(status, ensure_ascii=False, indent=2) + '\n```\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/marbert_v1.json')
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ['preparation','baseline','output']: prep.add_argument('--'+name,required=True)
    ingest = sub.add_parser('import-reviews')
    for name in ['review-package','reviewer-a','reviewer-b','output']: ingest.add_argument('--'+name,required=True)
    ingest.add_argument('--adjudication')
    analysis = sub.add_parser('analyze')
    for name in ['preparation','baseline','review-package','decisions','inference','output']:
        analysis.add_argument('--'+name,required=True)
    analysis.add_argument('--checkpoint')
    analysis.add_argument('--tokenizer-directory',help='Only byte-verified selected tokenizer files; no model weight fallback')
    analysis.add_argument('--bootstrap',type=int,default=2000)
    args = parser.parse_args(argv)
    try:
        if args.command=='analyze':require(not (args.checkpoint and args.tokenizer_directory),'Choose checkpoint or verified tokenizer directory')
        if args.command == 'prepare': result = prepare(args.preparation,args.baseline,args.output,read(args.config))
        elif args.command == 'import-reviews': result = import_reviews(args.review_package,args.reviewer_a,args.reviewer_b,args.adjudication,args.output)
        else: result = analyze(args,read(args.config))
        print(json.dumps(result,indent=2))
    except Exception as exc:
        parser.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__': main()
