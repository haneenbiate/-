"""Task 06A: audit pre-adjudication decisions without claiming human provenance."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from arsarcasm_audit.audit import sha256
from .preflight import require
from .task05 import (DECISIONS, VALUES, HUMAN, csv, fresh, registry, save, seal,
                     validate_decisions, resolve_reviews)

FREE_FIELDS = ['negation_cue', 'original_negation_scope', 'perturbed_negation_scope', 'linguistic_notes']
CONFLICT_FIELDS = DECISIONS + FREE_FIELDS[:3]


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def export_originals(decisions, package, output):
    """Export literal original categorical decisions and hashes of private free text."""
    decisions, package = Path(decisions), Path(package)
    registry(decisions); registry(package)
    template = csv(package/'reviewer_a.csv')
    a = validate_decisions(decisions/'independent_reviewer_a.csv', template)
    b = validate_decisions(decisions/'independent_reviewer_b.csv', template)
    path = decisions/'adjudication_decisions.csv'
    final, conflicts, _, coverage = resolve_reviews(template, a, b, path)
    expected = pd.read_csv(io.StringIO(final.to_csv(index=False)), dtype=str, keep_default_na=False)
    require(expected.equals(csv(decisions/'resolved_decisions.csv')), 'Saved final decisions differ from originals')
    require(coverage == json.loads((decisions/'review_coverage.json').read_text()), 'Review coverage differs')
    adjudication = validate_decisions(path, conflicts)
    inputs = []
    for role, frame in [('A', a), ('B', b), ('ADJ', adjudication), ('FINAL', final)]:
        table = frame[['variant_id']+DECISIONS].copy()
        table['role'] = role
        for field in FREE_FIELDS: table[field] = frame[field].map(digest)
        if role == 'FINAL':
            table['eligible_confirmed'] = frame.eligible_confirmed.astype(str)
            table['decision_basis'] = frame.decision_basis
        else:
            table['eligible_confirmed'] = ''; table['decision_basis'] = ''
        inputs.append(table)
    output = fresh(output)
    pd.concat(inputs, ignore_index=True).to_csv(output/'original_decisions.csv', index=False)
    evidence = json.loads(Path('reports/task05-reviewed/evidence.json').read_text())
    declared = json.loads(Path('reports/task05-reviewed/human_review_provenance.json').read_text())
    identities = [set(f.reviewer_id) for f in [a,b,adjudication]]
    save(output/'source_provenance.json', {
        'source':'original_pre_adjudication_A_B_and_explicit_adjudicator_records',
        'source_roles':'user_declared_human', 'independence_status':'unverified',
        'authenticated_distinct_people':False, 'blinding_verified':False,
        'distinct_recorded_identifiers':not any(identities[i]&identities[j] for i in range(3) for j in range(i)),
        'recorded_completion_counts':{'A':len(a),'B':len(b),'ADJ':len(adjudication)},
        'unique_timestamp_counts':{role:int(f.review_timestamp.nunique()) for role,f in [('A',a),('B',b),('ADJ',adjudication)]},
        'unique_rationale_counts':{role:int(f.linguistic_notes.nunique()) for role,f in [('A',a),('B',b),('ADJ',adjudication)]},
        'source_workbook_sha256':declared['source_sha256'],
        'original_file_sha256':{name:sha256(decisions/name) for name in ['independent_reviewer_a.csv','independent_reviewer_b.csv','adjudication_decisions.csv','resolved_decisions.csv']},
        'private_decision_registry_sha256':sha256(decisions/'artifact_hashes.json'),
        'review_package_registry_sha256':sha256(package/'artifact_hashes.json'),
        'worksheet_a_ai_review_title_flag':evidence['worksheet_a_ai_review_title_flag'],
        'all_workbooks_same_creator_metadata':evidence['all_workbooks_same_creator_metadata'],
        'interpretation':'Metadata flags and bulk timestamps neither prove nor disprove independent human review. No authenticated identities, isolated submission logs or blinding attestations are present.',
        'free_text_export':'SHA256 of exact UTF-8 cells; literal equality only, not semantic equivalence',
        'human_decisions_modified':False, 'llm_used_for_adjudication':False,
        'test_text_read':False, 'test_evaluated':False,
        'export_code_sha256':sha256(Path(__file__))})
    seal(output)


def audit_tables(frame):
    required = {'variant_id','role','eligible_confirmed','decision_basis',*DECISIONS,*FREE_FIELDS}
    require(set(frame) == required, 'Audit input schema differs')
    require(set(frame.role)=={'A','B','ADJ','FINAL'}, 'Missing original annotation roles')
    require(not frame.duplicated(['role','variant_id']).any(), 'Duplicate annotation assignment')
    roles = {r:f.set_index('variant_id').sort_index() for r,f in frame.groupby('role')}
    a,b,adj,final = [roles[r] for r in ['A','B','ADJ','FINAL']]
    require(set(a.index)==set(b.index)==set(final.index), 'Original assignments differ')
    require(len(a)>0, 'No original paired assignments')
    for field in DECISIONS:
        require(frame[field].isin(VALUES[field]).all(), f'Invalid annotation category: {field}')
    for field in FREE_FIELDS:
        require(frame[field].str.fullmatch('[0-9a-f]{64}').all(), f'Invalid free-text digest: {field}')
    conflicts = [i for i in a.index if any(a.at[i,f]!=b.at[i,f] for f in CONFLICT_FIELDS)]
    require(set(conflicts)==set(adj.index), 'Adjudication must cover every original conflict exactly')
    eligibility = final.semantic_validity.eq('preserved') & final.sentiment_preserved.eq('yes') & final.negation_present.ne('uncertain') & final.negation_affects_sentiment.ne('uncertain')
    require(final.eligible_confirmed.isin(['True','False']).all() and
            final.eligible_confirmed.eq('True').equals(eligibility), 'Eligibility differs from recorded decisions')
    for i in final.index:
        source = adj if i in adj.index else a
        for field in DECISIONS+FREE_FIELDS[:3]:
            require(final.at[i,field]==source.at[i,field], f'Resolution differs from explicit source: {i}, {field}')
        basis = 'explicit_third_reviewer_adjudication' if i in adj.index else 'two_explicit_independent_human_decisions'
        require(final.at[i,'decision_basis']==basis, 'Resolution basis differs')
    agreement, contingency, resolutions = [], [], []
    for field in DECISIONS+FREE_FIELDS:
        x,y = a[field], b[field]
        observed = float(x.eq(y).mean())
        chance = None; kappa = None
        if field in DECISIONS:
            px,py=x.value_counts(normalize=True),y.value_counts(normalize=True)
            chance=float(sum(px.get(v,0)*py.get(v,0) for v in VALUES[field]))
            if chance < 1-1e-15: kappa=float(cohen_kappa_score(x,y,labels=VALUES[field]))
            for av in VALUES[field]:
                for bv in VALUES[field]:
                    contingency.append({'dimension':field,'A':av,'B':bv,'n':int((x.eq(av)&y.eq(bv)).sum())})
        agreement.append({'dimension':field,'paired_n':len(a),'observed_agreement':observed,
                          'disagreement_n':int(x.ne(y).sum()), 'expected_chance_agreement':chance,
                          'cohen_kappa':kappa,'kappa_status':'computed' if kappa is not None else 'undefined_constant' if field in DECISIONS else 'not_applicable_free_text',
                          'interpretation':'nominal_categories' if field in DECISIONS else 'literal_cell_equality_only_not_semantic_agreement'})
    for i in conflicts:
        diff=[f for f in CONFLICT_FIELDS if a.at[i,f]!=b.at[i,f]]
        row={'variant_id':i,'category':'categorical' if any(f in DECISIONS for f in diff) else 'scope_wording_only',
             'disagreement_fields':'|'.join(diff), 'eligible_confirmed':final.at[i,'eligible_confirmed'],
             'resolution_source':'explicit_adjudicator_record','rationale_sha256':adj.at[i,'linguistic_notes']}
        for f in CONFLICT_FIELDS:
            av,bv,fv=a.at[i,f],b.at[i,f],final.at[i,f]
            row[f+'_A']=av; row[f+'_B']=bv; row[f+'_final']=fv
            row[f+'_resolution']='both' if fv==av==bv else 'A' if fv==av else 'B' if fv==bv else 'different_from_both'
        resolutions.append(row)
    return pd.DataFrame(agreement),pd.DataFrame(contingency),pd.DataFrame(resolutions)


def audit(inputs, output):
    inputs=Path(inputs); registry(inputs)
    frame=csv(inputs/'original_decisions.csv')
    agreement,contingency,resolutions=audit_tables(frame)
    provenance=json.loads((inputs/'source_provenance.json').read_text())
    require(provenance['independence_status']=='unverified' and provenance['authenticated_distinct_people'] is False,
            'This audit has no independent human authentication evidence')
    output=fresh(output)
    for name in ['original_decisions.csv','source_provenance.json']:(output/name).write_bytes((inputs/name).read_bytes())
    agreement.to_csv(output/'agreement.csv',index=False)
    agreement.to_latex(output/'agreement.tex',index=False,float_format='%.4f')
    contingency.to_csv(output/'contingency.csv',index=False)
    resolutions.to_csv(output/'disagreement_resolutions.csv',index=False)
    summary=resolutions.groupby(['category','eligible_confirmed']).size().rename('n').reset_index()
    summary.to_csv(output/'disagreement_categories.csv',index=False)
    resolution_rows=[]
    for f in CONFLICT_FIELDS:
        for value,part in resolutions.groupby(f+'_resolution'):
            resolution_rows.append({'dimension':f,'adjudicator_matches':value,'n_all_conflicts':len(part),
                'n_with_dimension_disagreement':int(part.disagreement_fields.str.split('|').map(lambda fs:f in fs).sum())})
    pd.DataFrame(resolution_rows).to_csv(output/'resolution_summary.csv',index=False)
    final=frame[frame.role.eq('FINAL')]
    ambiguity=final.groupby(DECISIONS+['eligible_confirmed']).size().rename('n').reset_index()
    ambiguity.to_csv(output/'final_decision_categories.csv',index=False)
    status={'status':'annotation_quality_audit_completed_provenance_unverified','paired_assignments':int(len(final)),
            'conflicts':len(resolutions),'categorical_conflicts':int(resolutions.category.eq('categorical').sum()),
            'scope_wording_only_conflicts':int(resolutions.category.eq('scope_wording_only').sum()),
            'eligible_cases_user_declared_human':int(final.eligible_confirmed.eq('True').sum()),
            'human_independence_verified':False,'training_performed':False,'test_evaluated':False}
    save(output/'run_status.json',status)
    save(output/'evidence.json',{'source_registry_sha256':sha256(inputs/'artifact_hashes.json'),
        'audit_code_sha256':sha256(Path(__file__)),'original_decisions_used':True,'adjudication_used_for_agreement':False,
        'missing_provenance':['authenticated distinct people','independent collection/submission records','blinding attestations','actual AI assistance disclosure beyond user declaration'],
        'no_fabricated_identities':True})
    (output/'methodology.md').write_text('''# Task 06A annotation quality audit

Agreement is recalculated from the original A/B decisions, never from consensus
or adjudicated labels. All 404 paired assignments are complete. Four categorical
dimensions use unweighted nominal Cohen's Kappa, observed agreement and explicit
uncertain categories. Full contingency tables expose prevalence imbalance.
Kappa is undefined for identical constant marginals; free-text cues/scopes and
rationales use literal equality only. Wording differences do not establish
semantic disagreement. Administrative IDs/timestamps are not annotation labels.

213 assignments have a conflict: 150 categorical and 63 additional cue/scope
wording-only cases. Categories are exclusive; dimension counts overlap. Explicit
third-role decisions resolve all 213; 191 nonconflicting cases retain the original
categorical and cue/scope consensus. The per-case resolution table records A/B/
final categories, literal cue/scope hashes, which reviewer (if either) the final
decision matches, and a hash of the unedited adjudicator rationale. Private
original rationale/cue/scope cells remain available in preserved intake records;
their meaning is not guessed from hashes. Adjudicators can choose different
categories or wording from both reviewers. This describes recorded resolutions,
not independent evidence that every resolution is linguistically correct.

## Provenance and limits of the 394 eligible variants

The user declared corrected A/B and adjudication files to be human reviews.
Distinct recorded identifiers are present, but two independent people, isolated
collection, blinding, qualifications and actual assistance cannot be established.
Provenance is explicitly UNVERIFIED. Each A/B file has one shared timestamp across
404 rows, compatible with bulk export rather than individual activity evidence.
The earlier AI-related worksheet title and shared creator metadata are recorded
as inspection flags, not proof of automated or human origin. No identities are
fabricated and LLM decisions are never used to adjudicate this audit.

394 eligible variants over 274 original IDs are conditional on user-declared
human decisions, not independently certified meaning preservation. Ten exclusions
remain unchanged. High raw semantic/sentiment agreement is dominated by preserved/
yes marginals; semantic-validity Kappa is approximately -0.009, sentiment Kappa
0.253, negation presence 0.837 and sentiment effect 0.459. Rare changed/ambiguous
cases and substantial scope/effect disagreements warrant caution. The stratified
nonrandom sample, repeated source IDs, uncertainty exclusions, subjective scope
judgments and cross-split near-duplicate flags limit population inference. No
retroactive reclassification or automatic confirmation is performed. More human
review is needed to authenticate provenance and validate new normalization outputs.

Reproduce from the published original categorical records and exact free-text
hashes with `python -m nasn.annotation_quality audit --inputs reports/task06a
--output runs/task06a-recomputed`. Full original Arabic review cells are private;
this public reproduction verifies counts, Kappa and literal resolution integrity,
not the semantic content or independence of private human judgments.
''',encoding='utf-8')
    seal(output)
    return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--decisions',required=True);e.add_argument('--review-package',required=True);e.add_argument('--output',required=True)
    a=sub.add_parser('audit');a.add_argument('--inputs',required=True);a.add_argument('--output',required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='export':export_originals(args.decisions,args.review_package,args.output)
        else:print(json.dumps(audit(args.inputs,args.output),indent=2))
    except Exception as exc:p.exit(2,f'ERROR: {type(exc).__name__}: {exc}\n')


if __name__=='__main__':main()
