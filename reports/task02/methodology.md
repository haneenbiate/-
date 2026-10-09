# Task 02: leakage-safe splitting methodology

No model was trained. No evaluation scores or model-selection results are claimed.

## Source and baseline recovery

Official source: https://github.com/iabufarha/ArSarcasm-v2, commit `eb85513e133cfc67e30aca4968fadbe2594fcb42`. Checksums are in metadata.json.
The restarted workspace contained only its initial README. Earlier Task 01 code, tests, reports and manifests were absent. Baseline assignments were reconstructed from the previously documented seed-42, 20% sentiment-stratified train_test_split, not read from recovered artifacts. Baseline overlap values below are newly measured on that reconstruction.

## Canonical grouping

Remove U+0640 tatweel, apply Unicode NFC, collapse Unicode whitespace and strip leading/trailing whitespace. Do not casefold, apply NFKC, fold hamza/alef/ya/ta marbuta, strip diacritics, remove punctuation/URLs/negation, or otherwise clean text. Canonical keys are for grouping only; raw CSVs and annotations remain unchanged. NFC combines canonically equivalent decompositions, not distinct Arabic letters. IDs are full SHA-256 canonical text hashes prefixed ct1_. Row IDs are zero-based CSV data-record positions anchored by file checksums.

## Quarantine and group-disjoint assignment

Quarantine every official training row whose canonical group occurs in official test text. Only test text is consulted for this decision. Quarantine rows are excluded from both fitting and validation, but retained in the master manifest with explicit reasons. All other training rows are retained, including duplicate rows and conflicting annotations; no majority label, deduplication or relabeling is applied. Each eligible canonical group stays wholly in one partition.
Seed: 42. Target validation row fraction: 0.2. Actual fraction after quarantine: 0.199984.
Generate 512 GroupShuffleSplit candidates from eligible official training data. Group counts target the requested fraction; row/class balance is approximate because group sizes differ. Rank candidates lexicographically by number of missing training classes, number of missing validation classes, then mean per-class validation count error (divided by each class total) plus row-fraction error. Break ties by candidate order. Only eligible training sentiment labels enter this optimization; test labels, test scores and near-duplicate findings do not. The finite search does not guarantee global optimal balance. Missing-class counts are in metadata.

## Deviations from official training

No source records are deleted or rewritten. The sole eligibility exclusion is test-text quarantine. The remaining official training rows are assigned to new group-disjoint train and validation partitions; these differ from the old row-wise split. training_deviations.csv lists every original training row, its old/new assignment, and any quarantine reason. quarantine_manifest.csv enumerates every excluded row. No conflict or near-duplicate exclusions are applied. Deviations are in manifests only.

| Count | Value |
| --- | ---: |
| train | 10037 |
| validation | 2509 |
| test | 3000 |
| quarantine | 2 |
| official_train | 12548 |
| official_test | 3000 |
| source_rows_deleted | 0 |
| quarantine_exact_match | 2 |
| quarantine_canonical_only | 0 |
| unique_nonoverlap_test_rows | 2994 |
| overlapping_test_rows | 2 |
| near_duplicate_review_pairs | 753 |

## Exact and canonical overlap

A group is one shared comparison text. Row pairs count every cross-partition combination. A conflicting pair has unequal sentiment labels; a conflicting group has at least one such cross-partition pair. Quarantine is excluded from Task 02 train/validation overlap checks.

| Stage | Comparison | Scope | Groups | Left rows | Right rows | Pairs | Conflict groups | Conflict pairs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reconstructed_task01 | exact_text | train_validation | 18 | 18 | 19 | 19 | 10 | 10 |
| reconstructed_task01 | exact_text | train_test | 2 | 2 | 2 | 2 | 0 | 0 |
| reconstructed_task01 | exact_text | validation_test | 0 | 0 | 0 | 0 | 0 | 0 |
| reconstructed_task01 | canonical_text | train_validation | 18 | 18 | 19 | 19 | 10 | 10 |
| reconstructed_task01 | canonical_text | train_test | 2 | 2 | 2 | 2 | 0 | 0 |
| reconstructed_task01 | canonical_text | validation_test | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | exact_text | train_validation | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | exact_text | train_test | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | exact_text | validation_test | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | canonical_text | train_validation | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | canonical_text | train_test | 0 | 0 | 0 | 0 | 0 | 0 |
| task02 | canonical_text | validation_test | 0 | 0 | 0 | 0 | 0 | 0 |

## Distributions

| Split | Field | Label | Count | Proportion |
| --- | --- | --- | ---: | ---: |
| train | sentiment | NEG | 3696 | 0.368238 |
| train | sentiment | NEU | 4598 | 0.458105 |
| train | sentiment | POS | 1743 | 0.173657 |
| train | sarcasm | FALSE | 8330 | 0.829929 |
| train | sarcasm | TRUE | 1707 | 0.170071 |
| train | dialect | msa | 6833 | 0.680781 |
| train | dialect | egypt | 2137 | 0.212912 |
| train | dialect | gulf | 520 | 0.051808 |
| train | dialect | levant | 512 | 0.051011 |
| train | dialect | magreb | 35 | 0.003487 |
| validation | sentiment | NEG | 923 | 0.367876 |
| validation | sentiment | NEU | 1149 | 0.457951 |
| validation | sentiment | POS | 437 | 0.174173 |
| validation | sarcasm | FALSE | 2049 | 0.816660 |
| validation | sarcasm | TRUE | 460 | 0.183340 |
| validation | dialect | msa | 1728 | 0.688721 |
| validation | dialect | egypt | 537 | 0.214029 |
| validation | dialect | gulf | 124 | 0.049422 |
| validation | dialect | levant | 112 | 0.044639 |
| validation | dialect | magreb | 8 | 0.003189 |
| test | sentiment | NEG | 1677 | 0.559000 |
| test | sentiment | NEU | 748 | 0.249333 |
| test | sentiment | POS | 575 | 0.191667 |
| test | sarcasm | FALSE | 2179 | 0.726333 |
| test | sarcasm | TRUE | 821 | 0.273667 |
| test | dialect | msa | 2323 | 0.774333 |
| test | dialect | egypt | 306 | 0.102000 |
| test | dialect | gulf | 322 | 0.107333 |
| test | dialect | levant | 47 | 0.015667 |
| test | dialect | magreb | 2 | 0.000667 |
| quarantine | sentiment | NEG | 2 | 1.000000 |
| quarantine | sentiment | NEU | 0 | 0.000000 |
| quarantine | sentiment | POS | 0 | 0.000000 |
| quarantine | sarcasm | FALSE | 1 | 0.500000 |
| quarantine | sarcasm | TRUE | 1 | 0.500000 |
| quarantine | dialect | msa | 1 | 0.500000 |
| quarantine | dialect | egypt | 1 | 0.500000 |
| quarantine | dialect | gulf | 0 | 0.000000 |
| quarantine | dialect | levant | 0 | 0.000000 |
| quarantine | dialect | magreb | 0 | 0.000000 |
| official_train | sentiment | NEG | 4621 | 0.368266 |
| official_train | sentiment | NEU | 5747 | 0.458001 |
| official_train | sentiment | POS | 2180 | 0.173733 |
| official_train | sarcasm | FALSE | 10380 | 0.827223 |
| official_train | sarcasm | TRUE | 2168 | 0.172777 |
| official_train | dialect | msa | 8562 | 0.682340 |
| official_train | dialect | egypt | 2675 | 0.213181 |
| official_train | dialect | gulf | 644 | 0.051323 |
| official_train | dialect | levant | 624 | 0.049729 |
| official_train | dialect | magreb | 43 | 0.003427 |
| official_test | sentiment | NEG | 1677 | 0.559000 |
| official_test | sentiment | NEU | 748 | 0.249333 |
| official_test | sentiment | POS | 575 | 0.191667 |
| official_test | sarcasm | FALSE | 2179 | 0.726333 |
| official_test | sarcasm | TRUE | 821 | 0.273667 |
| official_test | dialect | msa | 2323 | 0.774333 |
| official_test | dialect | egypt | 306 | 0.102000 |
| official_test | dialect | gulf | 322 | 0.107333 |
| official_test | dialect | levant | 47 | 0.015667 |
| official_test | dialect | magreb | 2 | 0.000667 |

## Duplicates and label conflicts

exact_text: 65 duplicate groups, 26 with conflicting sentiment labels. See duplicate_groups.csv for all member IDs and labels, group_members.csv for assignments.
canonical_text: 65 duplicate groups, 26 with conflicting sentiment labels. See duplicate_groups.csv for all member IDs and labels, group_members.csv for assignments.

## Official and secondary evaluation manifests

test_manifest.csv retains all official test rows, order and annotations. Input checksums are verified before and after preparation. test_evaluation_manifest.csv also retains every test row; it marks exact/canonical training overlap, canonical group size, and the first source row as a deterministic canonical representative. unique_nonoverlap identifies representatives with no original-training overlap. The exclusive categories are overlapping_official_train, unique_nonoverlap, and repeated_test_only. These flags are text-only. “Unique” here means canonical identity, not semantic uniqueness.
Use validation alone for model/hyperparameter selection. After freezing all choices, report evaluation on the full official test and optionally the predeclared secondary unique_nonoverlap slice. Neither test labels, overlap-label conflicts nor metrics may drive selection. No test examples are moved into training or validation.

## Near-duplicate inspection only

near_duplicate_review.csv uses canonical char TF-IDF 3–5 grams and exhaustive blockwise cosine >= 0.9 on combined source text, excluding equal canonical keys. It does not change groups, eligibility or assignments. Short texts below three characters have no near features. It is lexical, not semantic equivalence, with possible false positives/negatives. Features built from test text are audit-only. No manual review or automatic merging is claimed. Canonical disjointness does not establish absence of semantic leakage.

## Reproduction and evidence

Run commands in README.md. Outputs require a new/empty directory. metadata.json records seed, algorithm, source hashes, versions and CSV artifact hashes. Automated tests cover meaning-sensitive character preservation, quarantine, group disjointness, label conflicts, source immutability, deterministic manifests and test-label independence. No future publication or model behavior is implied.
