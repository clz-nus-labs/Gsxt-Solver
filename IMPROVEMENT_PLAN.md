# GSXT Solver improvement plan

This document defines the next development steps for improving task understanding,
character recognition, icon matching, ordering accuracy, and generalization.

## 1. Establish a measurable baseline

The first priority is to annotate the 50 bundled evaluation images without using them as
training samples.

Create a machine-readable manifest containing:

- task action: explicit order, semantic order, or detection only;
- modality: character, icon, or mixed;
- expected target source: header text, header icons, or body semantics;
- expected ordered labels;
- expected candidate count;
- bounding boxes and labels where practical;
- known ambiguity notes.

Report separate metrics instead of only an overall pass rate:

| Metric | Purpose |
| --- | --- |
| Task-action accuracy | Whether the instruction meaning was understood |
| Modality accuracy | Whether the task was classified as character or icon |
| Detection precision/recall | Whether body candidates were missed or over-detected |
| Character top-1/top-k accuracy | Quality of the single-character recognizer |
| Icon top-1/top-k accuracy | Quality of icon recognition |
| Exact-order accuracy | Whether the final sequence is completely correct |
| False-extra rate | Frequency of background items appended to the result |

This separation is essential: a wrong final order can come from instruction parsing,
detection, recognition, or global matching, and each requires a different fix.

## 2. Improve task-meaning inference

Current task interpretation relies on multiple thresholded hypotheses. The next version
should retain and score competing task specifications instead of committing early.

Recommended changes:

1. OCR the instruction region and target region separately.
2. Produce multiple `TaskSpec` hypotheses:
   - explicit character order;
   - explicit icon order;
   - semantic character order;
   - detection only.
3. Score hypotheses using independent evidence:
   - instruction-language markers;
   - prompt-region geometry;
   - number and quality of header icons;
   - body character/icon compatibility;
   - whether the proposed targets can be matched globally.
4. Select the hypothesis only after body recognition and matching.
5. Add an `uncertain` result when the best and second-best hypotheses are too close.

This avoids dataset-specific rules such as assuming that a particular language or header
position always implies one modality.

## 3. Character recognition: top-k plus lexicon joint decoding

The character path should use recognition alternatives directly rather than filling a
word after committing to low-confidence top-1 characters.

Planned decoder:

1. Keep the top-k character candidates and log probabilities for every detected region.
2. Build candidate sequences using beam search.
3. Score each sequence with:
   - OCR log probability;
   - word-frequency prior;
   - character bigram or language-model probability;
   - geometric/order consistency;
   - penalties for duplicate use and unmatched targets.
4. Return both the selected sequence and the competing hypotheses.

Training improvements:

- mine hard-confusion pairs from the 50-image evaluation output;
- add real cropped characters from newly labelled training images;
- expand fonts, blur, compression, color, rotation, occlusion, and background synthesis;
- rebalance rare characters and visually similar classes;
- calibrate recognition confidence on a held-out validation set.

The evaluation fixtures must remain outside the training set.

## 4. Icon recognition and prompt-to-body matching

A fixed closed-set icon classifier is fragile when prompt and body icons have different
scales, colors, or rendering styles.

Recommended hybrid approach:

1. Keep the current classifier top-k probabilities.
2. Train an embedding or Siamese model on matched prompt/body icon pairs.
3. Compute prompt-to-body visual similarity for every pair.
4. Combine classifier probability, embedding similarity, detector confidence, and geometry.
5. Use Hungarian matching for one-to-one global assignment.
6. Output only successfully matched prompt targets for explicit-order tasks.

Training data should include:

- the same icon rendered at different scales and colors;
- prompt fragments and partial crops;
- visually similar negative pairs;
- background shapes that previously became false icons;
- header and body versions of each icon.

This approach generalizes better to new icons because direct visual matching does not
depend entirely on a predefined class label.

## 5. Detection and duplicate suppression

Extra items and fragmented prompt detections should be addressed at both training and
post-processing levels.

Detection training:

- label prompt region and body region separately;
- add hard-negative backgrounds from false detections;
- include overlapping, tiny, blurred, and partially clipped targets;
- include header fragments as negatives for body-candidate detection;
- tune the validation set for precision as well as recall.

Post-processing:

- use class-aware NMS or weighted box fusion;
- merge boxes using IoU, containment, center distance, and size similarity;
- suppress boxes inconsistent with the selected task modality;
- enforce target-count constraints only after task interpretation succeeds;
- never append unmatched background candidates to explicit-order output.

The rules should depend on confidence and geometry, not individual test numbers.

## 6. Global structured decoding

Final selection should be treated as a constrained optimization problem.

For an explicit target sequence, construct a score matrix between prompt targets and body
candidates. Optimize a one-to-one assignment with:

- recognition/classification score;
- prompt/body visual similarity;
- target-order compatibility;
- duplicate-use penalty;
- unmatched-target penalty;
- background-candidate penalty.

For semantic character ordering, jointly optimize:

- candidate character alternatives;
- word segmentation;
- phrase frequency;
- spatial candidate selection.

This creates one consistent decision layer instead of a chain of irreversible local
decisions.

## 7. Active-learning loop

After every regression run:

1. collect low-confidence and incorrect cases;
2. classify each error by stage;
3. add only new training examples or general rules that address that error category;
4. retrain the affected model;
5. rerun the full held-out suite;
6. reject changes that improve one subset while degrading another.

Suggested error categories:

- instruction/action error;
- modality error;
- missed detection;
- extra detection;
- duplicate/fragmented detection;
- character confusion;
- icon confusion;
- ordering/global assignment error.

## 8. Packaging and deployment

The current package invokes source-based PaddleDetection and PaddleOCR components. A later
release should export static inference models and provide an in-process runtime.

Goals:

- remove the requirement to clone full upstream source repositories;
- support a normal `pip install` plus model download;
- provide pinned CPU and GPU environment examples;
- add Windows and Linux CI import tests;
- add model-version compatibility checks;
- keep the public `Solver` API stable while replacing the backend.
