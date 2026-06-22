# Model card

## Components

The release bundle contains three independently trained PaddlePaddle components:

- a binary character/icon object detector;
- a single-character recognizer;
- an icon classifier.

The final pipeline combines detection, per-candidate classification, OCR alternatives,
semantic lexicon decoding and instruction-order inference.

## Distribution

Weights are not committed to Git. They are published as assets of the private
`models-v0.1.0` GitHub Release and verified with SHA-256 hashes from
`src/gsxt_solver/assets/models.json`.

## Intended use

Use the models only with images and systems that you own or are explicitly authorized to
test. Evaluate accuracy on the target domain before relying on the output.

## Limitations

- Accuracy depends on detector coverage and the similarity of inputs to the training data.
- Rare Chinese characters, visually similar icons and ambiguous instructions can still be
  misclassified.
- GPU compatibility depends on matching PaddlePaddle, CUDA and CUDNN versions.
- The source-code license does not automatically grant redistribution rights for third-party
  training data.

## Provenance

Before making the repository or model release public, document every external dataset,
font, icon source and corpus, including its license and redistribution terms.
