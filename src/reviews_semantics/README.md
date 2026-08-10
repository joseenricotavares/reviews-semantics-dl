# reviews_semantics

Olist review-score classification service: given the free-text comment of a
Brazilian e-commerce order review, predicts the 1-5 star rating the customer
gave. Built on top of [`dlkit`](../dlkit/README.md); this package is the
concrete example of using `dlkit`'s abstractions for one real problem, not
the only way to use them.

The served model is a from-scratch bidirectional LSTM (`paradigms/bilstm`) -
the one paradigm from the source notebook (`notebooks/review-score-semantics.ipynb`)
with a trained artifact committed to this repository. Two other paradigms
explored in the notebook (Transformer fine-tuning, frozen-encoder feature
extraction) have minimal stub implementations under `paradigms/`,
but they are **not** wired into the API or CLI.

See the repository root `README.md` for how to run the service and the
batch CLI.
