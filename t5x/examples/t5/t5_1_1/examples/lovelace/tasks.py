import seqio
import tensorflow as tf
import t5
import functools
from t5.evaluation import metrics
from t5.data import preprocessors


# vocabulary = t5.data.get_default_vocabulary()
vocabulary = seqio.SentencePieceVocabulary(sentencepiece_model_file="gs://n2formal-public-data-europe/albert/tokenizer/galactica_enhanced.model")
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=False, dtype=tf.int32),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True, dtype=tf.int32)
}

seqio.TaskRegistry.add(
    "isabelle_tactic_prediction",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/isabelle/train.tfrecord",
            "validation": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/isabelle/valid.tfrecord",
            "test": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/isabelle/test.tfrecord",
        },
        feature_description={
            "inputs": tf.io.FixedLenFeature([], tf.string),
            "targets": tf.io.FixedLenFeature([], tf.string)
        }
    ),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu, metrics.accuracy, metrics.rouge]
)

seqio.TaskRegistry.add(
    "lean_tactic_prediction",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/lean/train.tfrecord",
            "validation": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/lean/valid.tfrecord",
            "test": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/lean/test.tfrecord",
        },
        feature_description={
            "inputs": tf.io.FixedLenFeature([], tf.string),
            "targets": tf.io.FixedLenFeature([], tf.string)
        }
    ),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.bleu, metrics.accuracy, metrics.rouge]
)


seqio.MixtureRegistry.add(
  "isabelle_lean_even_mixture",
  [("isabelle_tactic_prediction", 2357364), ("lean_tactic_prediction", 145970)]
)

seqio.TaskRegistry.add(
    "c4_v220_span_corruption",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])