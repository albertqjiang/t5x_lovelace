import seqio
import tensorflow as tf
import t5
from t5.evaluation import metrics


vocabulary = t5.data.get_default_vocabulary()
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=False, dtype=tf.int32),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True, dtype=tf.int32)
}

seqio.TaskRegistry.add(
    "tactic_prediction",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/train.tfrecord",
            "val": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/val.tfrecord",
            "test": "gs://n2formal-public-data-europe/albert/finetuning_data/tactic_prediction_data/test.tfrecord",
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
    metric_fns=[metrics.bleu, metrics.accuracy, metrics.rouge])