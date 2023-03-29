import seqio
import tensorflow as tf
import t5
import functools
from t5.evaluation import metrics
from t5.data import preprocessors

vocabulary = seqio.SentencePieceVocabulary(sentencepiece_model_file="gs://n2formal-public-data-europe/albert/tokenizer/galactica_enhanced.model")
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=False, dtype=tf.int32),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True, dtype=tf.int32)
}

seqio.TaskRegistry.remove("c4_v220_span_corruption")
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