import tensorflow as tf
import metadata
import featurizer
import parsers
import preprocess


def json_serving_input_fn():

    feature_columns = featurizer.create_feature_columns()

    inputs = {}

    for feat in feature_columns:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,
        inputs
    )


def csv_serving_input_fn():

    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )

    features = parsers.parse_csv(csv_row)
    features.pop(metadata.TARGET_NAME)
    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,
        {'csv_row': csv_row}
    )


def example_serving_input_fn():

    feature_columns = featurizer.create_feature_columns()

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    feature_scalars = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(feature_columns)
    )
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_scalars.iteritems()
    }
    return tf.contrib.learn.InputFnOps(
        preprocess.process_features(features),
        None,  # labels
        {'example_proto': example_bytestring}
    )

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}