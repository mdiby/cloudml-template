import tensorflow as tf
import metadata


def parse_csv(rows_string_tensor):

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=metadata.HEADER_DEFAULTS)
    features = dict(zip(metadata.HEADERS, columns))

    # Remove unused columns
    for col in metadata.UNUSED_FEATURE_NAMES:
        features.pop(col)

    return features
