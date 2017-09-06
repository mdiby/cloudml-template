import tensorflow as tf
import metadata
import preprocess
from tensorflow.python.feature_column import feature_column


def create_feature_columns():

    numeric_columns = list(
        map(lambda feature_name: tf.feature_column.numeric_column(feature_name),
            metadata.NUMERIC_FEATURE_NAMES)
    )

    categorical_column_with_vocabulary = list(
        map(lambda item: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1]),
            metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items())
    )

    categorical_column_with_hash_bucket = list(
        map(lambda item: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1]),
            metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.items())
    )

    feature_columns = numeric_columns + categorical_column_with_vocabulary + categorical_column_with_hash_bucket

    return preprocess.extend_feature_columns(feature_columns)


def get_deep_and_wide_columns(feature_columns, embedding_size=0, use_indicators=True):

    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn),
               feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column,feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
                   feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column,feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = []

    if use_indicators:

        indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
        )

    embedding_columns = []

    if embedding_size > 0:

        embedding_columns = list(
            map(lambda sparse_column: tf.feature_column.embedding_column(sparse_column, dimension=embedding_size),
                sparse_columns)
        )

    deep_columns = dense_columns + indicator_columns + embedding_columns
    wide_columns = categorical_columns + sparse_columns

    return deep_columns, wide_columns
