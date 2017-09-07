import metadata


def extend_feature_columns(feature_columns):
    """ Use to define additional feature columns, such as bucketized_column and crossed_column
    Default behaviour is to return the original feature_column list as is

    Args:
        feature_columns: [tf.feature_column] - list of base feature_columns to be extended
    Returns:
        [tf.feature_column]: extended feature_column list
    """

    return feature_columns


def process_features(features):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion
    Default behaviour is to return the original feature tensors dictionary as is

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensor dictionary
    """

    return features

