# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'regression'

# the header (all column names) of the input data file(s)
HEADERS = ['key','x','y','alpha','beta','target']

# the default values of all the columns of the input data, to help TF detect the data types of the columns
HEADER_DEFAULTS = [[0.], [0.], [0.], [''], [''], [0.]]

# column of type int or float
NUMERIC_FEATURE_NAMES = ["x", "y"]

# categorical features with few values (to be encoded as one-hot indicators)
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01','ax02'], 'beta': ['bx01', 'bx02']}

# categorical features with many values (to be treated using embedding)
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# all the categorical feature names
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# all the feature names to be used in the model
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

# target feature name (response or class variable)
TARGET_NAME = 'target'

# the class values target feature in a classification dataset
TARGET_LABELS = []

# column to be ignores (e.g. keys, constants, etc.)
UNUSED_FEATURE_NAMES = ['key']