TASK_TYPE = 'regression'

HEADERS = ['key','x','y','alpha','beta','target']

HEADER_DEFAULTS = [[0.], [0.], [0.], [''], [''], [0.]]

NUMERIC_FEATURE_NAMES = ["x", "y"]

# categorical features with few values
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01','ax02'], 'beta': ['bx01', 'bx02']}

# categorical features with many values
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES


TARGET_NAME = 'target'

TARGET_LABELS = []

UNUSED_FEATURE_NAMES = ['key']