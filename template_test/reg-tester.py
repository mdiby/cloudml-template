import argparse

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.training.python.training import hparam

from template import metadata
from template import featurizer
from template import input
from template import parsers
from template import parameters
from template import model


from template import task



metadata.TASK_TYPE="regression"
metadata.HEADERS = "key,x,y,alpha,beta,target".split(",")
metadata.HEADER_DEFAULTS = [[0.], [0.], [0.], [''], [''], [0.]]
metadata.NUMERIC_FEATURE_NAMES = ["x", "y"]
metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha':['ax01','ax02'], 'beta':['bx01', 'bx02']}

# metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpa':['ax01','ax02']}
# metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET= {'beta':10}

metadata.UNUSED_FEATURE_NAMES = ['key']
metadata.TARGET_NAME = 'target'

print("")

print(metadata.HEADERS)
print("")

feature_columns = featurizer.create_feature_columns()
print(feature_columns)
print("")


deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(feature_columns)
#deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(feature_columns, use_indicators=True, embedding_size=5)
print("deep columns:{}".format(deep_columns))
print("wide columns:{}".format(wide_columns))
print("")

data_file = "data/regression-data.csv"
#
features, target = input.generate_text_input_fn(
    [data_file],
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    parser_fn=parsers.parse_csv
)
#
print(features.keys())
print("")

args_parser = argparse.ArgumentParser()
args = parameters.initialise_arguments(args_parser)

parameters.HYPER_PARAMS=hparam.HParams(**args.__dict__)
run_config=run_config.RunConfig(model_dir=args.job_dir)

print(parameters.HYPER_PARAMS)
print("")


estimator=model.create_regressor(run_config)
print(estimator)
print("")

print("Experiment Started")
task.main()
print("Experiment Finished")


