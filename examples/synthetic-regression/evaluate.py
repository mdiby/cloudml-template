import math
import pandas as pd

import inference


PROJECT='ksalama-gcp-playground'
MODEL_NAME='synth_regressor'
MODEL_VERSION='v1'


def compute_rmse(estimates, actual):
    rmse = math.sqrt(((pd.Series(estimates) - actual) ** 2).sum() / len(actual))
    return rmse

test_data = pd.read_csv("data/test-data.csv", header=None, names="key,x,y,alpha,beta,target".split(','))

test_instances = list(test_data.apply(
    lambda row: {'x': row['x'], 'y': row['y'], 'alpha': row['alpha'], 'beta': row['beta']}
    , axis=1))


estimates = inference.estimate(instances=test_instances
                     ,project=PROJECT
                     ,model_name=MODEL_NAME
                     ,version=MODEL_VERSION)


rmse = compute_rmse(estimates,test_data.target)

print("Test RMSE: {}".format(round(rmse,3)))
