import math
import pandas as pd

import inference


PROJECT='ksalama-gcp-playground'
MODEL_NAME='synth_classifier'
VERSION='v1'


def compute_accuracy(predicted, actual):
    accuracy = ((pd.Series(predicted) == actual).astype(float).sum() / len(actual))*100
    return accuracy


test_data = pd.read_csv("data/test-data.csv", header=None, names="key,x,y,alpha,beta,target".split(','))

test_instances = list(test_data.apply(lambda row: {'x': row['x'], 'y': row['y'], 'alpha': row['alpha'], 'beta': row['beta']}
                                 , axis=1))


predictions = inference.predict(instances=test_instances
                     ,project=PROJECT
                     ,model_name=MODEL_NAME
                     ,version=VERSION)


accuracy = compute_accuracy(predictions,test_data.target)

print("Test Accuracy: {}%".format(round(accuracy,3)))
