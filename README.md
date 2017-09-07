# Cloud ML Engine - Trainer Package Template

### Repository Structure
1. **template**: includes all the python module files to adapt to your data to build the ML trainer.

2. **examples**: includes two examples, classification and regression, both on synthetic data. 
The examples show how the template is adapted given a dataset. 
In addition, each example includes a python script to perform prediction (inference) via invoking a deployed model's API.

3. **scripts**: includes scripts to 1) train the model locally, 2) train the model on Cloud ML Engine, 
and 3) deploy the model on GCP as well as to make prediction (inference) using the deployed model.


### Trainer Template Modules


|File Name| Purpose| Do You Need to Change?
|:---|:---|:---
|metadata.py|Defines: 1) Task type, 2) input data header, 2) numeric and categorical feature names,  4) target feature name, and 5) unused feature names  | **Yes**, as you will need to specify the metadata of your dataset 
|featurizer.py| 1) Creates tensorflow feature_column definitions based on the metadata of the features. 2) Creates deep and wide feature column lists. | **No**
|input.py|Generates a (scalable) data input function for training or evaluation from sharded files, using file name queue, so that entire data is not loaded in memory.| **Probably No, unless** you want to implement a data input from a different source.
|parsers.py|Includes functions to parse data from text files into tensors with the proper data types (based on the default values in the metadata).|**Probably No, unless** you want to parse data files in different formats (e.g. xml, json, etc.).
|preprocess.py|Use to 1) define additional feature columns, such as bucketized_column and crossed_column, and 2) to implement custom feature engineering logic, e.g. polynomial expansion.|**Probably Yes**, in order to implement you own feature engineering logic, **unless** you input data includes all the features as well as the engineered ones.
|model.py|Includes functions to create DNNLinearCombinedRegressor and DNNLinearCombinedClassifier, based on the hyper-parameters in the parameters.py module.|**Probably No, unless** you want to change something in the estimator, e.g., activation functions, optimizers, etc. 
|experiment.py|Defines evaluation metric and creates experiment function.| **Probably No, unless** you want to change the evaluation metric.
|serving.py|Includes serving functions that accepts CSV, JSON, and TF Example instances.| **No**
|parameters.py|Includes the function to parse and initialize the arguments, as well as maintaining the hyper-parameters (hparam object).| **Probably No, unless** you want to change/add parameters (e.g. for feature engineering). 
|task.py|Entry point to the trainer, as it includes the main function that runs the experiment.| **No**
