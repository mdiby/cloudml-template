from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import numpy as np

PROJECT='ksalama-gcp-playground'
MODEL_NAME='synth_classifier'
VERSION='v1'

TARGET_LABELS = ['positive','negative']

def predict(project, model_name, version, instances):

    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials,
                discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

    request_data = {'instances': instances}

    model_url = 'projects/{}/models/{}/versions/{}'.format(project, model_name, version)
    response = api.projects().predict(body=request_data, name=model_url).execute()

    predictions = response["predictions"]

    probabilities = np.array(list(map(lambda item: item["probabilities"]
        ,predictions)
        )
    )

    class_indexes = np.argmax(probabilities, axis=1)

    classes = list(map(lambda index:TARGET_LABELS[index],class_indexes))

    return classes


instances = [
    {
        'x': 1.3,
        'y': -0.5,
        'alpha': 'ax01',
        'beta': 'bx02'
    },

    {
        'x': -0.7,
        'y': -0.5,
        'alpha': 'ax02',
        'beta': 'bx02'
    }
]

# predictions = predict(project=PROJECT,
#                       model_name=MODEL_NAME,
#                       version=VERSION,
#                       instances=instances)
# print(predictions)