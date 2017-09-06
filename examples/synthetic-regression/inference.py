from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

PROJECT='ksalama-gcp-playground'
MODEL_NAME='synth_regressor'
VERSION='v1'


def estimate(project, model_name, version, instances):

    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('ml', 'v1', credentials=credentials,
                discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

    request_data = {'instances': instances}

    model_url = 'projects/{}/models/{}/versions/{}'.format(project, model_name, version)
    response = api.projects().predict(body=request_data, name=model_url).execute()

    estimates = list(map(lambda item: item["scores"]
        ,response["predictions"]
    ))

    return estimates

#
# instances = [
#     {
#         'x': 1.3,
#         'y': -0.5,
#         'alpha': 'ax01',
#         'beta': 'bx02'
#     },
#
#     {
#         'x': -0.7,
#         'y': -0.5,
#         'alpha': 'ax02',
#         'beta': 'bx02'
#     }
# ]
#
# estimates = estimate(instances=instances
#                      ,project=PROJECT
#                      ,model_name=MODEL_NAME
#                      ,version=VERSION)
#
# print(estimates)