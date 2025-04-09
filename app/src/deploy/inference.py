import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.utils.utils import get_sagemaker_client

load_dotenv()

region =  os.getenv('AWS_REGION') 
role = os.getenv('AWS_SAGEMAKER_ROLE') 
model_name = os.getenv('AWS_MODEL_NAME')
endpoint_name = os.getenv('AWS_ENDPOINT_NAME')

client = get_sagemaker_client()

def create_endpoint_config(config_name, model_name, instance_type, initial_instance_count):
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': initial_instance_count,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0
            }
        ]
    )

    print(f"Endpoint configuration created successfully")
    return endpoint_config_response

def create_endpoint(name, config):
    response = client.create_endpoint(
        EndpointName=name,
        EndpointConfigName=config
    )
    return response

config_name = f"{model_name}-config"
final_model_name = f"{model_name}-v1"

if __name__ == '__main__':
    # Create endpoint configuration
    endpoint_config_response = create_endpoint_config(
        config_name=config_name,
        model_name=final_model_name,
        instance_type='ml.t2.medium', # cheapest cost
        initial_instance_count=1
    )

    # Deploy
    response = create_endpoint(
        name=endpoint_name,
        config=config_name
    )