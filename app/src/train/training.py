import sys
import os
import boto3
import sagemaker
from sagemaker import Session, get_execution_role
from sagemaker.sklearn import SKLearn
from sagemaker import image_uris
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from app.utils.utils import get_sagemaker_client

load_dotenv()

# debug
print("AWS Access Key ID:", os.getenv('AWS_ACCESS_KEY_ID_ML'))
print("AWS Secret Access Key:", os.getenv('AWS_SECRET_ACCESS_KEY_ML'))
print("AWS Region:", os.getenv('AWS_REGION'))
print("SAGEMAKER ROLE:", os.getenv('AWS_SAGEMAKER_ROLE'))

bucket_name = os.getenv('AWS_BUCKET_NAME')
prefix = os.getenv('AWS_BUCKET_PREFIX')
region =  os.getenv('AWS_REGION') 
role = os.getenv('AWS_SAGEMAKER_ROLE') 
model_name = os.getenv('AWS_MODEL_NAME')

boto3.setup_default_session(region_name=region)
sagemaker_session = Session(boto3.session.Session(region_name=region))

s3_cleaned_path = f"s3://{bucket_name}/{prefix}/insurance_data_cleaned.csv"
output_path = f"s3://{bucket_name}/models"

latest_training_job = None

# Training as framework
def training():
    global latest_training_job

    # Set the source_dir to the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    sklearn_estimator = SKLearn(
        entry_point="train.py",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.0-1",
        py_version="py3",
        source_dir=current_dir,
        output_path=output_path,
        dependecies=['requirements.txt'],
        sagemaker_session=sagemaker_session, # Important when run outside notebook
        hyperparameters={
            'max_depth': 5,
            'min_samples_split': 2,
            'random_state': 42
        }
    )

    sklearn_estimator.fit({'train': s3_cleaned_path})

    latest_training_job = sklearn_estimator.latest_training_job.name
    return latest_training_job

def register(model_name, model_uri):
    client = get_sagemaker_client()

    sklearn_image_uri = image_uris.retrieve(
            framework='sklearn',
            region=region,
            version='0.23-1',
            py_version='py3',
            instance_type='ml.m5.large'
        )
    
    create_model_response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'Image': sklearn_image_uri,
            'ModelDataUrl': model_uri,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_REGION': region,
                'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{bucket_name}/{latest_training_job}/source/sourcedir.tar.gz',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
            }
        },
    )

    print("Model registered!")
    return create_model_response

if __name__ == '__main__':
    estimator = training()

    final_model_name = f"{model_name}-v1"
    model_uri = f"s3://{bucket_name}/models/{latest_training_job}/output/model.tar.gz"

    model_response = register(
        model_name=final_model_name,
        model_uri=model_uri
    )