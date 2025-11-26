"""
Reusable DockerOperator helper functions for Airflow DAGs.
Provides factory functions for creating consistent DockerOperator tasks.
"""
from datetime import timedelta
from typing import Dict, List, Optional, Any
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable


def get_garage_config() -> Dict[str, str]:
    """
    Get Garage configuration from Airflow Variables.
    
    Returns:
        Dictionary with Garage configuration
    """
    return {
        'GARAGE_ENDPOINT': Variable.get('GARAGE_ENDPOINT', default_var='http://192.168.29.163:3900'),
        'GARAGE_REGION': Variable.get('GARAGE_REGION', default_var='garage'),
        'GARAGE_CODE_BUCKET': Variable.get('GARAGE_CODE_BUCKET', default_var='code-repository'),
        'AWS_ACCESS_KEY_ID': Variable.get('GARAGE_AIRFLOW_ACCESS_KEY_ID', deserialize_json=False),
        'AWS_SECRET_ACCESS_KEY': Variable.get('GARAGE_AIRFLOW_SECRET_ACCESS_KEY', deserialize_json=False),
        'AWS_DEFAULT_REGION': Variable.get('GARAGE_REGION', default_var='garage'),
    }


def get_mlflow_config() -> Dict[str, str]:
    """
    Get MLflow configuration from Airflow Variables.
    
    Returns:
        Dictionary with MLflow configuration
    """
    garage_config = get_garage_config()
    return {
        **garage_config,
        'MLFLOW_TRACKING_URI': Variable.get('MLFLOW_TRACKING_URI', default_var='http://192.168.29.100:5000'),
        'MLFLOW_S3_ENDPOINT_URL': garage_config['GARAGE_ENDPOINT'],
    }


def create_training_task(
    task_id: str,
    client_name: str,
    model_type: str = 'no-rolling',
    code_version: str = 'latest',
    docker_image: Optional[str] = None,
    **kwargs
) -> DockerOperator:
    """
    Create a DockerOperator task for model training.
    
    Args:
        task_id: Unique task identifier
        client_name: Client name for training
        model_type: Type of model ('no-rolling' or 'rolling')
        code_version: Code version to download ('latest' or Git SHA)
        docker_image: Docker image to use (defaults to training base image)
        **kwargs: Additional arguments for DockerOperator
    
    Returns:
        Configured DockerOperator instance
    """
    garage_config = get_garage_config()
    mlflow_config = get_mlflow_config()
    
    # Determine target column based on model type
    target_column = 'TVUR_shopify_lineitems_quantity' if model_type == 'no-rolling' else 'TVUR_rolling7days_lineitems_quantity'
    
    # Default image if not provided
    if not docker_image:
        registry = Variable.get('DOCKER_REGISTRY_URL', default_var='localhost:5000')
        image_prefix = Variable.get('IMAGE_PREFIX', default_var='tft-demand-forecasting')
        docker_image = f"{registry}/{image_prefix}-training:latest"
    
    # Build environment variables - CODE_VERSION will be templated at runtime
    env_vars = {
        **mlflow_config,
        'CODE_VERSION': '{{ ti.xcom_pull(task_ids="get_code_version") }}' if code_version == 'latest' else code_version,
        'CLIENT_NAME': '{{ ti.xcom_pull(task_ids="get_client_name") }}',
        'MODEL_TYPE': model_type,
        'TARGET': target_column,
    }
    
    # Add training-specific parameters from Variables if available
    training_params = {
        'MAX_EPOCHS': Variable.get('TRAINING_MAX_EPOCHS', default_var='5'),
        'BATCH_SIZE': Variable.get('TRAINING_BATCH_SIZE', default_var='10'),
        'LEARNING_RATE': Variable.get('TRAINING_LEARNING_RATE', default_var='0.01'),
    }
    env_vars.update(training_params)
    
    # Build command
    command = [
        'python', 'Training/training_script.py',
        '--client_name', client_name,
        '--target', target_column,
    ]
    
    # Default DockerOperator arguments
    default_args = {
        'image': docker_image,
        'api_version': 'auto',
        'auto_remove': True,
        'docker_url': 'unix://var/run/docker.sock',
        'network_mode': 'bridge',
        'environment': env_vars,
        'command': command,
        'mount_tmp_dir': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=24),
    }
    
    # Merge with user-provided kwargs
    default_args.update(kwargs)
    
    return DockerOperator(
        task_id=task_id,
        **default_args
    )


def create_inference_task(
    task_id: str,
    client_name: str,
    code_version: str = 'latest',
    docker_image: Optional[str] = None,
    **kwargs
) -> DockerOperator:
    """
    Create a DockerOperator task for model inference.
    
    Args:
        task_id: Unique task identifier
        client_name: Client name for inference
        code_version: Code version to download ('latest' or Git SHA)
        docker_image: Docker image to use (defaults to inference base image)
        **kwargs: Additional arguments for DockerOperator
    
    Returns:
        Configured DockerOperator instance
    """
    garage_config = get_garage_config()
    mlflow_config = get_mlflow_config()
    
    # Default image if not provided
    if not docker_image:
        registry = Variable.get('DOCKER_REGISTRY_URL', default_var='localhost:5000')
        image_prefix = Variable.get('IMAGE_PREFIX', default_var='tft-demand-forecasting')
        docker_image = f"{registry}/{image_prefix}-inference:latest"
    
    # Build environment variables - CODE_VERSION will be templated at runtime
    env_vars = {
        **mlflow_config,
        'CODE_VERSION': '{{ ti.xcom_pull(task_ids="get_code_version") }}' if code_version == 'latest' else code_version,
        'CLIENT_NAME': '{{ ti.xcom_pull(task_ids="get_client_name") }}',
    }
    
    # Build command - use templated values for runtime resolution
    command = [
        'python', 'Inference/03_Inference.py',
        '--client_name', '{{ ti.xcom_pull(task_ids="get_client_name") }}',
    ]
    
    # Default DockerOperator arguments
    default_args = {
        'image': docker_image,
        'api_version': 'auto',
        'auto_remove': True,
        'docker_url': 'unix://var/run/docker.sock',
        'network_mode': 'bridge',
        'environment': env_vars,
        'command': command,
        'mount_tmp_dir': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=10),
        'execution_timeout': timedelta(hours=2),
    }
    
    # Merge with user-provided kwargs
    default_args.update(kwargs)
    
    return DockerOperator(
        task_id=task_id,
        **default_args
    )


def create_transformation_task(
    task_id: str,
    data_source: str,
    client_name: str,
    code_version: str = 'latest',
    docker_image: Optional[str] = None,
    **kwargs
) -> DockerOperator:
    """
    Create a DockerOperator task for data transformation.
    
    Args:
        task_id: Unique task identifier
        data_source: Name of the data source to transform
        client_name: Client name
        code_version: Code version to download ('latest' or Git SHA)
        docker_image: Docker image to use (defaults to transformation base image)
        **kwargs: Additional arguments for DockerOperator
    
    Returns:
        Configured DockerOperator instance
    """
    garage_config = get_garage_config()
    
    # Default image if not provided
    if not docker_image:
        registry = Variable.get('DOCKER_REGISTRY_URL', default_var='localhost:5000')
        image_prefix = Variable.get('IMAGE_PREFIX', default_var='tft-demand-forecasting')
        docker_image = f"{registry}/{image_prefix}-transformation:latest"
    
    # Map data source to script path
    script_mapping = {
        'shopify_products': 'Data Source Transformations/AWS01_Shopify_Products_Transform_final.py',
        'shopify_sales': 'Data Source Transformations/AWS02_Shopify_Sales_Transform_final.py',
        'facebook_ads': 'Data Source Transformations/AWS02_Facebook_Ads_final.py',
        'google_ads': 'Data Source Transformations/AWS02_GoogleAds_Transform_final.py',
        'google_analytics': 'Data Source Transformations/AWS02_GoogleAnalytics_Transform_final.py',
        'klaviyo': 'Data Source Transformations/AWS02_Klaviyo_preprocessing_final_NEW.py',
        'plan_data': 'Data Source Transformations/AWS02_Plan_Data_Preprocessing_final.py',
        'consolidation': 'Data Finalisation/02_Consolidation.py',
    }
    
    script_path = script_mapping.get(data_source, f'Data Source Transformations/AWS02_{data_source}_final.py')
    
    # Build environment variables - CODE_VERSION will be templated at runtime
    env_vars = {
        **garage_config,
        'CODE_VERSION': '{{ ti.xcom_pull(task_ids="get_code_version") }}' if code_version == 'latest' else code_version,
        'CLIENT_NAME': '{{ ti.xcom_pull(task_ids="get_client_name") }}',
        'DATA_SOURCE': data_source,
    }
    
    # Build command - use templated values for runtime resolution
    command = [
        'python', script_path,
        '--client_name', '{{ ti.xcom_pull(task_ids="get_client_name") }}',
    ]
    
    # Default DockerOperator arguments
    default_args = {
        'image': docker_image,
        'api_version': 'auto',
        'auto_remove': True,
        'docker_url': 'unix://var/run/docker.sock',
        'network_mode': 'bridge',
        'environment': env_vars,
        'command': command,
        'mount_tmp_dir': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=1),
    }
    
    # Merge with user-provided kwargs
    default_args.update(kwargs)
    
    return DockerOperator(
        task_id=task_id,
        **default_args
    )

