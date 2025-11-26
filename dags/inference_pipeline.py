"""
Inference Pipeline DAG

Generates demand forecasts using trained TFT models.
Uses hybrid deployment pattern: base Docker image + dynamic code download.

Author: ML Team
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
import pendulum

from dags.utils.docker_helpers import create_inference_task

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': Variable.get('ALERT_EMAIL', default_var='ml-team@example.com'),
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=2),
}

@task(task_id="get_client_name")
def get_client_name(**context):
    """
    Extract client name from DAG run configuration.
    
    Returns:
        str: Client name from conf or default
    """
    return context['dag_run'].conf.get('client_name', 'wefriends')

@task(task_id="get_code_version")
def get_code_version(**context):
    """
    Get code version to use. Can be 'latest' or specific Git SHA.
    
    Returns:
        str: Code version
    """
    return context['dag_run'].conf.get('code_version', 'latest')

with DAG(
    dag_id="tft_model_inference",
    default_args=default_args,
    description="Generate demand forecasts using trained TFT models",
    schedule_interval=None,  # Manual trigger
    start_date=pendulum.datetime(2022, 1, 1, tz="Europe/Berlin"),
    catchup=False,
    tags=["ml", "inference", "forecasting"],
    doc_md=__doc__,
) as dag:
    
    client_name = get_client_name()
    code_version = get_code_version()
    
    # Inference task
    inference_task = create_inference_task(
        task_id="generate_forecasts",
        client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
        code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
    )
    
    # Set dependencies
    client_name >> code_version >> inference_task

