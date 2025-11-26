"""
Data Consolidation Pipeline DAG

Consolidates transformed data from multiple sources.
Uses hybrid deployment pattern: base Docker image + dynamic code download.

Author: Data Team
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
import pendulum

from dags.utils.docker_helpers import create_transformation_task

# Default arguments
default_args = {
    'owner': 'data-team',
    'retries': 4,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=3),
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
    dag_id="data_consolidation",
    default_args=default_args,
    description="Consolidate data from multiple sources",
    schedule_interval=None,  # Manual trigger
    start_date=pendulum.datetime(2021, 1, 1, tz="Europe/Berlin"),
    catchup=False,
    tags=["data", "consolidation"],
    doc_md=__doc__,
) as dag:
    
    client_name = get_client_name()
    code_version = get_code_version()
    
    # Consolidation task
    consolidation = create_transformation_task(
        task_id="consolidate_data",
        data_source="consolidation",
        client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
        code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
    )
    
    # Set dependencies
    client_name >> code_version >> consolidation

