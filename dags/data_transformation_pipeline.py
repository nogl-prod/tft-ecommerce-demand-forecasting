"""
Data Transformation Pipeline DAG

Transforms data from multiple sources (Shopify, Google Ads, Facebook, etc.)
Uses hybrid deployment pattern: base Docker image + dynamic code download.

Author: Data Team
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task, task_group
from airflow.models import Variable
import pendulum

from dags.utils.docker_helpers import create_transformation_task

# Default arguments
default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
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

@task_group(group_id="data_sources")
def data_transformation_tasks(client_name_task, code_version_task):
    """Group of parallel data transformation tasks."""
    
    transformations = [
        "shopify_products",
        "shopify_sales", 
        "facebook_ads",
        "google_ads",
        "google_analytics",
        "klaviyo",
        "plan_data"
    ]
    
    tasks = []
    for data_source in transformations:
        task = create_transformation_task(
            task_id=f"transform_{data_source}",
            data_source=data_source,
            client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
            code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
        )
        tasks.append(task)
    
    return tasks

with DAG(
    dag_id="data_transformation",
    default_args=default_args,
    description="Transform data from multiple sources",
    schedule_interval=None,  # Manual trigger
    start_date=pendulum.datetime(2021, 1, 1, tz="Europe/Berlin"),
    catchup=False,
    tags=["data", "transformation"],
    doc_md=__doc__,
) as dag:
    
    client_name = get_client_name()
    code_version = get_code_version()
    
    # Run all transformations in parallel
    transformation_tasks = data_transformation_tasks(client_name, code_version)
    
    # Consolidation task (runs after all transformations)
    consolidation = create_transformation_task(
        task_id="consolidate_data",
        data_source="consolidation",
        client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
        code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
    )
    
    # Set dependencies
    client_name >> code_version >> transformation_tasks >> consolidation

