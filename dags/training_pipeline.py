"""
Training Pipeline DAG

Trains TFT models using Docker containers with code downloaded from Garage.
Uses hybrid deployment pattern: base Docker image + dynamic code download.

Author: ML Team
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import pendulum

from dags.utils.docker_helpers import create_training_task, get_garage_config

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': Variable.get('ALERT_EMAIL', default_var='ml-team@example.com'),
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=24),
    'sla': timedelta(hours=20),  # SLA: training should complete within 20 hours
}

@task(task_id="get_client_name")
def get_client_name(**context):
    """
    Extract client name from DAG run configuration.
    
    Returns:
        str: Client name from conf or default
    """
    return context['dag_run'].conf.get('client_name', 'wefriends')

@task(task_id="calculate_data_cutoff")
def calculate_data_cutoff(**context):
    """
    Calculate data cutoff date using execution date (idempotent).
    
    Returns:
        str: Data cutoff date in YYYY-MM-DD format
    """
    execution_date = context['execution_date']
    data_cutoff = execution_date - timedelta(days=1)
    return data_cutoff.strftime("%Y-%m-%d")

@task(task_id="get_code_version")
def get_code_version(**context):
    """
    Get code version to use. Can be 'latest' or specific Git SHA.
    
    Returns:
        str: Code version
    """
    return context['dag_run'].conf.get('code_version', 'latest')

with DAG(
    dag_id="tft_model_training",
    default_args=default_args,
    description="Train TFT models using Docker and MLflow with Garage storage",
    schedule_interval=None,  # Manual trigger
    start_date=pendulum.datetime(2022, 1, 1, tz="Europe/Berlin"),
    catchup=False,
    tags=["ml", "training", "tft", "production"],
    doc_md=__doc__,
    max_active_runs=1,  # Prevent concurrent runs
) as dag:
    
    # Get dynamic values
    client_name = get_client_name()
    data_cutoff = calculate_data_cutoff()
    code_version = get_code_version()
    
    # Training tasks group
    with TaskGroup("training_tasks", tooltip="Model Training Tasks") as training_group:
        
        # Top Seller Model Training (no-rolling)
        train_top_seller = create_training_task(
            task_id="train_top_seller",
            client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
            model_type="no-rolling",
            code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
        )
        
        # Long Tail Model Training (rolling)
        train_long_tail = create_training_task(
            task_id="train_long_tail",
            client_name="{{ ti.xcom_pull(task_ids='get_client_name') }}",
            model_type="rolling",
            code_version="{{ ti.xcom_pull(task_ids='get_code_version') }}",
        )
        
        # Run both training tasks in parallel
        [train_top_seller, train_long_tail]
    
    # Notification task (optional)
    @task(task_id="notify_completion")
    def notify_completion(**context):
        """Send notification on successful completion."""
        client = context['ti'].xcom_pull(task_ids='get_client_name')
        print(f"Training completed for {client}")
        # Add notification logic (email, Slack, etc.) here
    
    # Set dependencies
    client_name >> data_cutoff >> code_version >> training_group >> notify_completion()

