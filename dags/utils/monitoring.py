"""
Monitoring and logging utilities for Airflow DAGs.
Provides structured logging and metrics collection for hybrid deployment pattern.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from airflow.models import Variable


def setup_task_logging(task_id: str, component: str) -> logging.Logger:
    """
    Set up structured logging for a task.
    
    Args:
        task_id: Task identifier
        component: Component type (training, inference, transformation)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"{component}.{task_id}")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Add component to logger context
    logger = logging.LoggerAdapter(logger, {'component': component})
    
    return logger


def log_code_download_start(logger: logging.Logger, code_version: str, bucket: str):
    """Log code download start."""
    logger.info(
        f"Starting code download",
        extra={
            'code_version': code_version,
            'bucket': bucket,
            'event': 'code_download_start',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_code_download_complete(logger: logging.Logger, code_version: str, duration: float, size_mb: float):
    """Log code download completion."""
    logger.info(
        f"Code download completed",
        extra={
            'code_version': code_version,
            'duration_seconds': duration,
            'size_mb': size_mb,
            'event': 'code_download_complete',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_code_download_failure(logger: logging.Logger, code_version: str, error: str, retry_count: int):
    """Log code download failure."""
    logger.error(
        f"Code download failed: {error}",
        extra={
            'code_version': code_version,
            'error': str(error),
            'retry_count': retry_count,
            'event': 'code_download_failure',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_task_start(logger: logging.Logger, task_id: str, component: str, client_name: str):
    """Log task start."""
    logger.info(
        f"Task started: {task_id}",
        extra={
            'task_id': task_id,
            'component': component,
            'client_name': client_name,
            'event': 'task_start',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_task_complete(logger: logging.Logger, task_id: str, duration: float, status: str = 'success'):
    """Log task completion."""
    logger.info(
        f"Task completed: {task_id}",
        extra={
            'task_id': task_id,
            'duration_seconds': duration,
            'status': status,
            'event': 'task_complete',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_task_failure(logger: logging.Logger, task_id: str, error: str):
    """Log task failure."""
    logger.error(
        f"Task failed: {task_id}",
        extra={
            'task_id': task_id,
            'error': str(error),
            'event': 'task_failure',
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def get_metrics_config() -> Dict[str, Any]:
    """
    Get metrics configuration from Airflow Variables.
    
    Returns:
        Dictionary with metrics configuration
    """
    return {
        'enable_metrics': Variable.get('ENABLE_METRICS', default_var='true').lower() == 'true',
        'metrics_endpoint': Variable.get('METRICS_ENDPOINT', default_var=''),
        'metrics_namespace': Variable.get('METRICS_NAMESPACE', default_var='tft_pipeline'),
    }

