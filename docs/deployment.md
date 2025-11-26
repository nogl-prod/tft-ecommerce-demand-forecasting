# Hybrid Deployment Pattern - Deployment Guide

## Overview

This project uses a hybrid deployment pattern that combines:
- **Base Docker images**: Contain all dependencies (Python packages, system tools)
- **Dynamic code download**: Application code downloaded from Garage at runtime
- **CI/CD automation**: Automated builds and deployments via GitHub Actions

## Architecture

```
┌─────────────────┐
│  GitHub Push    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     GitHub Actions CI/CD            │
│  ┌───────────────────────────────┐  │
│  │ Build Base Docker Images      │  │
│  │ (only when deps change)       │  │
│  └───────────┬───────────────────┘  │
│              │                       │
│  ┌───────────▼───────────────────┐  │
│  │ Package & Upload Code         │  │
│  │ to Garage (always)            │  │
│  └───────────┬───────────────────┘  │
└──────────────┼───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         Garage Storage              │
│  - code-repository/                │
│  - code-<SHA>.tar.gz              │
│  - code-latest.tar.gz              │
└──────────────┬───────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Airflow DAG Execution         │
│  ┌───────────────────────────────┐  │
│  │ DockerOperator                │  │
│  │ 1. Pull base image            │  │
│  │ 2. Download code from Garage │  │
│  │ 3. Execute training/inference │  │
│  └───────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Prerequisites

### 1. Garage Setup

- Garage server running and accessible
- Buckets created:
  - `code-repository` - For code archives
  - `training-data` - For training datasets
  - `model-artifacts` - For trained models
  - `data-pipeline` - For intermediate data
  - `mlflow-artifacts` - For MLflow artifacts
- Access keys configured:
  - CI/CD key (write to code-repository)
  - Airflow runtime key (read code, read/write data)

### 2. Container Registry

Set up a Docker registry (self-hosted recommended):
- Docker Registry
- Harbor
- Or use Docker Hub

### 3. Airflow Configuration

- Airflow instance with DockerOperator support
- Docker daemon accessible from Airflow workers
- Airflow Variables configured (see Configuration section)

### 4. MLflow Server

- MLflow tracking server running
- Configured to use Garage for artifact storage

## Configuration

### GitHub Secrets

Add these secrets in GitHub (Settings → Secrets and variables → Actions):

**Secrets:**
- `GARAGE_CICD_ACCESS_KEY_ID` - CI/CD access key
- `GARAGE_CICD_SECRET_ACCESS_KEY` - CI/CD secret key
- `GARAGE_ENDPOINT` - Garage endpoint URL (e.g., `https://garage.nogl.tech`)
- `GARAGE_REGION` - Garage region (usually `garage`)
- `DOCKER_REGISTRY_USERNAME` - Registry username (if using private registry)
- `DOCKER_REGISTRY_PASSWORD` - Registry password (if using private registry)

**Variables:**
- `GARAGE_CODE_BUCKET` - `code-repository`
- `GARAGE_TRAINING_DATA_BUCKET` - `training-data`
- `GARAGE_MODEL_BUCKET` - `model-artifacts`
- `GARAGE_DATA_PIPELINE_BUCKET` - `data-pipeline`
- `GARAGE_MLFLOW_BUCKET` - `mlflow-artifacts`
- `DOCKER_REGISTRY_URL` - Registry URL (e.g., `localhost:5000` or `registry.example.com`)
- `IMAGE_PREFIX` - `tft-demand-forecasting`

### Airflow Variables

Add these variables in Airflow UI (Admin → Variables):

**Garage Configuration:**
- `GARAGE_ENDPOINT` - `http://192.168.29.163:3900` (or Cloudflare tunnel URL)
- `GARAGE_REGION` - `garage`
- `GARAGE_CODE_BUCKET` - `code-repository`
- `GARAGE_TRAINING_DATA_BUCKET` - `training-data`
- `GARAGE_MODEL_BUCKET` - `model-artifacts`
- `GARAGE_DATA_PIPELINE_BUCKET` - `data-pipeline`
- `GARAGE_MLFLOW_BUCKET` - `mlflow-artifacts`

**Garage Credentials (Secrets):**
- `GARAGE_AIRFLOW_ACCESS_KEY_ID` - Airflow runtime access key
- `GARAGE_AIRFLOW_SECRET_ACCESS_KEY` - Airflow runtime secret key

**MLflow Configuration:**
- `MLFLOW_TRACKING_URI` - `http://192.168.29.100:5000`
- `MLFLOW_S3_ENDPOINT_URL` - Same as `GARAGE_ENDPOINT`

**Docker Registry:**
- `DOCKER_REGISTRY_URL` - Registry URL
- `IMAGE_PREFIX` - `tft-demand-forecasting`

**Training Parameters (Optional):**
- `TRAINING_MAX_EPOCHS` - `5`
- `TRAINING_BATCH_SIZE` - `10`
- `TRAINING_LEARNING_RATE` - `0.01`

**Monitoring:**
- `ALERT_EMAIL` - Email for failure notifications
- `ENABLE_METRICS` - `true` or `false`

## Deployment Process

### Step 1: Initial Setup

1. **Set up Garage buckets and keys** (already done)
2. **Configure GitHub Secrets and Variables**
3. **Configure Airflow Variables**
4. **Set up container registry**

### Step 2: Build Base Images

Base images are built automatically when:
- `requirements.txt` changes
- `docker/` directory changes
- Manual trigger via `workflow_dispatch`

To manually trigger image build:

```bash
# Via GitHub UI
Actions → Deploy to Garage - Hybrid Pattern → Run workflow → 
  Check "Force rebuild Docker images" → Run workflow
```

### Step 3: Deploy Code

Code is automatically packaged and uploaded to Garage on every push to `main` branch.

The workflow:
1. Creates compressed archive of code
2. Uploads to Garage with Git SHA tag
3. Updates `code-latest.tar.gz` pointer
4. Validates upload

### Step 4: Run Airflow DAGs

1. **Trigger DAG manually** in Airflow UI
2. **Provide configuration**:
   ```json
   {
     "client_name": "wefriends",
     "code_version": "latest"
   }
   ```
3. **Monitor execution** in Airflow UI

## Usage Examples

### Training Pipeline

```python
# Trigger via Airflow UI with config:
{
  "client_name": "wefriends",
  "code_version": "latest"
}
```

The DAG will:
1. Download code from Garage
2. Run training for both model types (top seller and long tail)
3. Log models to MLflow (stored in Garage)
4. Save checkpoints

### Inference Pipeline

```python
# Trigger via Airflow UI with config:
{
  "client_name": "wefriends",
  "code_version": "latest"
}
```

The DAG will:
1. Download code from Garage
2. Load trained models from MLflow
3. Generate forecasts
4. Export to database

### Data Transformation Pipeline

```python
# Trigger via Airflow UI with config:
{
  "client_name": "wefriends",
  "code_version": "latest"
}
```

The DAG will:
1. Download code from Garage
2. Run all transformations in parallel
3. Consolidate results

## Troubleshooting

### Code Download Fails

**Symptoms:**
- Task fails with "Failed to download code"
- 403 Forbidden errors

**Solutions:**
1. Check Airflow Variables for Garage credentials
2. Verify Garage endpoint is accessible
3. Check bucket permissions for Airflow key
4. Review entrypoint.sh logs in task logs

### Docker Image Pull Fails

**Symptoms:**
- "Image not found" errors
- Registry authentication failures

**Solutions:**
1. Verify registry URL in Airflow Variables
2. Check if base images were built (check GitHub Actions)
3. Ensure registry is accessible from Airflow workers
4. For self-hosted registry, check Docker daemon configuration

### MLflow Artifact Upload Fails

**Symptoms:**
- Models not appearing in MLflow UI
- S3 errors in training logs

**Solutions:**
1. Verify MLflow tracking URI
2. Check Garage endpoint configuration
3. Ensure MLflow key has write access to mlflow-artifacts bucket
4. Check MLflow server logs

### Performance Issues

**Symptoms:**
- Slow code downloads
- Long task startup times

**Solutions:**
1. Check Garage server performance
2. Verify network connectivity
3. Consider caching code in persistent volume
4. Monitor download times in logs

## Rollback Procedures

### Rollback Code Version

1. **Find previous Git SHA** from GitHub commits
2. **Trigger DAG with specific version**:
   ```json
   {
     "client_name": "wefriends",
     "code_version": "<previous_git_sha>"
   }
   ```

### Rollback Docker Image

1. **Find previous image tag** from registry
2. **Update Airflow Variable** `IMAGE_TAG` to previous version
3. **Or specify image directly** in DAG configuration

## Monitoring

### Code Download Metrics

Monitor in Airflow task logs:
- Download duration
- Archive size
- Success/failure rate

### Task Execution Metrics

Monitor in Airflow:
- Task duration
- Success rate
- Resource usage

### MLflow Metrics

Monitor in MLflow UI:
- Model versions
- Training metrics
- Artifact storage usage

## Best Practices

1. **Always test in staging** before production
2. **Use specific code versions** for production (not `latest`)
3. **Monitor base image builds** - rebuild only when needed
4. **Rotate access keys** regularly
5. **Keep code archives small** - exclude unnecessary files
6. **Use caching** for code downloads when possible
7. **Monitor Garage storage** usage
8. **Set up alerts** for failures

## Security Considerations

1. **Never commit secrets** to repository
2. **Use Airflow Variables** for sensitive data
3. **Rotate access keys** regularly
4. **Limit key permissions** (least privilege)
5. **Scan Docker images** for vulnerabilities
6. **Use non-root users** in containers
7. **Enable audit logging** for Garage access

## Support

For issues or questions:
1. Check Airflow task logs
2. Review GitHub Actions workflow logs
3. Check Garage server logs
4. Review MLflow server logs
5. Contact ML/Data team

