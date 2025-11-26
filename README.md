# TFT E-commerce Demand Forecasting

A comprehensive machine learning pipeline for e-commerce demand forecasting using Temporal Fusion Transformer (TFT) models. This system processes data from multiple sources (Shopify, Amazon, Google Ads, Facebook Ads, etc.) and generates accurate demand forecasts for e-commerce businesses.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides an end-to-end solution for e-commerce demand forecasting:

- **Data Collection**: Integrates with multiple data sources (Shopify, Amazon SP-API, Google Ads, Facebook Ads, Klaviyo, external data)
- **Data Processing**: Transforms and consolidates data from various sources
- **Model Training**: Trains Temporal Fusion Transformer models using PyTorch Forecasting
- **Inference**: Generates demand forecasts with quantile predictions
- **Evaluation**: Comprehensive metrics for model performance assessment
- **Demand Shaping**: Bundle demand shaping capabilities

## Architecture

The system follows a modular architecture with a hybrid deployment pattern:

```
Data Sources → Data Transformation → Data Consolidation → Model Training → Inference → Evaluation
```

### Deployment Architecture

The project uses a **hybrid deployment pattern** that combines:
- **Base Docker images**: Contain all dependencies (Python packages, system tools)
- **Dynamic code download**: Application code downloaded from Garage at runtime
- **CI/CD automation**: Automated builds and deployments via GitHub Actions

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

### Key Components

1. **Data Sourcing**: Collects data from various e-commerce platforms and external sources
2. **Data Transformation**: Transforms raw data into a format suitable for ML models
3. **Data Consolidation**: Merges data from multiple sources
4. **Training**: Trains TFT models on historical data with MLflow tracking
5. **Inference**: Generates forecasts for future demand
6. **Evaluation**: Assesses model performance using various metrics
7. **Orchestration**: Airflow DAGs manage the entire pipeline

## Features

- **Multi-source Data Integration**: Shopify, Amazon, Google Ads, Facebook Ads, Klaviyo, Weather, Holidays
- **Temporal Fusion Transformer Models**: State-of-the-art time series forecasting
- **Quantile Predictions**: Provides uncertainty estimates (q0-q6 quantiles)
- **Multi-client Support**: Handles multiple e-commerce clients
- **Demand Shaping**: Bundle demand shaping capabilities
- **Hybrid Deployment**: Base Docker images + dynamic code download from Garage
- **MLflow Integration**: Model versioning and artifact storage with Garage
- **Airflow Orchestration**: Automated workflows using DockerOperator
- **CI/CD Pipeline**: Automated builds and deployments via GitHub Actions

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- AWS account (for cloud deployment)
- Access to data sources (Shopify, Amazon, etc.)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nogl-prod/tft-ecommerce-demand-forecasting.git
cd tft-ecommerce-demand-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install inference-specific dependencies (if needed):
```bash
cd Inference
pip install -r requirements.txt
cd ..
```

4. Set up environment variables (see [Environment Variables](#environment-variables))

5. Configure database connection (see [Configuration](#configuration))

## Configuration

### Database Configuration

1. Copy the example database configuration:
```bash
cp databaseAWSRDS.ini.example databaseAWSRDS.ini
```

2. Edit `databaseAWSRDS.ini` with your database credentials:
```ini
[client_name]
host=your-database-host.rds.amazonaws.com
database=your_database
user=postgres
password=your_password
```

### OneDrive Configuration

1. Copy the example OneDrive configuration:
```bash
cp onedrive.ini.example onedrive.ini
```

2. Edit `onedrive.ini` with your Microsoft Azure AD credentials, OR use environment variables (recommended).

### Static Variables

The `static_variables.py` file uses environment variables for sensitive data. See `static_variables.py.example` for the structure.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

### Microsoft Azure AD / OneDrive
```bash
MSAL_CLIENT_ID=your_client_id
MSAL_CLIENT_SECRET=your_client_secret
MSAL_TENANT_ID=your_tenant_id
MSAL_SITE_ID=noglai.sharepoint.com
MSAL_REFRESH_TOKEN=your_refresh_token  # Optional
```

### AWS Configuration
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_SESSION_TOKEN=your_aws_session_token  # Optional
AWS_DEFAULT_REGION=eu-central-1
```

### MLflow Configuration
```bash
MLFLOW_TRACKING_URI=http://your-mlflow-server:port
MLFLOW_S3_ENDPOINT_URL=http://garage-endpoint:3900  # Garage endpoint for artifacts
AWS_ACCESS_KEY_ID=your_garage_access_key
AWS_SECRET_ACCESS_KEY=your_garage_secret_key
AWS_DEFAULT_REGION=garage
```

### External API Keys
```bash
VISUALCROSSING_API_KEY=your_visualcrossing_api_key  # For weather data
```

### Development Database (Optional)
```bash
DEV_DB_HOST=your-dev-database-host.rds.amazonaws.com
DEV_DB_PASSWORD=your_dev_database_password
```

**Note**: Never commit `.env` files or actual configuration files with credentials to version control. Use the `.example` files as templates.

## Usage

### Local Development

#### Training a Model

```bash
python Training/training_script.py \
    --client_name wefriends \
    --data_loader_batch_size 6 \
    --max_encoder_length 380 \
    --max_prediction_length 90 \
    --trainer_max_epochs 5
```

#### Running Inference

```bash
python Inference/03_Inference.py --client_name wefriends
```

#### Evaluation

```bash
python Evaluation/_result.py \
    --client_name wefriends \
    --schema forecasts \
    --variant_id 12345 \
    --y_true_col lineitems_quantity \
    --y_pred_col VOIDS_forecast_q3
```

### Production Deployment (Hybrid Pattern)

The project uses a hybrid deployment pattern with Airflow and Garage. See [Deployment Guide](docs/deployment.md) for detailed instructions.

#### Quick Start

1. **Configure GitHub Secrets and Variables** (see [Deployment Guide](docs/deployment.md))
2. **Configure Airflow Variables** (see [Deployment Guide](docs/deployment.md))
3. **Trigger DAGs** via Airflow UI with configuration:
   ```json
   {
     "client_name": "wefriends",
     "code_version": "latest"
   }
   ```

#### Available DAGs

- `tft_model_training` - Train TFT models
- `tft_model_inference` - Generate forecasts
- `data_transformation` - Transform data from multiple sources
- `data_consolidation` - Consolidate transformed data

## Project Structure

```
.
├── Data Sourcing/              # Data collection from various sources
│   └── amazon_spapi/           # Amazon Selling Partner API integration
├── Data Source Transformations/ # Data transformation scripts
├── Data Finalisation/          # Data consolidation and finalization
├── Training/                   # Model training scripts
├── Inference/                  # Inference and forecasting
│   ├── 03_Inference.py        # Main inference script
│   ├── src/                   # Source code
│   └── tests/                 # Unit tests
├── Evaluation/                 # Model evaluation metrics and scripts
├── Demand Shaping/            # Demand shaping functionality
├── dags/                      # Airflow DAGs (hybrid deployment pattern)
│   ├── training_pipeline.py   # Training DAG
│   ├── inference_pipeline.py  # Inference DAG
│   ├── data_transformation_pipeline.py  # Data transformation DAG
│   └── utils/                 # DAG utilities
│       ├── docker_helpers.py  # DockerOperator helper functions
│       └── monitoring.py      # Monitoring utilities
├── docker/                     # Docker configuration
│   ├── Dockerfile.base        # Base image with dependencies
│   ├── Dockerfile.training.base    # Training-specific image
│   ├── Dockerfile.inference.base   # Inference-specific image
│   ├── Dockerfile.transformation.base  # Transformation-specific image
│   └── entrypoint.sh          # Code download entrypoint script
├── docs/                      # Documentation
│   └── deployment.md          # Deployment guide
├── backup_dags/               # Legacy Airflow DAGs (SageMaker)
├── .github/
│   └── workflows/
│       ├── deploy.yml         # CI/CD workflow (hybrid pattern)
│       └── main.yml           # Legacy workflow
├── configAWSRDS.py            # Database configuration helper
├── Support_Functions.py       # Shared utility functions
├── static_variables.py        # Static variables (uses env vars)
├── defines.py                 # Event definitions and constants
├── requirements.txt           # Python dependencies
└── Dockerfile                 # Legacy Dockerfile (SageMaker)
```

### Key Files

- `Training/training_script.py`: Main training script for TFT models
- `Inference/03_Inference.py`: Main inference script for generating forecasts
- `Evaluation/_metrics.py`: Evaluation metrics (SMAPE, MAPE, RMSE, etc.)
- `Support_Functions.py`: Shared utility functions
- `configAWSRDS.py`: Database configuration parser

## Environment Variables

The project uses environment variables for sensitive configuration. Required variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `MSAL_CLIENT_ID` | Microsoft Azure AD client ID | Yes |
| `MSAL_CLIENT_SECRET` | Microsoft Azure AD client secret | Yes |
| `MSAL_TENANT_ID` | Microsoft Azure AD tenant ID | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key ID | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | Yes |
| `AWS_DEFAULT_REGION` | AWS region | No (default: eu-central-1) |
| `MSAL_REFRESH_TOKEN` | Optional refresh token | No |
| `MSAL_SITE_ID` | SharePoint site ID | No |
| `VISUALCROSSING_API_KEY` | VisualCrossing API key for weather data | Yes (if using weather data) |
| `DEV_DB_HOST` | Development database host | No (fallback for dev environment) |
| `DEV_DB_PASSWORD` | Development database password | No (fallback for dev environment) |

See `.env.example` for a complete template (note: `.env.example` may be blocked by gitignore, but the structure is documented in the template files).

## Deployment

### Hybrid Deployment Pattern

This project uses a hybrid deployment pattern that combines:
- **Base Docker images**: Contain all dependencies, rebuilt only when dependencies change
- **Dynamic code download**: Application code downloaded from Garage at runtime
- **CI/CD automation**: Automated builds and deployments via GitHub Actions

**Benefits:**
- Fast code updates (no image rebuilds needed)
- Stable base images (dependencies cached)
- Version control for code deployments
- Efficient resource usage

For detailed deployment instructions, see [Deployment Guide](docs/deployment.md).

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically:
1. Builds Docker base images when dependencies change
2. Packages and uploads code to Garage on every push
3. Validates deployments
4. Scans images for security vulnerabilities

### Airflow DAGs

DAGs are located in the `dags/` directory:
- Use DockerOperator for task execution
- Download code from Garage at runtime
- Support versioned code deployments
- Include comprehensive error handling and retries

## Security Best Practices

1. **Never commit credentials**: All sensitive data should be in environment variables or excluded files
2. **Use template files**: Copy `.example` files and fill in your values locally
3. **Review .gitignore**: Ensure sensitive files are excluded
4. **Rotate credentials**: Regularly update API keys and passwords
5. **Use Airflow Variables**: Store secrets in Airflow Variables (encrypted)
6. **Least privilege**: Use separate access keys for CI/CD and runtime
7. **Image scanning**: Docker images are automatically scanned for vulnerabilities
8. **Non-root containers**: All containers run as non-root users

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Write unit tests for new features

## License

[Add your license information here]

## Support

For issues and questions, please open an issue on GitHub.

## Acknowledgments

- PyTorch Forecasting for the TFT implementation
- AWS SageMaker for model training infrastructure
- All data source providers (Shopify, Amazon, Google, Facebook, etc.)

