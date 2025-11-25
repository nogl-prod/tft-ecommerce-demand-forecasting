# Inference Module

This module contains scripts and utilities for generating demand forecasts using trained Temporal Fusion Transformer (TFT) models.

## Overview

The inference module generates demand forecasts for e-commerce products using pre-trained TFT models. It supports:

- **Top Seller Models**: For high-volume products
- **Long Tail Models**: For products with less historical data
- **Extreme Long Tail**: Simple forecasting for products with minimal data
- **Quantile Predictions**: Provides uncertainty estimates (q0-q6)

## Main Scripts

### 03_Inference.py

Main inference script that generates forecasts for a specified client.

**Usage:**
```bash
python Inference/03_Inference.py --client_name wefriends
```

**Arguments:**
- `--client_name` (required): Name of the client (must match a section in databaseAWSRDS.ini)

**What it does:**
1. Loads the best trained models (top seller and long tail)
2. Prepares data for inference
3. Generates forecasts for the next 90 days
4. Adjusts forecasts based on marketing budgets and trends
5. Exports results to PostgreSQL database

**Output Tables:**
- `forecasts.forecasts_28days`: Detailed forecasts with quantiles
- `forecasts.forecasts_28days_category`: Category-level aggregated forecasts
- `forecasts.forecasts_28days_incl_history`: Forecasts combined with historical data

### Other Inference Scripts

- `03_InferenceV2.py`: Alternative inference implementation
- `CrossClient_Inference.py`: Cross-client inference capabilities
- `Backtesting_Inference.py`: Backtesting functionality for model validation

## Configuration

### Model Paths

Models are expected to be located at:
- Top Seller: `/opt/ml/processing/input2/trained_models/tft/no-rolling/{client_name}/`
- Long Tail: `/opt/ml/processing/input2/trained_models/tft/rolling/{client_name}/`

### Data Inputs

The script expects consolidated data files:
- `{date}_{client_name}_TopSeller_consolidated_cutoff.csv`
- `{date}_{client_name}_LongTail_consolidated_cutoff.csv`
- `{date}_{client_name}_Kicked_consolidated_cutoff.csv`

### Database Configuration

Ensure `databaseAWSRDS.ini` is configured with the correct database credentials for your client.

## Key Functions

### prepare_data_for_inference()

Prepares data for inference by:
- Renaming features (replacing "." with "_")
- Categorizing features (static, time-varying, known/unknown)
- Setting up encoder/decoder lengths
- Handling special events (holidays, sales events)

### get_results()

Generates predictions from the model:
- Creates encoder and decoder data
- Runs model prediction
- Formats results with quantiles (q0-q6)
- Returns DataFrame with forecasts

### load_TFT()

Loads a trained TFT model from checkpoint:
- Loads model from file path
- Sets device to CPU
- Disables randomness (eval mode)
- Returns model ready for inference

## Forecast Adjustments

The inference process includes several adjustments:

1. **Market Development**: Applies market change factor (default: 0.65)
2. **Marketing Budget Ratios**: Adjusts based on planned marketing spend vs. last year
3. **Product Ratios**: Accounts for changes in product portfolio
4. **Trend Adjustments**: Uses statistical trend analysis from historical data

## Output Format

Forecasts include:
- **Quantile Predictions**: q0 (min), q1, q2 (median), q3, q4, q5, q6 (max)
- **Revenue Calculations**: Forecast quantity × RRP
- **Category Aggregations**: Weekly category-level summaries
- **ROAS Metrics**: Return on ad spend calculations

## Testing

Run unit tests:
```bash
cd Inference
pytest tests/
```

## Requirements

See `requirements.txt` for Python dependencies. Key libraries:
- pytorch-forecasting
- pytorch-lightning
- pandas
- numpy
- sqlalchemy
- boto3 (for S3 model access)

## Troubleshooting

### Model Not Found
- Ensure models are trained and available at expected paths
- Check that model files follow naming convention: `{date}_tft-{epoch}-{val_loss}.ckpt`

### Database Connection Issues
- Verify `databaseAWSRDS.ini` configuration
- Check network connectivity to RDS instance
- Ensure database user has necessary permissions

### Data Format Issues
- Verify input CSV files have required columns
- Check feature naming conventions (SC_, SR_, TVKC_, TVKR_, TVUC_, TVUR_ prefixes)
- Ensure date formats are consistent

## Related Documentation

- Main project README: `../README.md`
- Evaluation module: `../Evaluation/README.md`
- Training module: `../Training/training_script.py`

