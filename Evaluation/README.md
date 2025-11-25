# Evaluation Module

This module provides comprehensive evaluation metrics and tools for assessing the performance of demand forecasting models.

## Overview

The evaluation module contains a collection of classes for various evaluation metrics that can be used for assessing the performance of predictive models. All of the classes inherit from the AbstractMetric class, which defines an interface for the __call__() method that takes in predicted and true values, along with optional weights, and returns the corresponding metric value.

## Metrics Overview

Here is an overview of the metrics that are defined in the script:

1. **SMAPE** - Symmetric Mean Absolute Percentage Error

SMAPE: Symmetric Mean Absolute Percentage Error - this metric can be useful in demand forecasting or shaping to measure the accuracy of your predictions in percentage terms. For example, if your predicted demand is 10% lower than the true demand, SMAPE will show a 10% error. This can help you identify which products or regions have the highest or lowest errors and adjust your forecasting or shaping strategy accordingly.

2. **MSE and RMSE** - Mean Squared Error and Root Mean Squared Error

MSE and RMSE: Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are commonly used metrics in demand forecasting or shaping to measure the average difference between predicted and true demand values. They are useful when you want to penalize larger errors more heavily than smaller errors as they put more weight on the extreme outliers. They can help you identify which products or regions have larger errors and may need more attention in your forecasting or shaping strategy.

3. **MAPE** - Mean Absolute Percentage Error

MAPE: Mean Absolute Percentage Error - this metric can be a useful tool for demand-forecasting or shaping teams to compare the accuracy of different models, especially when the magnitude of demand values may vary significantly across products or regions. MAPE has the advantage of measuring errors in percentage terms, enabling you to compare different models on a uniform scale.

4. **WMSMAPE** - Weighted Mean Symmetric Mean Absolute Percentage Error

WMSMAPE: Weighted Mean Symmetric Mean Absolute Percentage Error - similar to SMAPE, but supports the weighting of samples to give more importance to specific samples. This can be especially useful in demand shaping scenarios or forecasting where some products or regions contribute more to your business's bottom line, and hence deserve more attention in forecasting or shaping strategy.

5. **Accuracy** - Accuracy in percentage

Accuracy: Accuracy in percentage is useful in demand shaping scenarios, especially when predicting or classifying demand trends. The accuracy metric tells you the percentage of correctly predicted events or signals based on the total events. For example, if 80% of your predicted demand upticks are actually observed as upticks, the accuracy will be calculated as 80%.

6. **DirectionalSymmetry** - Directional Symmetry

DirectionalSymmetry: Directional Symmetry - this metric can be useful in measuring how closely your demand predictions are tracking the changes in observed trends. High directional symmetry indicates that the prediction values are trending in the same direction as the actual demand values. In contrast, low directional symmetry indicates prediction values and actual demand values are diverging - this suggests that your demand predictions are not accurately tracking changes in the market trends.
The METRICS dictionary maps the names of the metrics to their corresponding class objects, and can be used with the Evaluator class in the Evaluator.py script to evaluate the performance of different models on a given dataset.

Note that some of these metrics (such as R-squared and RMSE) are not suitable for evaluating the performance of models that make individual predictions for each row in a dataset, and require a global aggregate to make sense. Similarly, some of the metrics (such as Accuracy and MAPE) do not allow for weighted samples, and may not be suitable for datasets where certain samples are more important than others.

## Usage

To use these metrics, you can import the relevant classes from the script into your Python code, and pass in the predicted and true values (and weights if applicable) to the corresponding __call__() method of the metric. The metric value returned by the __call__() method can then be used to evaluate the performance of your model.

_result.py
The _result.py script contains a set of functions that are used to evaluate and process data for model selection and evaluation. The script is designed to be run from the command line or as part of a larger pipeline.

Requirements
The _result.py script requires several libraries to be installed, including:

os
sys
logging
pathlib
pickle
traceback
time
multiprocessing
argparse
re
numpy
pandas
sqlalchemy
sklearn
yaml
Usage
The _result.py script can be executed from the command line using the following arguments:

--client_name: The name of the client (section in the config file).
--schema: The name of the schema in the database.
--y_true_col: The name of the column containing the true values. (default='lineitems_quantity')
--y_pred_col: The name of the column containing the predicted values. (default='NOGL_forecast_q3')
--variant_id: Variant ID to filter the data. (required)
--metric_keys: A list of metric key suffixes to use for evaluation.
Functions
The _result.py script contains several helpful functions:

config(section, filename)
Reads configuration data from a YAML file.

section: The section in the config file to read.
filename: The name of the YAML config file.
model_evaluation(data, y_true_col, y_pred_col, metrics_suffixes, per_row)
Evaluates a model using specified metrics.

data: The input DataFrame containing true and predicted values.
y_true_col: The name of the column containing the true values.
y_pred_col: The name of the column containing the predicted values.
metrics_suffixes: A list of metric suffixes to use for evaluation.
per_row: If True, compute the metric for each row in the DataFrame. If False, compute the metric for the entire DataFrame.
fetch_data(engine, y_true_col, y_pred_col, variant_id, schema_name, table_name_pattern)
Fetches data from the database based on the provided schema and table name pattern.

engine: SQLAlchemy engine object for connecting to the database.
y_true_col: The name of the column containing the true values.
y_pred_col: The name of the column containing the predicted values.
variant_id: Variant ID to filter the data.
schema_name: The name of the schema in the database. (default='forecasts')
table_name_pattern: The pattern to match table names in the schema. (default='forecasts_28days_incl_history')
process_dataset(item, y_true_col, metric_keys)
Transforms and evaluates a dataset.

item: A tuple containing the table name and the dataset.
y_true_col: The name of the column containing the true values.
metric_keys: A list of metric keys to use for evaluation.
main(client_name, schema, y_true_col, y_pred_col, variant_id, metric_keys)
The main function that fetches data, processes it, and saves the results in a database.

client_name: The name of the client (section in the config file).
schema: The name of the schema in the database.
y_true_col: The name of the column containing the true values.
y_pred_col: The name of the column containing the predicted values.
variant_id: Variant ID to filter the data.
metric_keys: A list of metric keys to use for evaluation (optional).
The main function fetches data from the database, processes it, evaluates it using the model_evaluation function, and saves the results in the specified schema in the Postgres database. The results are saved in two formats: a CSV file and a SQL table. The script takes in several command line arguments that are used to configure the evaluation.

## Usage

### Running Evaluation

```bash
python Evaluation/_result.py \
    --client_name wefriends \
    --schema forecasts \
    --variant_id 12345 \
    --y_true_col lineitems_quantity \
    --y_pred_col NOGL_forecast_q3 \
    --metric_keys q1 q2 q3 q4 q5
```

### Command Line Arguments

- `--client_name`: The name of the client (section in the config file) - **Required**
- `--schema`: The name of the schema in the database - **Required**
- `--y_true_col`: The name of the column containing the true values (default: 'lineitems_quantity')
- `--y_pred_col`: The name of the column containing the predicted values (default: 'NOGL_forecast_q3')
- `--variant_id`: Variant ID to filter the data - **Required**
- `--metric_keys`: A list of metric key suffixes to use for evaluation (optional)

## Configuration

The evaluation module uses YAML configuration files. See `configAWSRDS.yaml` for database connection settings.

## Related Documentation

- Main project README: `../README.md`
- Inference module: `../Inference/README.md`

