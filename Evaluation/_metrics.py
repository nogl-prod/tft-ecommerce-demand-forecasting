#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename : _metrics.py
# Author : Tuhin Mallick
# Import necessary libraries
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics

# Define an abstract class for custom metric implementations


class AbstractMetric(ABC):      
    @staticmethod
    @abstractmethod
    def __call__(pred, label, weights=None):
        pass

# Define a SMAPE class that inherits from the AbstractMetric class


class SMAPE(AbstractMetric):
    name = "SMAPE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.

        Returns:
            float: The SMAPE metric value.
        """
        if not weights:
            weights = None
        return 100 * np.average(
            2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)),
            weights=weights,
        )


# Define a function to compute the Normalised Quantile Loss


def normalised_quantile_loss(y_pred, y, quantile, weights=None):
    """
    Compute the normalised quantile loss.

    Implementation of the q-Risk function from https://arxiv.org/pdf/1912.09363.pdf.

    Args:
        y_pred (np.ndarray): Predicted values.
        y (np.ndarray): True values.
        quantile (float): Quantile value (between 0 and 1).
        weights (np.ndarray, optional): Weights for each sample, default is None.

    Returns:
        float: The normalised quantile loss.
    """
    # Compute the prediction underflow
    prediction_underflow = y - y_pred
    # Compute the weighted errors
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (
        1.0 - quantile
    ) * np.maximum(-prediction_underflow, 0.0)
    # Check if weights are provided
    if weights is not None:
        # Apply the weights
        weighted_errors = weighted_errors * weights
        # Compute the normaliser
        y = y * weights
    # Check if there are any zero values in y
    if (np.array(y) == 0).sum() > 0:
        y = np.where(y == 0, 1e-5, y)  # Replace zero values with a small constant
    # Compute the normalised quantile loss
    loss = weighted_errors.sum()
    # Sum of the absolute values of the true values
    normaliser = abs(y).sum()
    # Return the normalised quantile loss
    return 2 * loss / normaliser  # Normalised Quantile Loss


class P50_loss(AbstractMetric):
    name = "P50"
    selector = 1

    @staticmethod
    def __call__(y_true, y_pred, weights=None):
        """
        Calculate the P50 quantile loss of the predictions.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The P50 quantile loss.
        """
        return normalised_quantile_loss(y_true, y_pred, 0.5, weights)


class P90_loss(AbstractMetric):
    name = "P90"
    selector = 2

    @staticmethod
    def __call__(y_true, y_pred, weights=None):
        """
        Calculate the P90 quantile loss of the predictions.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The P90 quantile loss.
        """
        return normalised_quantile_loss(y_true, y_pred, 0.9, weights)


# Normalized Deviation


class ND(AbstractMetric):
    name = "ND"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Normalized Deviation of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample.

        Returns:
            float: The Normalized Deviation metric value.
        """
        diff = np.abs(y_true - y_pred)

        # Check if there are any zero values in y_true
        if (np.array(y_true) == 0).sum() > 0:
            y_true = np.where(y_true == 0, 1e-5, y_true)  # Replace zero values with a small constant

        return (
            np.sum(diff * weights) / np.sum(np.abs(y_true) * weights)
            if weights
            else np.sum(diff) / np.sum(np.abs(y_true))
        )



class MAE(AbstractMetric):
    name = "MAE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None, return_individual=False):
        """
        Calculate the Mean Absolute Error (MAE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual errors per sample, default is False.

        Returns:
            float or np.ndarray: The MAE metric value, or individual errors if return_individual is True.
        """
        if not weights:
            weights = None
        if return_individual:
            return np.average(np.abs(y_pred - y_true), weights=weights, axis=0)
        else:
            return np.average(np.abs(y_pred - y_true), weights=weights)


class MSE(AbstractMetric):
    name = "MSE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None, return_individual=False):
        """
        Calculate the Mean Squared Error (MSE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual errors per sample, default is False.

        Returns:
            float or np.ndarray: The MSE metric value, or individual errors if return_individual is True.
        """
        if not weights:
            weights = None
        if return_individual:
            return np.average((y_pred - y_true) ** 2, weights=weights, axis=0)
        else:
            return np.average((y_pred - y_true) ** 2, weights=weights)


class R_Squared(AbstractMetric):
    name = "R_Squared"

    @staticmethod
    def __call__(y_pred, y_true, weights=None, return_individual=True):
        """
        Calculate the R-squared (coefficient of determination) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual R-squared values per sample, default is False.

        Returns:
            float or np.ndarray: The R-squared metric value, or individual R-squared values if return_individual is True.
        """
        if not weights:
            import pdb;pdb.set_trace()
            return (
                skmetrics.r2_score(y_pred, y_true, multioutput="raw_values")
                if return_individual
                else skmetrics.r2_score(y_pred, y_true)
            )
        values = skmetrics.r2_score(y_pred, y_true, multioutput="raw_values")
        if return_individual:
            return values * weights
        return np.sum(values * weights) / np.sum(weights)


class WMSMAPE(AbstractMetric):
    name = "WMSMAPE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None, return_individual=False):
        """
        Calculate the Weighted Mean Symmetric Mean Absolute Percentage Error (WMSMAPE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.
            return_individual (bool): If True, return individual WMSMAPE values per sample, default is False.

        Returns:
            float or np.ndarray: The WMSMAPE metric value, or individual WMSMAPE values if return_individual is True.
        """
        if weights:
            if return_individual:
                return (
                    2
                    * weights
                    * np.abs(y_pred - y_true)
                    / (np.maximum(y_true, 1) + np.abs(y_pred))
                )
            else:
                return (
                    100.0
                    / np.sum(weights)
                    * np.sum(
                        2
                        * weights
                        * np.abs(y_pred - y_true)
                        / (np.maximum(y_true, 1) + np.abs(y_pred))
                    )
                )
        if return_individual:
            return (
                2 * np.abs(y_pred - y_true) / (np.maximum(y_true, 1) + np.abs(y_pred))
            )
        else:
            return (
                100.0
                / len(y_true)
                * np.sum(
                    2
                    * np.abs(y_pred - y_true)
                    / (np.maximum(y_true, 1) + np.abs(y_pred))
                )
            )


class Accuracy(AbstractMetric):
    name = "Accuracy"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the accuracy of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The accuracy metric value in percentage.
        """
        try:
            if weights is not None:
                raise NotImplementedError("Weighted accuracy is not supported.")
            # Handle division by zero
            if (np.array(y_true) == 0).sum() > 0:
                y_true = np.where(y_true == 0, 1e-5, y_true)
            acc = 1 - np.mean(
                np.clip(np.abs((y_true - y_pred) / y_true), a_min=0, a_max=1)
            )
            acc_percent = acc * 100
        except Exception as e:
            acc_percent = 0
            print(f"ERROR: calculating accuracy - {str(e)}")
        return acc_percent


class MAPE(AbstractMetric):
    name = "MAPE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The MAPE metric value in percentage.
        """
        try:
            if weights is not None:
                raise NotImplementedError("Weighted MAPE is not supported.")
            # Handle division by zero
            if (np.array(y_true) == 0).sum() > 0:
                y_true = np.where(y_true == 0, 1e-5, y_true)
                error_percent = np.mean(
                    np.clip(np.abs((y_true - y_pred) / y_true), a_min=0, a_max=1)
                )
            else:
                error_percent = np.mean(np.abs((y_true - y_pred) / y_true))
        except Exception as e:
            error_percent = 100
            print(f"ERROR: calculating MAPE - {str(e)}")
        # print(y_pred)
        return error_percent * 100


class RMSE(AbstractMetric):
    name = "RMSE"

    @staticmethod
    def __call__(y_pred, y_true, weights=None):
        """
        Calculate the Root Mean Squared Error (RMSE) of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            weights (np.ndarray): Weights for each sample, default is None.

        Returns:
            float: The RMSE metric value.
        """
        if not weights:
            weights = None
        return np.sqrt(
            skmetrics.mean_squared_error(
                y_true=y_true, y_pred=y_pred
            )
        )


class DirectionalSymmetry(AbstractMetric):
    name = "DirectionalSymmetry"

    @staticmethod
    def __call__(y_pred, y_true, tolerance=1, weights=None):
        """
        Calculate the directional symmetry of the predictions.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.
            tolerance (int, optional): Tolerance level for percentage change. Default is 1.
            weights (np.ndarray): Weights for each sample, not supported in this metric.

        Returns:
            float: The directional symmetry metric value in percentage.
        """

        name = "DirectionalSymmetry"
        import pdb; pdb.set_trace()
        try:
            if weights is not None:
                raise NotImplementedError(
                    "Weighted directional symmetry is not supported."
                )
            # Check if tolerance is valid
            if tolerance < 0:
                raise ValueError("Tolerance cannot be less than zero!")
            # Define common variables for true and predicted differences
            if y_true.ndim > 0:
                true_diff = np.diff(y_true)
            else:
                # Handle the case when y_true is a scalar or an empty array
                # You can raise an error or set true_diff to an appropriate value depending on your use case
                raise ValueError("y_true must have at least one dimension")

            pred_diff = np.diff(y_pred)

            # Case not zero: modify true_diff and pred_diff
            if tolerance != 0:
                # Scale tolerance
                tolerance /= 100

                # Get %change for true values and update true_diff accordingly
                tmp = pd.Series(y_true).pct_change().iloc[1:]
                true_diff[tmp.abs() < tolerance] = 0

                # Get %change for predicted values and update pred_diff accordingly
                tmp = pd.Series(y_pred).pct_change().iloc[1:]
                pred_diff[tmp.abs() < tolerance] = 0
            # Core formula for directional symmetry
            d = (true_diff * pred_diff) > 0
            d[
                (true_diff == 0) & (pred_diff == 0)
            ] = 1  # Case of plateau for both y_true and y_pred
            dsymm = np.round(100 * d.sum() / len(d), 2)
        except Exception as e:
            dsymm = 0
            print(f"ERROR: calculating directional symmetry - {str(e)}")
        return dsymm


METRICS = {
    "SMAPE": SMAPE,
    "WMSMAPE": WMSMAPE,             # Cannot be used for per row calculation
    "MSE": MSE,
    "MAE": MAE,
    "P50": P50_loss,
    "P90": P90_loss,
    "R_Squared": R_Squared,         # Cannot be used for per row calculation
    "ND": ND,
    "Accuracy": Accuracy,
    "MAPE": MAPE,
    "RMSE": RMSE,           # Cannot be used for per row calculation
    "DirectionalSymmetry": DirectionalSymmetry,     # Cannot be used for per row calculation
}
