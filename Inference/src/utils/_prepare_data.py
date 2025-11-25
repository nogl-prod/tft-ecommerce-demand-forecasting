import pandas as pd
from typing import Tuple
import concurrent.futures
import traceback
import logging

# Set up logging to output to console
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

class DataWrangling:
    """
    A class for preparing data for machine learning inference.

    Attributes:
    - static_categoricals (List[str]): List of column names for static categorical features.
    - static_reals (List[str]): List of column names for static numerical features.
    - time_varying_known_categoricals (List[str]): List of column names for time-varying
      known categorical features.
    - time_varying_known_reals (List[str]): List of column names for time-varying
      known numerical features.
    - time_varying_unknown_categoricals (List[str]): List of column names for time-varying
      unknown categorical features.
    - time_varying_unknown_reals (List[str]): List of column names for time-varying
      unknown numerical features.

    Methods:
    - prepare_data_for_inference: Prepare data for inference.
    - _rename_columns: Rename columns in a dataframe.
    - _categorize_static_categoricals: Categorize static categorical features.
    - _categorize_static_reals: Categorize static numerical features.
    - _categorize_time_varying_known_categoricals: Categorize time-varying known
      categorical features.
    - _categorize_time_varying_known_reals: Categorize time-varying known numerical
      features.
    - _categorize_time_varying_unknown_categoricals: Categorize time-varying unknown
      categorical features.
    - _categorize_time_varying_unknown_reals: Categorize time-varying unknown numerical
      features.
    - _categorize_columns: Categorize columns in a dataframe.
    - _adjust_data_types: Adjust data types in a dataframe.
    - _limit_time_frame: Limit the time frame of a dataframe.

    Usage:
    ```
    # create a DataWrangling object
    dw = DataWrangling()

    # prepare data for inference
    preprocessed_data, last_day, training_cutoff, end_of_timeseries = \
        dw.prepare_data_for_inference(data, last_observation_day)

    # rename columns in a dataframe
    dw._rename_columns(data)

    # categorize static categorical features
    dw._categorize_static_categoricals(data)

    # categorize static numerical features
    dw._categorize_static_reals(data)

    # categorize time-varying known categorical features
    dw._categorize_time_varying_known_categoricals(data)

    # categorize time-varying known numerical features
    dw._categorize_time_varying_known_reals(data)

    # categorize time-varying unknown categorical features
    dw._categorize_time_varying_unknown_categoricals(data)

    # categorize time-varying unknown numerical features
    dw._categorize_time_varying_unknown_reals(data)

    # categorize columns in a dataframe
    dw._categorize_columns(data)

    # adjust data types in a dataframe
    dw._adjust_data_types(data)

    # limit the time frame of a dataframe
    limited_data, last_day, training_cutoff, end_of_timeseries = \
        dw._limit_time_frame(data, last_observation_day, max_prediction_length)
    ```
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.static_categoricals = []
        self.static_reals = []
        self.time_varying_known_categoricals = []
        self.time_varying_known_reals = []
        self.time_varying_unknown_categoricals = []
        self.time_varying_unknown_reals = []

    def prepare_data_for_inference(
        self,
        last_observation_day: str,
        max_prediction_length: int = 90,
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """
        Prepare data for inference by renaming columns, categorizing columns,
        and adjusting data types. Then, limit the time frame to the last observation
        period and return the preprocessed data as well as relevant time frame information.

        Args:
        - last_observation_day (str): Last day of the observation period.
        - max_prediction_length (int): Maximum length of the prediction period.

        Returns:
        - Tuple[pd.DataFrame, float, float, float]: Preprocessed data, last day of available data,
          training cutoff, and end of time series.

        Raises:
        - Exception: Any exception encountered during method execution.

        Usage:
        ```
        dw = DataWrangling()
        preprocessed_data, last_day, training_cutoff, end_of_timeseries = \
            dw.prepare_data_for_inference(data, last_observation_day)
        ```
        """
        try:
            logging.debug("Preparing data for inference")
            logging.info("Renaming columns...")
            self._rename_columns()
            logging.info("Categorizing columns...")
            self._categorize_columns()
            logging.info("Adjusting data types...")
            self._adjust_data_types()
            logging.info("Handling special events...")
            self._handle_special_events()
            return self._limit_time_frame(
            last_observation_day, max_prediction_length
            )
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.info("Error encountered during data preparation: {}".format(e))
            raise e

    def _rename_columns(self) -> None:
        """
        Rename columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        logging.debug("Renaming columns")
        self.data.columns = self.data.columns.str.replace(".", "_")

    def _categorize_static_categoricals(self) -> None:
        """
        Categorize static categorical columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.static_categoricals = [
            column for column in self.data.columns if column.startswith("SC_")
        ]

    def _categorize_static_reals(self) -> None:
        """
        Categorize static real valued columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.static_reals = [
            column for column in self.data.columns if column.startswith("SR_")
        ]

    def _categorize_time_varying_known_categoricals(self) -> None:
        """
        Categorize time varying known categorical columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.time_varying_known_categoricals = [
            column for column in self.data.columns if column.startswith("TVKC_")
        ]

    def _categorize_time_varying_known_reals(self) -> None:
        """
        Categorize time varying known real valued columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.time_varying_known_reals = [
            column for column in self.data.columns if column.startswith("TVKR_")
        ]

    def _categorize_time_varying_unknown_categoricals(self) -> None:
        """
        Categorize time varying unknown categorical columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.time_varying_unknown_categoricals = [
            column for column in self.data.columns if column.startswith("TVUC_")
        ]

    def _categorize_time_varying_unknown_reals(self) -> None:
        """
        Categorize time varying unknown real valued columns in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        self.time_varying_unknown_reals = [
            column for column in self.data.columns if column.startswith("TVUR_")
        ]
    def _handle_special_events(self) -> None:
        special_events = [
            "TVKC_external_holidays_and_special_events_by_date_external_importantSalesEvent",
            "TVKC_external_holidays_and_special_events_by_date_external_secondarySalesEvent",
            "TVKC_external_holidays_and_special_events_by_date_black_friday",
            "TVKC_external_holidays_and_special_events_by_date_cyber_monday",
            "TVKC_external_holidays_and_special_events_by_date_mothers_day",
            "TVKC_external_holidays_and_special_events_by_date_valentines_day",
            "TVKC_external_holidays_and_special_events_by_date_christmas_eve",
            "TVKC_external_holidays_and_special_events_by_date_fathers_day",
            "TVKC_external_holidays_and_special_events_by_date_orthodox_new_year",
            "TVKC_external_holidays_and_special_events_by_date_chinese_new_year",
            "TVKC_external_holidays_and_special_events_by_date_rosenmontag",
            "TVKC_external_holidays_and_special_events_by_date_carneval",
            "TVKC_external_holidays_and_special_events_by_date_start_of_ramadan",
            "TVKC_external_holidays_and_special_events_by_date_start_of_eurovision",
            "TVKC_external_holidays_and_special_events_by_date_halloween",
            "TVKC_external_holidays_and_special_events_by_date_saint_nicholas",
            "TVKC_external_holidays_and_special_events_by_date_external_holiday",
        ]

        for holiday_col in special_events:
            if holiday_col in self.time_varying_known_categoricals:
                self.time_varying_known_categoricals.remove(holiday_col)

        # Create the "special_events" column and set it to a default value (e.g., 0)

        self.time_varying_known_categoricals.append("special_events")


    # TODO: Might have to remove multiprocessing
    def _categorize_columns(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._categorize_static_categoricals),
                executor.submit(self._categorize_static_reals),
                executor.submit(self._categorize_time_varying_known_categoricals),
                executor.submit(self._categorize_time_varying_known_reals),
                executor.submit(
                    self._categorize_time_varying_unknown_categoricals
                ),
                executor.submit(self._categorize_time_varying_unknown_reals),
            ]
            for _ in concurrent.futures.as_completed(futures):
                pass

    def _adjust_data_types(self) -> None:
        """
        Adjust the data types in the given DataFrame.

        Args:
            data: A pandas DataFrame to be processed.

        Returns:
            None
        """
        logging.debug("Adjusting data types")
        
        # ADJUST to TVKR_time_idx (time_varying_known_reals.remove('TVKR_time_idx'))
        self.data['TVKR_time_idx'] = self.data['TVKR_time_idx'].astype("int")
        self.time_varying_known_reals.remove('TVKR_time_idx')
        
        self.data[self.static_categoricals + self.time_varying_known_categoricals +
            self.time_varying_unknown_categoricals] = \
            self.data[self.static_categoricals + self.time_varying_known_categoricals +
                self.time_varying_unknown_categoricals].astype(str).astype("category")
        self.data[self.static_reals + self.time_varying_known_reals +
            self.time_varying_unknown_reals] = \
            self.data[self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals].astype(float)

    def _limit_time_frame(
        self, last_observation_day: str, max_prediction_length: int
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """
        Limit the time frame to the last observation period and return the resulting dataframe
        as well as relevant time frame information.

        Args:
        - data (pd.DataFrame): Dataframe containing the data to be limited.
        - last_observation_day (str): Last day of the observation period.
        - max_prediction_length (int): Maximum length of the prediction period.

        Returns:
        - Tuple[pd.DataFrame, float, float, float]: Limited data, last day of available data,
          training cutoff, and end of time series.

        Raises:
        - Exception: Any exception encountered during method execution.

        Usage:
        ```
        dw = DataWrangling()
        limited_data, last_day, training_cutoff, end_of_timeseries = \
            dw._limit_time_frame(data, last_observation_day, max_prediction_length)
        ```
        """
        try:
            logging.debug("Limiting time frame")
            last_day_we_have_data = self.data[
                 pd.to_datetime(self.data.TVKC_daydate, format='%Y-%m-%d') == pd.to_datetime(last_observation_day, format='%Y-%m-%d')
            ].TVKR_time_idx.mean()
            if last_day_we_have_data.is_integer():
                last_day_we_have_data = (int(last_day_we_have_data))
            else:
                logging.info("last_day_we_have_data = ", last_day_we_have_data)
                raise ValueError("Different time_idx for daydates in data.")
            end_of_timeseries = last_day_we_have_data + max_prediction_length
            logging.info("Limiting time frame...")
            data_inf = self.data[self.data.TVKR_time_idx <= end_of_timeseries]
            training_cutoff = data_inf.TVKR_time_idx.max() - max_prediction_length
            return data_inf, last_day_we_have_data, training_cutoff, end_of_timeseries
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.info("Error encountered during time frame limiting: {}".format(e))
            raise e
