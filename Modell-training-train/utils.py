from datetime import datetime

import numpy as np

from train import TrainConfig
from typing import Type
import pandas as pd
import os


def save_df_to_csv(train_config: Type[TrainConfig], df: pd.DataFrame, result_subfolder: str, name: str) -> None:
    """
    Save a dataframe to csv, in the output folder
    @param train_config: train_config object
    @param df: dataframe to save
    @param name: name of the file

    """
    if not os.path.exists(train_config.get_result_path()):
        os.makedirs(train_config.get_result_path())
    if not os.path.exists(os.path.join(train_config.get_result_path(), result_subfolder)):
        os.makedirs(os.path.join(train_config.get_result_path(), result_subfolder))
    df.to_csv(os.path.join(train_config.get_result_path(), result_subfolder, name), mode="a")


def convert_series_to_datetimes(dates: pd.Series) -> list:
    """
    convert a series of str values to a date if the pattern matches
    @param dates: a pandas series with dates
    @return: list of dates in a date type
    """
    dates = dates.astype(str)
    dates_list = dates.to_list()
    dates_list_format = []
    for date in dates_list:
        dates_list_format.append(try_parsing_date(date))
    return dates_list_format


def try_parsing_date(text: str) -> datetime.date:
    """
    convert a str to a date time if the dateformates matches if not than None
    @param text: str with a date
    @return: date
    """
    # if you find another date in your dataset then use the formatting given bei datetime to insert it
    for fmt in ('%Y-%m-%d', '%m-%Y', '%y', '%m.%Y', '%m/%y', '%m/%Y', '%Y', '%Y.%w', '00-%Y', '-%y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

def get_all_folders_in_path(path: str) -> list:
    """
    get all folders in a path
    @param path: path to the folder
    @return: list of folders
    """
    folders = [f.path for f in os.scandir(path) if f.is_dir()]
    return folders