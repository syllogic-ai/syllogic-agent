import logging
from typing import Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# State reducers for LangGraph concurrent updates
def take_last(old: Any, new: Any) -> Any:
    """Reducer that takes the last (most recent) value"""
    return new if new is not None else old
    

def merge_lists(old: List[Any], new: Any) -> List[Any]:
    """Reducer that extends lists with new items"""
    if old is None:
        old = []
    if new is None:
        return old
    if isinstance(new, list):
        return old + new
    return old + [new]


async def convert_data_to_chart_data_1d(
    data_cur: pd.DataFrame, data_cols: list[str], x_key: str, y_col: str
) -> list[dict]:
    """Convert the data to a chart data array for 1D charts (like a pie chart).

    Args:
        data_cur: The current data
        data_cols: The x-key values to use
        x_key: The x-key column to use
        y_col: The y-value column to use

    Returns:
        list[dict]: The chart data array
    """
    chart_data_array = []

    for col in data_cols:
        item = {}
        item[col] = convert_value(data_cur[data_cur[x_key] == col].iloc[0][y_col])
        chart_data_array.append(item)
    return chart_data_array


async def convert_data_to_chart_data(
    data_cur: pd.DataFrame, data_cols: list[str], x_key: str
) -> list[dict]:
    """Convert the data to a chart data array.

    Args:
        data_cur: The current data
        data_cols: The Y-axis columns to convert
        x_key: The X-axis column to use

    Returns:
        list[dict]: The chart data array
    """
    chart_data_array = []
    for i in range(min(len(data_cur), 100)):  # Limit to 100 data points
        item = {}
        if len(x_key) > 0:
            item[x_key] = data_cur.iloc[i][x_key]
        for col in data_cols:
            item[col] = convert_value(data_cur.iloc[i][col])
        chart_data_array.append(item)
    return chart_data_array


def convert_chart_data_to_chart_config(data_cols: list[str], colors: list[str]) -> dict:
    chart_config = {}
    for i, col in enumerate(data_cols):
        chart_config[col] = {
            "color": colors[i % len(colors)],
            "label": col.replace("_", " ").lower(),
        }
    return chart_config


def convert_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_)):
        return bool(value)
    return value


def remove_null_pairs(d):
    """Recursively removes key-value pairs from a dictionary where the value is None.

    Args:
        d (dict): The dictionary to process
    Returns:
        dict: A new dictionary with None values removed.
    """
    if not isinstance(d, dict):
        return d

    result = {}
    for key, value in d.items():
        if value is None:
            # Skip None values
            continue

        if isinstance(value, dict):
            # Recursively process nested dictionaries
            nested_result = remove_null_pairs(value)
            if nested_result:  # Only add if the nested dict is not empty
                result[key] = nested_result
        elif isinstance(value, list):
            # Process lists that might contain dictionaries
            processed_list = [
                remove_null_pairs(item) if isinstance(item, dict) else item
                for item in value
            ]
            # Filter out None values from the list
            processed_list = [item for item in processed_list if item is not None]
            if processed_list:  # Only add if the list is not empty
                result[key] = processed_list
        else:
            # For non-dict, non-list values, keep them as is
            result[key] = value

    return result
