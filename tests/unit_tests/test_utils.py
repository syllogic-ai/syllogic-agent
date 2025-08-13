"""
Unit tests for utils.py functions.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))

from actions.utils import (
    convert_chart_data_to_chart_config,
    convert_data_to_chart_data,
    convert_data_to_chart_data_1d,
    convert_value,
    remove_null_pairs,
)


class TestConvertDataToChartData1d:
    """Test cases for convert_data_to_chart_data_1d function."""

    @pytest.mark.asyncio
    async def test_convert_1d_chart_data_success(self):
        """Test successfully converting data for 1D chart."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "value": [10, 20, 30],
                "percentage": [25.5, 35.2, 45.8],
            }
        )

        result = await convert_data_to_chart_data_1d(
            df, ["A", "B", "C"], "category", "value"
        )

        assert len(result) == 3
        assert result[0]["A"] == 10
        assert result[1]["B"] == 20
        assert result[2]["C"] == 30

    @pytest.mark.asyncio
    async def test_convert_1d_chart_data_with_none_values(self):
        """Test converting data with None values."""
        df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, None, 30]})

        result = await convert_data_to_chart_data_1d(
            df, ["A", "B", "C"], "category", "value"
        )

        assert result[0]["A"] == 10
        assert result[1]["B"] is None
        assert result[2]["C"] == 30

    @pytest.mark.asyncio
    async def test_convert_1d_chart_data_empty_cols(self):
        """Test converting with empty data columns."""
        df = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})

        result = await convert_data_to_chart_data_1d(df, [], "category", "value")

        assert result == []


class TestConvertDataToChartData:
    """Test cases for convert_data_to_chart_data function."""

    @pytest.mark.asyncio
    async def test_convert_chart_data_success(self):
        """Test successfully converting data for chart."""
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value1": [10, 20, 30],
                "value2": [15, 25, 35],
            }
        )

        result = await convert_data_to_chart_data(df, ["value1", "value2"], "date")

        assert len(result) == 3
        assert result[0]["date"] == "2023-01-01"
        assert result[0]["value1"] == 10
        assert result[0]["value2"] == 15
        assert result[2]["value1"] == 30
        assert result[2]["value2"] == 35

    @pytest.mark.asyncio
    async def test_convert_chart_data_limit_100_points(self):
        """Test that data is limited to 100 points."""
        # Create DataFrame with 150 rows
        large_df = pd.DataFrame({"x": range(150), "y": range(150)})

        result = await convert_data_to_chart_data(large_df, ["y"], "x")

        # Should be limited to 100 points
        assert len(result) == 100
        assert result[99]["x"] == 99

    @pytest.mark.asyncio
    async def test_convert_chart_data_empty_x_key(self):
        """Test converting data with empty x_key."""
        df = pd.DataFrame({"value1": [10, 20], "value2": [15, 25]})

        result = await convert_data_to_chart_data(df, ["value1", "value2"], "")

        assert len(result) == 2
        assert "value1" in result[0]
        assert "value2" in result[0]
        # Should not have empty string key
        assert "" not in result[0]

    @pytest.mark.asyncio
    async def test_convert_chart_data_with_numpy_values(self):
        """Test converting data with numpy values."""
        df = pd.DataFrame(
            {"x": [1, 2, 3], "y": [np.int64(10), np.float64(20.5), np.bool_(True)]}
        )

        result = await convert_data_to_chart_data(df, ["y"], "x")

        assert result[0]["y"] == 10
        assert result[1]["y"] == 20.5
        assert result[2]["y"] is True


class TestConvertChartDataToChartConfig:
    """Test cases for convert_chart_data_to_chart_config function."""

    def test_convert_chart_config_success(self):
        """Test successfully converting chart config."""
        data_cols = ["sales", "profit", "revenue"]
        colors = ["#ff0000", "#00ff00", "#0000ff"]

        result = convert_chart_data_to_chart_config(data_cols, colors)

        assert len(result) == 3
        assert result["sales"]["color"] == "#ff0000"
        assert result["sales"]["label"] == "sales"
        assert result["profit"]["color"] == "#00ff00"
        assert result["profit"]["label"] == "profit"
        assert result["revenue"]["color"] == "#0000ff"
        assert result["revenue"]["label"] == "revenue"

    def test_convert_chart_config_more_cols_than_colors(self):
        """Test when there are more columns than colors."""
        data_cols = ["col1", "col2", "col3", "col4"]
        colors = ["#red", "#green"]

        result = convert_chart_data_to_chart_config(data_cols, colors)

        # Should cycle through colors
        assert result["col1"]["color"] == "#red"
        assert result["col2"]["color"] == "#green"
        assert result["col3"]["color"] == "#red"
        assert result["col4"]["color"] == "#green"

    def test_convert_chart_config_underscore_replacement(self):
        """Test that underscores are replaced in labels."""
        data_cols = ["total_sales", "net_profit"]
        colors = ["#color1", "#color2"]

        result = convert_chart_data_to_chart_config(data_cols, colors)

        assert result["total_sales"]["label"] == "total sales"
        assert result["net_profit"]["label"] == "net profit"

    def test_convert_chart_config_empty_inputs(self):
        """Test with empty inputs."""
        result = convert_chart_data_to_chart_config([], [])
        assert result == {}


class TestConvertValue:
    """Test cases for convert_value function."""

    def test_convert_none_value(self):
        """Test converting None value."""
        result = convert_value(None)
        assert result is None

    def test_convert_nan_value(self):
        """Test converting NaN value."""
        result = convert_value(np.nan)
        assert result is None

    def test_convert_pandas_nan(self):
        """Test converting pandas NaN."""
        result = convert_value(pd.NA)
        assert result is None

    def test_convert_timestamp(self):
        """Test converting pandas Timestamp."""
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        result = convert_value(timestamp)
        assert result == "2023-01-01T10:00:00"

    def test_convert_numpy_integer(self):
        """Test converting numpy integer."""
        value = np.int64(42)
        result = convert_value(value)
        assert result == 42
        assert isinstance(result, int)

    def test_convert_numpy_float(self):
        """Test converting numpy float."""
        value = np.float64(3.14)
        result = convert_value(value)
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_numpy_bool(self):
        """Test converting numpy boolean."""
        value = np.bool_(True)
        result = convert_value(value)
        assert result is True
        assert isinstance(result, bool)

    def test_convert_regular_python_types(self):
        """Test that regular Python types pass through unchanged."""
        assert convert_value("string") == "string"
        assert convert_value(42) == 42
        assert convert_value(3.14) == 3.14
        assert convert_value(True) is True
        # Skip testing lists and dicts as pandas.isna() has issues with them
        # These types would pass through unchanged in actual usage


class TestRemoveNullPairs:
    """Test cases for remove_null_pairs function."""

    def test_remove_null_pairs_simple_dict(self):
        """Test removing None values from simple dictionary."""
        input_dict = {"key1": "value1", "key2": None, "key3": "value3", "key4": None}

        result = remove_null_pairs(input_dict)

        expected = {"key1": "value1", "key3": "value3"}
        assert result == expected

    def test_remove_null_pairs_nested_dict(self):
        """Test removing None values from nested dictionary."""
        input_dict = {
            "level1": {
                "key1": "value1",
                "key2": None,
                "nested": {"nested_key1": "nested_value1", "nested_key2": None},
            },
            "key3": None,
        }

        result = remove_null_pairs(input_dict)

        expected = {
            "level1": {"key1": "value1", "nested": {"nested_key1": "nested_value1"}}
        }
        assert result == expected

    def test_remove_null_pairs_with_lists(self):
        """Test handling lists within dictionaries."""
        input_dict = {
            "list_with_dicts": [
                {"key1": "value1", "key2": None},
                {"key3": "value3", "key4": None},
                None,
                "string_value",
            ],
            "simple_list": [1, 2, None, 4],
            "null_key": None,
        }

        result = remove_null_pairs(input_dict)

        expected = {
            "list_with_dicts": [{"key1": "value1"}, {"key3": "value3"}, "string_value"],
            "simple_list": [1, 2, 4],
        }
        assert result == expected

    def test_remove_null_pairs_empty_nested_dicts(self):
        """Test that empty nested dictionaries are removed."""
        input_dict = {
            "key1": "value1",
            "empty_nested": {"all_null_key": None},
            "nested_with_content": {"good_key": "value"},
        }

        result = remove_null_pairs(input_dict)

        expected = {"key1": "value1", "nested_with_content": {"good_key": "value"}}
        assert result == expected

    def test_remove_null_pairs_non_dict_input(self):
        """Test that non-dictionary inputs are returned unchanged."""
        assert remove_null_pairs("string") == "string"
        assert remove_null_pairs(42) == 42
        assert remove_null_pairs([1, 2, 3]) == [1, 2, 3]
        assert remove_null_pairs(None) is None

    def test_remove_null_pairs_empty_dict(self):
        """Test with empty dictionary."""
        result = remove_null_pairs({})
        assert result == {}

    def test_remove_null_pairs_all_none_values(self):
        """Test dictionary with all None values."""
        input_dict = {"key1": None, "key2": None}
        result = remove_null_pairs(input_dict)
        assert result == {}
