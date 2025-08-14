"""Data processing tools for widget generation."""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import actions conditionally to avoid import errors during development
try:
    from actions.dashboard import (
        get_data_from_file,
        get_sample_from_file,
        get_schema_from_file,
    )
    from actions.utils import convert_data_to_chart_data, convert_value
    HAS_ACTIONS = True
except ImportError:
    # Fallback for when actions module is not available
    HAS_ACTIONS = False


class DataProcessor:
    """Helper class for data processing operations."""

    @staticmethod
    def fetch_file_data(file_id: str) -> Dict[str, Any]:
        """Fetch complete data for a file."""
        if not HAS_ACTIONS:
            raise Exception("Actions module not available - cannot fetch file data")
        
        try:
            data = get_data_from_file(file_id)
            schema = get_schema_from_file(file_id)
            sample = get_sample_from_file(file_id)

            return {
                "data": data,
                "schema": schema,
                "sample": sample,
                "file_id": file_id,
            }
        except Exception as e:
            raise Exception(f"Failed to fetch data for file {file_id}: {str(e)}")

    @staticmethod
    def prepare_data_context(file_ids: List[str]) -> Dict[str, Any]:
        """Prepare comprehensive data context for code generation."""
        context = {
            "files": {},
            "schemas": {},
            "samples": {},
            "total_files": len(file_ids),
        }

        for file_id in file_ids:
            try:
                file_info = DataProcessor.fetch_file_data(file_id)
                context["files"][file_id] = file_info["data"]
                context["schemas"][file_id] = file_info["schema"]
                context["samples"][file_id] = file_info["sample"]
            except Exception as e:
                context[f"error_{file_id}"] = str(e)

        return context

    @staticmethod
    def convert_to_widget_format(data: Any, widget_type: str) -> Dict[str, Any]:
        """Convert data to the appropriate widget format."""
        try:
            if widget_type in ["line", "area"]:
                return DataProcessor._convert_to_line_format(data)
            elif widget_type in ["bar", "radial"]:
                return DataProcessor._convert_to_bar_format(data)
            elif widget_type == "pie":
                return DataProcessor._convert_to_pie_format(data)
            elif widget_type == "table":
                return DataProcessor._convert_to_table_format(data)
            elif widget_type == "kpi":
                return DataProcessor._convert_to_kpi_format(data)
            else:
                return {"error": f"Unknown widget type: {widget_type}"}
        except Exception as e:
            return {"error": f"Conversion error: {str(e)}"}

    @staticmethod
    def _convert_to_line_format(data: Any) -> Dict[str, Any]:
        """Convert data to line chart format."""
        if isinstance(data, pd.DataFrame):
            if len(data.columns) >= 2:
                x_col = data.columns[0]
                y_col = data.columns[1]
                return {"x": data[x_col].tolist(), "y": data[y_col].tolist()}
        elif isinstance(data, dict):
            if "x" in data and "y" in data:
                return data
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "x" in data[0] and "y" in data[0]:
                return {
                    "x": [item["x"] for item in data],
                    "y": [item["y"] for item in data],
                }

        return {"error": "Cannot convert to line chart format"}

    @staticmethod
    def _convert_to_bar_format(data: Any) -> Dict[str, Any]:
        """Convert data to bar chart format."""
        if isinstance(data, pd.DataFrame):
            if len(data.columns) >= 2:
                cat_col = data.columns[0]
                val_col = data.columns[1]
                return {
                    "categories": data[cat_col].tolist(),
                    "values": data[val_col].tolist(),
                }
        elif isinstance(data, dict):
            if "categories" in data and "values" in data:
                return data
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # Try to find category/value keys
                keys = list(data[0].keys())
                if len(keys) >= 2:
                    cat_key = keys[0]
                    val_key = keys[1]
                    return {
                        "categories": [item[cat_key] for item in data],
                        "values": [item[val_key] for item in data],
                    }

        return {"error": "Cannot convert to bar chart format"}

    @staticmethod
    def _convert_to_pie_format(data: Any) -> List[Dict[str, Any]]:
        """Convert data to pie chart format."""
        if isinstance(data, pd.DataFrame):
            if len(data.columns) >= 2:
                label_col = data.columns[0]
                value_col = data.columns[1]
                return [
                    {"label": row[label_col], "value": row[value_col]}
                    for _, row in data.iterrows()
                ]
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                if "label" in data[0] and "value" in data[0]:
                    return data
                else:
                    # Try to convert from other formats
                    keys = list(data[0].keys())
                    if len(keys) >= 2:
                        label_key = keys[0]
                        value_key = keys[1]
                        return [
                            {"label": item[label_key], "value": item[value_key]}
                            for item in data
                        ]

        return [{"error": "Cannot convert to pie chart format"}]

    @staticmethod
    def _convert_to_table_format(data: Any) -> List[Dict[str, Any]]:
        """Convert data to table format."""
        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        elif isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Convert single dict to list
            return [data]

        return [{"error": "Cannot convert to table format"}]

    @staticmethod
    def _convert_to_kpi_format(data: Any) -> Dict[str, Any]:
        """Convert data to KPI format."""
        if isinstance(data, (int, float)):
            return {"value": data}
        elif isinstance(data, dict):
            if "value" in data:
                return data
            else:
                # Try to extract a single numeric value
                numeric_values = {
                    k: v for k, v in data.items() if isinstance(v, (int, float))
                }
                if numeric_values:
                    key, value = list(numeric_values.items())[0]
                    return {"value": value, "label": key}
        elif isinstance(data, pd.DataFrame):
            if not data.empty:
                # Get first numeric value
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value = data[numeric_cols[0]].iloc[0]
                    return {"value": value, "label": numeric_cols[0]}

        return {"error": "Cannot convert to KPI format"}


class CodeGenerator:
    """Helper for generating widget processing code."""

    @staticmethod
    def generate_basic_template(widget_type: str, has_files: bool = True) -> str:
        """Generate basic code template for widget type."""

        if widget_type in ["line", "area"]:
            return CodeGenerator._line_chart_template(has_files)
        elif widget_type in ["bar", "radial"]:
            return CodeGenerator._bar_chart_template(has_files)
        elif widget_type == "pie":
            return CodeGenerator._pie_chart_template(has_files)
        elif widget_type == "table":
            return CodeGenerator._table_template(has_files)
        elif widget_type == "kpi":
            return CodeGenerator._kpi_template(has_files)
        else:
            return CodeGenerator._generic_template(has_files)

    @staticmethod
    def _line_chart_template(has_files: bool) -> str:
        if has_files:
            return """
try:
    # Get data from the first file
    file_id = list(raw_data.keys())[0]
    data = raw_data[file_id]
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        import pandas as pd
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Extract x and y data
    x_col = df.columns[0]
    y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    result = {
        "x": df[x_col].tolist(),
        "y": df[y_col].tolist()
    }
except Exception as e:
    result = {"error": str(e)}
"""
        else:
            return """
try:
    # Sample line chart data
    result = {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 25, 15, 30, 20]
    }
except Exception as e:
    result = {"error": str(e)}
"""

    @staticmethod
    def _bar_chart_template(has_files: bool) -> str:
        if has_files:
            return """
try:
    # Get data from the first file
    file_id = list(raw_data.keys())[0]
    data = raw_data[file_id]
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        import pandas as pd
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Extract categories and values
    cat_col = df.columns[0]
    val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    result = {
        "categories": df[cat_col].tolist(),
        "values": df[val_col].tolist()
    }
except Exception as e:
    result = {"error": str(e)}
"""
        else:
            return """
try:
    # Sample bar chart data
    result = {
        "categories": ["A", "B", "C", "D"],
        "values": [23, 45, 56, 78]
    }
except Exception as e:
    result = {"error": str(e)}
"""

    @staticmethod
    def _pie_chart_template(has_files: bool) -> str:
        if has_files:
            return """
try:
    # Get data from the first file
    file_id = list(raw_data.keys())[0]
    data = raw_data[file_id]
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        import pandas as pd
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Extract labels and values
    label_col = df.columns[0]
    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    result = [
        {"label": row[label_col], "value": row[value_col]}
        for _, row in df.iterrows()
    ]
except Exception as e:
    result = {"error": str(e)}
"""
        else:
            return """
try:
    # Sample pie chart data
    result = [
        {"label": "Category A", "value": 30},
        {"label": "Category B", "value": 25},
        {"label": "Category C", "value": 45}
    ]
except Exception as e:
    result = {"error": str(e)}
"""

    @staticmethod
    def _table_template(has_files: bool) -> str:
        if has_files:
            return """
try:
    # Get data from the first file
    file_id = list(raw_data.keys())[0]
    data = raw_data[file_id]
    
    # Convert to list of dictionaries
    if isinstance(data, list):
        result = data
    else:
        # Convert DataFrame to records
        result = data.to_dict('records') if hasattr(data, 'to_dict') else [data]
        
except Exception as e:
    result = {"error": str(e)}
"""
        else:
            return """
try:
    # Sample table data
    result = [
        {"name": "John", "age": 30, "city": "New York"},
        {"name": "Jane", "age": 25, "city": "San Francisco"},
        {"name": "Bob", "age": 35, "city": "Chicago"}
    ]
except Exception as e:
    result = {"error": str(e)}
"""

    @staticmethod
    def _kpi_template(has_files: bool) -> str:
        if has_files:
            return """
try:
    # Get data from the first file
    file_id = list(raw_data.keys())[0]
    data = raw_data[file_id]
    
    # Extract KPI value
    if isinstance(data, (int, float)):
        result = {"value": data}
    elif isinstance(data, dict):
        # Find first numeric value
        numeric_values = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        if numeric_values:
            key, value = list(numeric_values.items())[0]
            result = {"value": value, "label": key}
        else:
            result = {"value": len(data), "label": "Count"}
    else:
        # Calculate count or sum
        result = {"value": len(data) if hasattr(data, '__len__') else 1, "label": "Total"}
        
except Exception as e:
    result = {"error": str(e)}
"""
        else:
            return """
try:
    # Sample KPI data
    result = {
        "value": 12500,
        "label": "Total Sales",
        "change": "+12%",
        "trend": "up"
    }
except Exception as e:
    result = {"error": str(e)}
"""

    @staticmethod
    def _generic_template(has_files: bool) -> str:
        return """
try:
    # Generic data processing
    if raw_data:
        file_id = list(raw_data.keys())[0]
        result = raw_data[file_id]
    else:
        result = {"message": "No data available"}
except Exception as e:
    result = {"error": str(e)}
"""
