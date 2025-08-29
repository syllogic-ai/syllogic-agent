import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

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


# Import utilities for robust module loading across different environments


def add_src_to_path() -> str:
    """Add src directory to Python path and return the path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Current dir is: .../src/actions
    # We need to go up 1 level to get to src: actions -> src
    src_dir = os.path.join(current_dir, "..")
    src_dir = os.path.abspath(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return src_dir


def add_project_root_src_to_path() -> str:
    """Add project/src directory to Python path and return the path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Current dir is: .../src/actions
    # Go up 2 levels to project root, then into src
    project_root = os.path.join(current_dir, "..", "..")
    src_path = os.path.join(project_root, "src")
    src_path = os.path.abspath(src_path)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return src_path


def find_src_directory() -> str:
    """Find the src directory by looking for specific markers."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check all paths in sys.path first for an existing src directory
    for path in sys.path:
        if path and os.path.exists(path):
            config_path = os.path.join(path, "config.py")
            actions_path = os.path.join(path, "actions")
            if os.path.exists(config_path) and os.path.exists(actions_path):
                return path
    
    # Look for the src directory by walking up from current location
    search_dir = current_dir
    for _ in range(15):  # Increased limit for deeper nesting
        potential_src = search_dir
        
        # Check if this looks like the src directory
        config_path = os.path.join(potential_src, "config.py")
        actions_path = os.path.join(potential_src, "actions")
        
        if os.path.exists(config_path) and os.path.exists(actions_path):
            return potential_src
            
        # Go up one level
        search_dir = os.path.dirname(search_dir)
        
        # Safety check to prevent going too far up
        if search_dir == "/" or search_dir == os.path.dirname(search_dir):
            break
    
    # Fallback: assume we're in src/actions
    fallback = os.path.join(current_dir, "..")
    return os.path.abspath(fallback)


def robust_import(module_name: str, required_attributes: List[str], 
                 extra_setup_funcs: Optional[List[Callable]] = None) -> Any:
    """
    Robustly import a module with multiple fallback strategies.
    
    Args:
        module_name: Name of the module to import (e.g., 'actions.dashboard')
        required_attributes: List of attributes that must exist in the module
        extra_setup_funcs: Optional list of setup functions to run before each import attempt
        
    Returns:
        The imported module
        
    Raises:
        ImportError: If all import strategies fail
    """
    if extra_setup_funcs is None:
        extra_setup_funcs = []
    
    # Strategy functions that prepare the environment for import
    def add_smart_src_path():
        """Add the intelligently found src directory to path."""
        src_dir = find_src_directory()
        src_dir = os.path.abspath(src_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        return src_dir
    
    def fix_path_duplication():
        """Fix path duplication issues and add correct src path."""
        # Remove any paths that end with /src/src
        fixed_paths = []
        for path in sys.path:
            if path.endswith("/src/src") or path.endswith("\\src\\src"):
                # Fix the doubled path
                fixed_path = path[:-4]  # Remove the extra /src
                if fixed_path not in fixed_paths:
                    fixed_paths.append(fixed_path)
            elif path not in fixed_paths:
                fixed_paths.append(path)
        
        # Find and add the correct src directory
        src_dir = find_src_directory()
        src_dir = os.path.abspath(src_dir)
        if src_dir not in fixed_paths:
            fixed_paths.insert(0, src_dir)
        
        # Update sys.path
        sys.path[:] = fixed_paths
        return src_dir
    
    strategies = [
        lambda: None,  # Try direct import first
        fix_path_duplication,  # Try fixing path duplication issues
        add_smart_src_path,
        add_src_to_path,
        add_project_root_src_to_path,
    ]
    
    last_error = None
    strategy_errors = []
    
    for i, strategy_func in enumerate(strategies):
        try:
            # Run the strategy setup
            setup_result = None
            if strategy_func:
                setup_result = strategy_func()
            
            # Run any extra setup functions
            for setup_func in extra_setup_funcs:
                setup_func()
            
            # Try to import the module
            module = __import__(module_name, fromlist=required_attributes)
            
            # Verify all required attributes exist
            for attr in required_attributes:
                if not hasattr(module, attr):
                    raise AttributeError(f"Module {module_name} missing required attribute: {attr}")
            
            return module
            
        except (ImportError, AttributeError) as e:
            strategy_name = strategy_func.__name__ if strategy_func else "direct_import"
            strategy_errors.append(f"Strategy {i+1} ({strategy_name}): {str(e)}")
            last_error = e
            if i == len(strategies) - 1:  # Last strategy failed
                break
            continue
    
    # All strategies failed - provide detailed error information
    error_details = "\n".join(strategy_errors)
    error_msg = f"Failed to import {module_name} with all strategies.\nStrategy details:\n{error_details}\nCurrent Python path: {sys.path[:8]}"
    raise ImportError(error_msg)


def import_actions_dashboard():
    """Import actions.dashboard module with all required functions."""
    required_functions = ['get_data_from_file', 'get_sample_from_file', 'get_schema_from_file', 
                         'create_widget', 'update_widget', 'delete_widget']
    
    return robust_import('actions.dashboard', required_functions)


def import_config():
    """Import config module with required functions."""
    required_functions = ['get_supabase_client', 'get_e2b_api_key']
    
    return robust_import('config', required_functions)


# Validation utilities for widget data processing


def analyze_schema_validation_error(result_data, validation_error):
    """
    Analyze schema validation errors and provide detailed feedback about what's wrong.

    Args:
        result_data: The data that failed validation
        validation_error: The Pydantic validation error

    Returns:
        str: Detailed error message explaining what's wrong
    """
    try:
        from pydantic import ValidationError

        error_details = []

        # Check if it's a Pydantic ValidationError
        if isinstance(validation_error, ValidationError):
            for error in validation_error.errors():
                loc = (
                    " -> ".join(str(x) for x in error["loc"])
                    if error["loc"]
                    else "root"
                )
                msg = error["msg"]
                error_type = error["type"]

                if error_type == "missing":
                    error_details.append(f"Missing required field: '{loc}'")
                elif error_type == "type_error":
                    error_details.append(f"Wrong type for field '{loc}': {msg}")
                elif error_type == "literal_error":
                    error_details.append(f"Invalid value for field '{loc}': {msg}")
                elif error_type == "value_error":
                    error_details.append(f"Invalid value for field '{loc}': {msg}")
                else:
                    error_details.append(
                        f"Error in field '{loc}': {msg} (type: {error_type})"
                    )
        else:
            error_details.append(f"Validation error: {str(validation_error)}")

        # Additional analysis of the result structure
        if isinstance(result_data, dict):
            schema_fields = {
                "chartType": "Must be one of: line, bar, pie, area, radial, kpi, table",
                "title": "Must be a string describing the chart",
                "description": "Must be a string explaining what the chart shows",
                "data": "Must be an array of objects containing the actual data points",
                "chartConfig": "Must be a dictionary with data field names as keys",
                "xAxisConfig": "Must be an object with 'dataKey' field",
            }

            missing_fields = []
            wrong_types = []

            for field, description in schema_fields.items():
                if field not in result_data:
                    missing_fields.append(f"'{field}': {description}")
                elif field == "data" and not isinstance(result_data[field], list):
                    wrong_types.append(
                        f"'{field}': Expected array, got {type(result_data[field]).__name__}"
                    )
                elif field in ["chartConfig", "xAxisConfig"] and not isinstance(
                    result_data[field], dict
                ):
                    wrong_types.append(
                        f"'{field}': Expected object, got {type(result_data[field]).__name__}"
                    )
                elif field in ["title", "description", "chartType"] and not isinstance(
                    result_data[field], str
                ):
                    wrong_types.append(
                        f"'{field}': Expected string, got {type(result_data[field]).__name__}"
                    )

            if missing_fields:
                error_details.extend([f"Missing: {field}" for field in missing_fields])
            if wrong_types:
                error_details.extend([f"Type error: {field}" for field in wrong_types])

        # Combine all error details
        if error_details:
            return "Validation errors: " + "; ".join(error_details)
        else:
            return f"Schema validation failed: {str(validation_error)}"

    except Exception as e:
        return f"Error analyzing validation failure: {str(e)}. Original error: {str(validation_error)}"


def get_chart_config_schema_string():
    """
    Programmatically extract the ChartConfigSchema as a formatted string for code generation prompts.

    Returns:
        str: A formatted string representation of the ChartConfigSchema that can be used in prompts.
    """
    try:
        # Try to import ChartConfigSchema
        from agent.models import ChartConfigSchema
        
        # Get the JSON schema for ChartConfigSchema
        schema = ChartConfigSchema.model_json_schema()

        # Extract properties and their descriptions
        properties = schema.get("properties", {})

        # Build a formatted string representation
        schema_lines = []
        schema_lines.append("ChartConfigSchema Structure:")
        schema_lines.append("```json")
        schema_lines.append("{")

        # Add each property with its type and description
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get(
                "type", prop_info.get("anyOf", [{}])[0].get("type", "unknown")
            )
            description = prop_info.get("description", "No description")

            # Handle special cases for complex types
            if prop_name == "chartType":
                enum_values = prop_info.get("enum", [])
                schema_lines.append(
                    f'  "{prop_name}": "{" | ".join(enum_values)}" // {description}'
                )
            elif prop_name == "chartConfig":
                schema_lines.append(f'  "{prop_name}": {{')
                schema_lines.append(f'    "<item_name>": {{')
                schema_lines.append(
                    f'      "label": "string", // Display label for the chart item'
                )
                schema_lines.append(
                    f'      "color": "string"  // Color value (e.g., "var(--chart-1)" or hex)'
                )
                schema_lines.append(f"    }}")
                schema_lines.append(f"  }}, // {description}")
            elif prop_name == "xAxisConfig":
                schema_lines.append(f'  "{prop_name}": {{')
                schema_lines.append(
                    f'    "dataKey": "string" // The data key to use for the X-axis'
                )
                schema_lines.append(f"  }}, // {description}")
            elif prop_name == "data":
                schema_lines.append(f'  "{prop_name}": [ // {description}')
                schema_lines.append(
                    f'    {{ "key": "value", ... }}, // Array of data objects'
                )
                schema_lines.append(f"    ...")
                schema_lines.append(f"  ],")
            else:
                schema_lines.append(f'  "{prop_name}": "{prop_type}" // {description}')

        schema_lines.append("}")
        schema_lines.append("```")

        # Add the example from the model's schema
        example = schema.get("example")
        if example and isinstance(example, dict):
            schema_lines.append("\nExample:")
            schema_lines.append("```json")
            schema_lines.append(json.dumps(example, indent=2))
            schema_lines.append("```")

        return "\n".join(schema_lines)

    except Exception as e:
        # Fallback to a hardcoded schema if extraction fails
        return """ChartConfigSchema Structure:
```json
{
  "chartType": "line | bar | pie | area | radial | kpi | table", // Type of chart
  "title": "string", // Title of the chart
  "description": "string", // Description of the chart
  "data": [ // Data for the chart
    { "key": "value", ... }, // Array of data objects
    ...
  ],
  "chartConfig": {
    "<item_name>": {
      "label": "string", // Display label for the chart item
      "color": "string"  // Color value (e.g., "var(--chart-1)" or hex)
    }
  }, // Dictionary of chart items
  "xAxisConfig": {
    "dataKey": "string" // The data key to use for the X-axis
  } // X-axis configuration
}
```"""
