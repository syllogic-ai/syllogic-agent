"""
Dashboard data management utilities for file operations.
Provides functions to fetch data, schema, and samples from stored files.
"""

import pandas as pd
import requests
import os
from typing import Dict, Any, Optional, List
from supabase import Client
import logging
import io
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

## Files

def get_data_from_file(
    supabase: Client,
    file_id: str
) -> pd.DataFrame:
    """
    Get complete data from a file using its storage path.
    
    Args:
        supabase: Supabase client instance
        file_id: File identifier
        
    Returns:
        pandas.DataFrame containing the file data
        
    Raises:
        Exception: If file not found or data retrieval fails
    """
    try:
        # Get file information from database
        file_info = supabase.table("files").select("storage_path, original_filename, mime_type").eq("id", file_id).single().execute()
        
        if not file_info.data:
            raise Exception(f"File {file_id} not found")
        
        storage_path = file_info.data["storage_path"]
        original_filename = file_info.data["original_filename"]
        mime_type = file_info.data.get("mime_type")
        
        # Get base URL from environment variables
        supabase_storage_base_url = os.getenv("SUPABASE_STORAGE_BASE_URL")
        if not supabase_storage_base_url:
            # Construct from SUPABASE_URL if SUPABASE_STORAGE_BASE_URL not available
            supabase_url = os.getenv("SUPABASE_URL", "")
            if supabase_url:
                supabase_storage_base_url = f"{supabase_url}/storage/v1/object/public"
            else:
                raise Exception("SUPABASE_STORAGE_BASE_URL or SUPABASE_URL environment variable not set")
        
        # Construct full URL
        file_url = f"{supabase_storage_base_url}/{storage_path}"
        
        # Download file content
        response = requests.get(file_url)
        response.raise_for_status()
        
        # Determine file type and read accordingly
        if original_filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(response.text))
        elif original_filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(response.content))
        elif original_filename.lower().endswith('.json'):
            df = pd.read_json(io.StringIO(response.text))
        else:
            # Default to CSV parsing
            df = pd.read_csv(io.StringIO(response.text))
        
        logger.info(f"Successfully loaded data from file {file_id}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error getting data from file {file_id}: {str(e)}")
        raise


def get_schema_from_file(
    supabase: Client,
    file_id: str
) -> Dict[str, Any]:
    """
    Get schema information from a file.
    
    Args:
        supabase: Supabase client instance
        file_id: File identifier
        
    Returns:
        Dict containing schema information with columns and their types
        
    Raises:
        Exception: If file not found or schema extraction fails
    """
    try:
        # Get the dataframe
        df = get_data_from_file(supabase, file_id)
        
        # Extract schema information
        schema = {
            "columns": [],
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
        
        for column in df.columns:
            column_info = {
                "name": column,
                "type": str(df[column].dtype),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique())
            }
            
            # Add sample values for better understanding
            non_null_values = df[column].dropna()
            if len(non_null_values) > 0:
                column_info["sample_values"] = non_null_values.head(3).tolist()
            
            schema["columns"].append(column_info)
        
        logger.info(f"Successfully extracted schema from file {file_id}: {len(schema['columns'])} columns")
        return schema
        
    except Exception as e:
        logger.error(f"Error getting schema from file {file_id}: {str(e)}")
        raise


def get_sample_from_file(
    supabase: Client,
    file_id: str,
    num_rows: int = 3
) -> Dict[str, Any]:
    """
    Get sample data from a file (header + specified number of rows).
    
    Args:
        supabase: Supabase client instance
        file_id: File identifier
        num_rows: Number of data rows to return (default: 3)
                 Note: This returns header + num_rows, so 4 total rows for num_rows=3
        
    Returns:
        Dict containing sample data with headers and rows
        
    Raises:
        Exception: If file not found or sample extraction fails
    """
    try:
        # Get the dataframe
        df = get_data_from_file(supabase, file_id)
        
        # Get sample data (header + requested number of rows)
        sample_df = df.head(num_rows)
        
        # Convert to a format that's easy to work with
        sample_data = {
            "headers": list(df.columns),
            "rows": sample_df.values.tolist(),
            "total_rows_in_file": len(df),
            "sample_rows_returned": len(sample_df),
            "requested_rows": num_rows
        }
        
        logger.info(f"Successfully extracted sample from file {file_id}: {len(sample_data['rows'])} rows returned")
        return sample_data
        
    except Exception as e:
        logger.error(f"Error getting sample from file {file_id}: {str(e)}")
        raise


## Widgets

def create_widget(
    supabase: Client,
    dashboard_id: str,
    title: str,
    widget_type: str,
    config: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
    sql: Optional[str] = None,
    layout: Optional[Dict[str, Any]] = None,
    chat_id: Optional[str] = None,
    order: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a new widget in the dashboard.
    
    Args:
        supabase: Supabase client instance
        dashboard_id: Dashboard identifier
        title: Widget title
        widget_type: Widget type ('text', 'chart', 'kpi', 'table')
        config: Widget configuration object
        data: Optional widget data
        sql: Optional SQL query for the widget
        layout: Optional React Grid Layout position
        chat_id: Optional chat ID if widget was created from chat
        order: Optional order for positioning
        **kwargs: Additional widget properties
        
    Returns:
        Dict containing the created widget data
        
    Raises:
        Exception: If widget creation fails
    """
    try:
        widget_id = str(uuid.uuid4())
        
        # Prepare widget data
        widget_data = {
            "id": widget_id,
            "dashboard_id": dashboard_id,
            "title": title,
            "type": widget_type,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add optional fields if provided
        if data is not None:
            widget_data["data"] = data
        if sql is not None:
            widget_data["sql"] = sql
        if layout is not None:
            widget_data["layout"] = layout
        if chat_id is not None:
            widget_data["chat_id"] = chat_id
        if order is not None:
            widget_data["order"] = order
            
        # Add any additional fields from kwargs
        for key, value in kwargs.items():
            if value is not None and key not in widget_data:
                widget_data[key] = value
        
        # Insert widget into database
        result = supabase.table("widgets").insert(widget_data).execute()
        
        if result.data:
            logger.info(f"Created widget {widget_id} in dashboard {dashboard_id}")
            return result.data[0]
        else:
            raise Exception(f"Failed to create widget: No data returned")
            
    except Exception as e:
        logger.error(f"Error creating widget in dashboard {dashboard_id}: {str(e)}")
        raise


def update_widget(
    supabase: Client,
    widget_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Update an existing widget with the provided fields.
    
    Args:
        supabase: Supabase client instance
        widget_id: Widget identifier
        **kwargs: Fields to update (title, type, config, data, sql, layout, etc.)
        
    Returns:
        Dict containing the updated widget data
        
    Raises:
        Exception: If widget update fails
    """
    try:
        # Prepare update data
        update_data = {
            "updated_at": datetime.now().isoformat()
        }
        
        # Add fields to update from kwargs
        allowed_fields = [
            "title", "type", "config", "data", "sql", "layout", 
            "chat_id", "order", "is_configured", "cache_key", "last_data_fetch"
        ]
        
        for key, value in kwargs.items():
            if key in allowed_fields and value is not None:
                update_data[key] = value
        
        # Update widget in database
        result = supabase.table("widgets").update(update_data).eq("id", widget_id).execute()
        
        if result.data:
            logger.info(f"Updated widget {widget_id}")
            return result.data[0]
        else:
            logger.warning(f"No widget found with ID {widget_id} to update")
            return {}
            
    except Exception as e:
        logger.error(f"Error updating widget {widget_id}: {str(e)}")
        raise


def delete_widget(
    supabase: Client,
    widget_id: str
) -> bool:
    """
    Delete a widget from the database.
    
    Args:
        supabase: Supabase client instance
        widget_id: Widget identifier
        
    Returns:
        Boolean indicating if deletion was successful
        
    Raises:
        Exception: If widget deletion fails
    """
    try:
        # Delete widget from database
        result = supabase.table("widgets").delete().eq("id", widget_id).execute()
        
        if result.data:
            logger.info(f"Deleted widget {widget_id}")
            return True
        else:
            logger.warning(f"No widget found with ID {widget_id} to delete")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting widget {widget_id}: {str(e)}")
        raise


async def get_widget_specs(
    widget_id: str,
    supabase: Client
) -> Dict[str, Any]:
    """
    Get complete widget specifications including all fields from the database.
    
    Args:
        widget_id: Widget identifier
        supabase: Supabase client instance
        
    Returns:
        Dict containing complete widget specifications
        
    Raises:
        Exception: If widget not found or retrieval fails
    """
    try:
        # Get complete widget data from database
        result = supabase.table("widgets").select("*").eq("id", widget_id).single().execute()
        
        if not result.data:
            raise Exception(f"Widget {widget_id} not found")
        
        widget_specs = result.data
        
        logger.info(f"Successfully retrieved widget specs for {widget_id}")
        return widget_specs
        
    except Exception as e:
        logger.error(f"Error getting widget specs for {widget_id}: {str(e)}")
        raise
    
        # INSERT_YOUR_CODE
def get_widgets_from_dashboard_id(
    dashboard_id: str,
    supabase: Client
) -> List[Dict[str, Any]]:
    """
    Get all widgets associated with a given dashboard ID.

    Args:
        dashboard_id: Dashboard identifier
        supabase: Supabase client instance

    Returns:
        List of dicts, each containing widget data

    Raises:
        Exception: If retrieval fails
    """
    try:
        result = supabase.table("widgets").select("*").eq("dashboard_id", dashboard_id).execute()
        widgets = result.data or []
        logger.info(f"Retrieved {len(widgets)} widgets for dashboard {dashboard_id}")
        return widgets
    except Exception as e:
        logger.error(f"Error getting widgets for dashboard {dashboard_id}: {str(e)}")
        raise


def get_widget_from_widget_id(
    widget_id: str,
    supabase: Client
) -> Optional[Dict[str, Any]]:
    """
    Get a single widget by its widget ID.

    Args:
        widget_id: Widget identifier
        supabase: Supabase client instance

    Returns:
        Dict containing widget data, or None if not found

    Raises:
        Exception: If retrieval fails
    """
    try:
        result = supabase.table("widgets").select("*").eq("id", widget_id).single().execute()
        if not result.data:
            logger.warning(f"Widget {widget_id} not found")
            return None
        logger.info(f"Retrieved widget {widget_id}")
        return result.data
    except Exception as e:
        logger.error(f"Error getting widget {widget_id}: {str(e)}")
        raise
