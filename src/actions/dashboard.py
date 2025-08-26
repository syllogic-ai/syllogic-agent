"""Dashboard data management utilities for file operations.
Provides functions to fetch data, schema, and samples from stored files.
"""

import io
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from supabase import Client

from agent.models import CreateWidgetInput, UpdateWidgetInput, Widget
# Import config using the robust import utility to avoid circular dependencies
def _get_supabase_client():
    """Get Supabase client using robust import."""
    try:
        # Try direct import first
        from config import get_supabase_client
        return get_supabase_client()
    except ImportError:
        # Fallback to robust import from utils
        from .utils import import_config
        config_module = import_config()
        return config_module.get_supabase_client()

logger = logging.getLogger(__name__)

## Files


def get_data_from_file(file_id: str) -> pd.DataFrame:
    """Get complete data from a file using its storage path.

    Args:
        file_id: File identifier

    Returns:
        pandas.DataFrame containing the file data

    Raises:
        Exception: If file not found or data retrieval fails
    """
    try:
        supabase = _get_supabase_client()

        # Get file information from database
        file_info = (
            supabase.table("files")
            .select("storage_path, original_filename, mime_type")
            .eq("id", file_id)
            .single()
            .execute()
        )

        if not file_info.data:
            raise Exception(f"File {file_id} not found")

        storage_path = file_info.data["storage_path"]
        original_filename = file_info.data["original_filename"]
        file_info.data.get("mime_type")

        # Get base URL from environment variables
        supabase_storage_base_url = os.getenv("SUPABASE_STORAGE_BASE_URL")
        if not supabase_storage_base_url:
            # Construct from SUPABASE_URL if SUPABASE_STORAGE_BASE_URL not available
            supabase_url = os.getenv("SUPABASE_URL", "")
            if supabase_url:
                supabase_storage_base_url = f"{supabase_url}/storage/v1/object/public"
            else:
                raise Exception(
                    "SUPABASE_STORAGE_BASE_URL or SUPABASE_URL environment variable not set"
                )

        # Construct full URL
        file_url = f"{supabase_storage_base_url}/{storage_path}"

        # Download file content
        response = requests.get(file_url)
        response.raise_for_status()

        # Determine file type and read accordingly
        if original_filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(response.text))
        elif original_filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(response.content))
        elif original_filename.lower().endswith(".json"):
            df = pd.read_json(io.StringIO(response.text))
        else:
            # Default to CSV parsing
            df = pd.read_csv(io.StringIO(response.text))

        logger.info(
            f"Successfully loaded data from file {file_id}: {df.shape[0]} rows, {df.shape[1]} columns"
        )
        return df

    except Exception as e:
        logger.error(f"Error getting data from file {file_id}: {str(e)}")
        raise


def get_files_from_dashboard(dashboard_id: str) -> List[str]:
    """Get the list of file IDs associated with a dashboard.

    Args:
        dashboard_id: Dashboard identifier

    Returns:
        List of file IDs (strings) associated with the dashboard

    Raises:
        Exception: If dashboard not found or query fails
    """
    try:
        supabase = _get_supabase_client()

        # Query the files table directly using dashboard_id foreign key
        response = (
            supabase.table("files")
            .select("id")
            .eq("dashboard_id", dashboard_id)
            .execute()
        )
        if not response.data:
            logger.warning(f"No files found for dashboard {dashboard_id}")
            return []
        file_ids = [row["id"] for row in response.data if "id" in row]
        logger.info(f"Found {len(file_ids)} files for dashboard {dashboard_id}")
        return file_ids
    except Exception as e:
        logger.error(f"Error getting files from dashboard {dashboard_id}: {str(e)}")
        raise


def get_schema_from_file(file_id: str) -> Dict[str, Any]:
    """Get schema information from a file.

    Args:
        file_id: File identifier

    Returns:
        Dict containing schema information with columns and their types

    Raises:
        Exception: If file not found or schema extraction fails
    """
    try:
        # Get the dataframe
        df = get_data_from_file(file_id)

        # Extract schema information
        schema = {
            "columns": [],
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }

        for column in df.columns:
            column_info = {
                "name": column,
                "type": str(df[column].dtype),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique()),
            }

            # Add sample values for better understanding
            non_null_values = df[column].dropna()
            if len(non_null_values) > 0:
                column_info["sample_values"] = non_null_values.head(3).tolist()

            schema["columns"].append(column_info)

        logger.info(
            f"Successfully extracted schema from file {file_id}: {len(schema['columns'])} columns"
        )
        return schema

    except Exception as e:
        logger.error(f"Error getting schema from file {file_id}: {str(e)}")
        raise


def get_sample_from_file(file_id: str, num_rows: int = 3) -> Dict[str, Any]:
    """Get sample data from a file (header + specified number of rows).

    Args:
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
        df = get_data_from_file(file_id)

        # Get sample data (header + requested number of rows)
        sample_df = df.head(num_rows)

        # Convert to a format that's easy to work with
        sample_data = {
            "headers": list(df.columns),
            "rows": sample_df.values.tolist(),
            "total_rows_in_file": len(df),
            "sample_rows_returned": len(sample_df),
            "requested_rows": num_rows,
        }

        logger.info(
            f"Successfully extracted sample from file {file_id}: {len(sample_data['rows'])} rows returned"
        )
        return sample_data

    except Exception as e:
        logger.error(f"Error getting sample from file {file_id}: {str(e)}")
        raise


## Widgets


def create_widget(widget_input: CreateWidgetInput) -> Widget:
    """Create a new widget in the dashboard.

    Args:
        widget_input: CreateWidgetInput containing widget creation data

    Returns:
        Widget model containing the created widget data

    Raises:
        Exception: If widget creation fails
    """
    try:
        supabase = _get_supabase_client()
        # Use provided widget_id if available, otherwise generate new UUID
        widget_id = widget_input.widget_id if widget_input.widget_id else str(uuid.uuid4())

        # Prepare widget data from input model
        widget_data = {
            "id": widget_id,
            "dashboard_id": widget_input.dashboard_id,
            "title": widget_input.title,
            "type": widget_input.widget_type,
            "config": widget_input.config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Add optional fields if provided
        # NOTE: description field is not in the widgets table schema, so we skip it
        # if widget_input.description is not None:
        #     widget_data["description"] = widget_input.description
        if widget_input.data is not None:
            widget_data["data"] = widget_input.data
        if widget_input.sql is not None:
            widget_data["sql"] = widget_input.sql
        if widget_input.layout is not None:
            widget_data["layout"] = widget_input.layout
        if widget_input.chat_id is not None:
            widget_data["chat_id"] = widget_input.chat_id
        if widget_input.order is not None:
            widget_data["order"] = widget_input.order

        # Insert widget into database
        result = supabase.table("widgets").insert(widget_data).execute()

        if result.data:
            logger.info(
                f"Created widget {widget_id} in dashboard {widget_input.dashboard_id}"
            )
            return Widget(**result.data[0])
        else:
            raise Exception("Failed to create widget: No data returned")

    except Exception as e:
        logger.error(
            f"Error creating widget in dashboard {widget_input.dashboard_id}: {str(e)}"
        )
        raise


def update_widget(update_input: UpdateWidgetInput) -> Widget:
    """Update an existing widget with the provided fields.

    Args:
        update_input: UpdateWidgetInput containing update data

    Returns:
        Widget model containing the updated widget data

    Raises:
        Exception: If widget update fails
    """
    try:
        supabase = _get_supabase_client()

        # Prepare update data
        update_data = {"updated_at": datetime.now().isoformat()}

        # Add fields to update from input model
        if update_input.title is not None:
            update_data["title"] = update_input.title
        if update_input.widget_type is not None:
            update_data["type"] = update_input.widget_type
        if update_input.config is not None:
            update_data["config"] = update_input.config
        if update_input.data is not None:
            update_data["data"] = update_input.data
        if update_input.sql is not None:
            update_data["sql"] = update_input.sql
        if update_input.layout is not None:
            update_data["layout"] = update_input.layout
        if update_input.order is not None:
            update_data["order"] = update_input.order
        if update_input.is_configured is not None:
            update_data["is_configured"] = update_input.is_configured
        if update_input.cache_key is not None:
            update_data["cache_key"] = update_input.cache_key

        # Update widget in database
        result = (
            supabase.table("widgets")
            .update(update_data)
            .eq("id", update_input.widget_id)
            .execute()
        )

        if result.data:
            logger.info(f"Updated widget {update_input.widget_id}")
            return Widget(**result.data[0])
        else:
            logger.warning(
                f"No widget found with ID {update_input.widget_id} to update"
            )
            # Return a minimal widget object for backwards compatibility
            return Widget(
                id=update_input.widget_id, dashboard_id="", title="", type="", config={}
            )

    except Exception as e:
        logger.error(f"Error updating widget {update_input.widget_id}: {str(e)}")
        raise


def delete_widget(widget_id: str) -> bool:
    """Delete a widget from the database.

    Args:
        widget_id: Widget identifier

    Returns:
        Boolean indicating if deletion was successful

    Raises:
        Exception: If widget deletion fails
    """
    try:
        supabase = _get_supabase_client()

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


async def get_widget_specs(widget_id: str) -> Widget:
    """Get complete widget specifications including all fields from the database.

    Args:
        widget_id: Widget identifier
        supabase: Supabase client instance

    Returns:
        Widget model containing complete widget specifications

    Raises:
        Exception: If widget not found or retrieval fails
    """
    try:
        supabase = _get_supabase_client()

        # Get complete widget data from database
        result = (
            supabase.table("widgets").select("*").eq("id", widget_id).single().execute()
        )

        if not result.data:
            raise Exception(f"Widget {widget_id} not found")

        widget = Widget(**result.data)

        logger.info(f"Successfully retrieved widget specs for {widget_id}")
        return widget

    except Exception as e:
        logger.error(f"Error getting widget specs for {widget_id}: {str(e)}")
        raise

        # INSERT_YOUR_CODE


def get_widgets_from_dashboard_id(dashboard_id: str) -> List[Widget]:
    """Get all widgets associated with a given dashboard ID.

    Args:
        dashboard_id: Dashboard identifier
        supabase: Supabase client instance

    Returns:
        List of Widget models, each containing widget data

    Raises:
        Exception: If retrieval fails
    """
    try:
        supabase = _get_supabase_client()

        result = (
            supabase.table("widgets")
            .select("*")
            .eq("dashboard_id", dashboard_id)
            .execute()
        )
        widgets = [Widget(**widget) for widget in result.data] if result.data else []
        logger.info(f"Retrieved {len(widgets)} widgets for dashboard {dashboard_id}")
        return widgets
    except Exception as e:
        logger.error(f"Error getting widgets for dashboard {dashboard_id}: {str(e)}")
        raise


def get_widget_from_widget_id(widget_id: str) -> Optional[Widget]:
    """Get a single widget by its widget ID.

    Args:
        widget_id: Widget identifier
        supabase: Supabase client instance

    Returns:
        Widget model containing widget data, or None if not found

    Raises:
        Exception: If retrieval fails
    """
    try:
        supabase = _get_supabase_client()

        result = (
            supabase.table("widgets").select("*").eq("id", widget_id).single().execute()
        )
        if not result.data:
            logger.warning(f"Widget {widget_id} not found")
            return None
        logger.info(f"Retrieved widget {widget_id}")
        return Widget(**result.data)
    except Exception as e:
        logger.error(f"Error getting widget {widget_id}: {str(e)}")
        raise


def get_schemas_from_files(file_ids: List[str]) -> Dict[str, Any]:
    """Get schemas from multiple files.

    Args:
        file_ids: List of file identifiers

    Returns:
        Dict mapping file_id to schema information

    Raises:
        Exception: If schema retrieval fails for any file
    """
    try:
        schemas = {}
        for file_id in file_ids:
            try:
                schema = get_schema_from_file(file_id)
                # Convert to FileSchema format expected by models
                from agent.models import ColumnInfo, FileSchema

                columns = [
                    ColumnInfo(
                        name=col["name"],
                        type=col["type"],
                        null_count=col["null_count"],
                        unique_count=col["unique_count"],
                        sample_values=col.get("sample_values", []),
                    )
                    for col in schema["columns"]
                ]

                file_schema = FileSchema(
                    file_id=file_id,
                    columns=columns,
                    total_rows=schema["total_rows"],
                    total_columns=schema["total_columns"],
                )
                schemas[file_id] = file_schema

            except Exception as e:
                logger.warning(f"Could not get schema for file {file_id}: {str(e)}")
                continue

        logger.info(
            f"Retrieved schemas for {len(schemas)} out of {len(file_ids)} files"
        )
        return schemas

    except Exception as e:
        logger.error(f"Error getting schemas from files: {str(e)}")
        raise


def get_sample_data_from_files(file_ids: List[str]) -> Dict[str, Any]:
    """Get sample data from multiple files.

    Args:
        file_ids: List of file identifiers

    Returns:
        Dict mapping file_id to sample data

    Raises:
        Exception: If sample data retrieval fails for any file
    """
    try:
        samples = {}
        for file_id in file_ids:
            try:
                sample_data = get_sample_from_file(file_id, num_rows=3)
                # Convert to FileSampleData format expected by models
                from agent.models import FileSampleData

                file_sample = FileSampleData(
                    file_id=file_id,
                    headers=sample_data["headers"],
                    rows=sample_data["rows"],
                    total_rows_in_file=sample_data["total_rows_in_file"],
                    sample_rows_returned=sample_data["sample_rows_returned"],
                )
                samples[file_id] = file_sample

            except Exception as e:
                logger.warning(
                    f"Could not get sample data for file {file_id}: {str(e)}"
                )
                continue

        logger.info(
            f"Retrieved sample data for {len(samples)} out of {len(file_ids)} files"
        )
        return samples

    except Exception as e:
        logger.error(f"Error getting sample data from files: {str(e)}")
        raise


# Export all public functions
__all__ = [
    # File operations
    "get_data_from_file",
    "get_files_from_dashboard",
    "get_schema_from_file",
    "get_sample_from_file",
    "get_schemas_from_files",
    "get_sample_data_from_files",
    # Widget operations
    "create_widget",
    "update_widget",
    "delete_widget",
    "get_widget_specs",
    "get_widgets_from_dashboard_id",
    "get_widget_from_widget_id",
]
