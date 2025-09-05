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

## Dashboard Data Analysis


def get_available_data(dashboard_id: str) -> Dict[str, any]:
    """Get available data files and their schemas for a dashboard.

    Args:
        dashboard_id: Dashboard identifier

    Returns:
        Dict containing available files, schemas, and summary
    """
    try:
        supabase = _get_supabase_client()

        # Get all files associated with this dashboard
        response = (
            supabase.table("files")
            .select("*")
            .eq("dashboard_id", dashboard_id)
            .execute()
        )

        if not response.data:
            return {
                "available_files": [],
                "file_schemas": [],
                "data_summary": "No data files available for this dashboard.",
            }

        available_files = []
        file_schemas = []

        for file_record in response.data:
            file_id = file_record["id"]
            file_name = file_record.get("name", f"File {file_id}")
            available_files.append(file_id)

            try:
                # Get schema for this file
                schema = get_schema_from_file(file_id)
                file_schemas.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "schema": schema,
                        "file_type": file_record.get("type", "unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get schema for file {file_id}: {e}")
                file_schemas.append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "schema": None,
                        "error": str(e),
                    }
                )

        # Create summary of available data
        data_summary = _create_data_summary(file_schemas)

        return {
            "available_files": available_files,
            "file_schemas": file_schemas,
            "data_summary": data_summary,
        }

    except Exception as e:
        logger.error(f"Error getting available data: {e}")
        return {
            "available_files": [],
            "file_schemas": [],
            "data_summary": f"Error retrieving data: {str(e)}",
        }


def _create_data_summary(file_schemas: List[Dict]) -> str:
    """Create a human-readable summary of available data."""
    if not file_schemas:
        return "No data files available."

    summary_parts = []
    summary_parts.append(f"Available data files: {len(file_schemas)} file(s)")

    for file_info in file_schemas:
        file_name = file_info.get("file_name", "Unknown")
        file_id = file_info.get("file_id", "Unknown")

        if file_info.get("error"):
            summary_parts.append(f"- {file_name} (ID: {file_id}): Error loading schema")
            continue

        schema = file_info.get("schema")
        if not schema:
            summary_parts.append(f"- {file_name} (ID: {file_id}): No schema available")
            continue

        columns = schema.get("columns", [])
        total_rows = schema.get("total_rows", 0)

        column_summary = []
        for col in columns[:5]:  # Show first 5 columns
            col_name = col.get("name", "unknown")
            col_type = col.get("type", "unknown")
            column_summary.append(f"{col_name} ({col_type})")

        if len(columns) > 5:
            column_summary.append(f"... and {len(columns) - 5} more columns")

        summary_parts.append(
            f"- {file_name} (ID: {file_id}): {total_rows} rows, "
            f"columns: {', '.join(column_summary)}"
        )

    return "\n".join(summary_parts)


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
        widget_id = (
            widget_input.widget_id if widget_input.widget_id else str(uuid.uuid4())
        )

        # Prepare widget data from input model - aligned with new schema
        widget_data = {
            "id": widget_id,
            "dashboard_id": widget_input.dashboard_id,
            "title": widget_input.title,
            "type": widget_input.widget_type,
            "config": widget_input.config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_configured": widget_input.is_configured
            if widget_input.is_configured is not None
            else False,
        }

        # Add optional fields if provided (only fields that exist in the new schema)
        if widget_input.data is not None:
            widget_data["data"] = widget_input.data
        if widget_input.order is not None:
            widget_data["order"] = widget_input.order
        if widget_input.summary is not None:
            widget_data["summary"] = widget_input.summary

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
        # Add fields to update from input model (only fields that exist in the new schema)
        if update_input.data is not None:
            update_data["data"] = update_input.data
        if update_input.order is not None:
            update_data["order"] = update_input.order
        if update_input.is_configured is not None:
            update_data["is_configured"] = update_input.is_configured
        if update_input.summary is not None:
            update_data["summary"] = update_input.summary

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


def update_widgets_configuration_status(
    widget_ids: List[str], is_configured: bool = True
) -> Dict[str, Any]:
    """Update is_configured status for multiple widgets.

    Args:
        widget_ids: List of widget IDs to update
        is_configured: Configuration status to set (default True)

    Returns:
        Dict with update results

    Raises:
        Exception: If widget update fails
    """
    try:
        if not widget_ids:
            return {
                "success": True,
                "updated_count": 0,
                "message": "No widget IDs provided",
            }

        supabase = _get_supabase_client()

        # Update all widgets at once using in_ filter
        update_data = {
            "is_configured": is_configured,
            "updated_at": datetime.now().isoformat(),
        }

        result = (
            supabase.table("widgets")
            .update(update_data)
            .in_("id", widget_ids)
            .execute()
        )

        updated_count = len(result.data) if result.data else 0

        logger.info(
            f"Updated is_configured={is_configured} for {updated_count} widgets"
        )

        return {
            "success": True,
            "updated_count": updated_count,
            "requested_count": len(widget_ids),
            "message": f"Successfully updated configuration status for {updated_count} widgets",
        }

    except Exception as e:
        error_msg = (
            f"Error updating configuration status for widgets {widget_ids}: {str(e)}"
        )
        logger.error(error_msg)
        return {
            "success": False,
            "updated_count": 0,
            "message": error_msg,
            "error": str(e),
        }


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


## Reference Widget Operations


def fetch_reference_widget_details(widget_id: str) -> Dict[str, Any]:
    """Fetch widget details including config from database for reference purposes.

    This helper function is used by text block agents to fetch details of widgets
    they need to reference or explain. Follows CLAUDE.md guidelines by centralizing
    database operations in actions.

    Args:
        widget_id: Widget identifier to fetch

    Returns:
        Dict containing widget details with config, title, type, etc.

    Raises:
        Exception: If widget not found or fetch fails
    """
    try:
        supabase = _get_supabase_client()

        # Fetch widget details from database
        widget_result = (
            supabase.table("widgets")
            .select(
                "id, title, type, config, data, dashboard_id, summary, order, is_configured"
            )
            .eq("id", widget_id)
            .single()
            .execute()
        )

        if not widget_result.data:
            raise Exception(f"Reference widget {widget_id} not found")

        widget_data = widget_result.data

        logger.info(f"Successfully fetched reference widget details for {widget_id}")

        return {
            "widget_id": widget_data["id"],
            "title": widget_data.get("title", ""),
            "description": "",  # Description not in database schema - use empty string
            "widget_type": widget_data.get(
                "type", ""
            ),  # Field is 'type' not 'widget_type'
            "config": widget_data.get("config", {}),
            "data": widget_data.get("data", {}),
            "summary": widget_data.get("summary", ""),
            "order": widget_data.get("order", 0),
            "is_configured": widget_data.get("is_configured", False),
            "dashboard_id": widget_data.get("dashboard_id", ""),
        }

    except Exception as e:
        error_msg = (
            f"Failed to fetch reference widget details for {widget_id}: {str(e)}"
        )
        logger.error(error_msg)
        raise Exception(error_msg) from e


def fetch_multiple_reference_widget_details(
    widget_ids: List[str],
) -> List[Dict[str, Any]]:
    """Fetch details for multiple reference widgets from database.

    This helper function efficiently fetches details for multiple widgets
    that need to be referenced by text blocks.

    Args:
        widget_ids: List of widget identifiers to fetch

    Returns:
        List of dicts containing widget details, empty list if none found

    Raises:
        Exception: If database query fails
    """
    if not widget_ids:
        return []

    try:
        supabase = _get_supabase_client()

        # Fetch multiple widgets in a single query
        widgets_result = (
            supabase.table("widgets")
            .select(
                "id, title, type, config, data, dashboard_id, summary, order, is_configured"
            )
            .in_("id", widget_ids)
            .execute()
        )

        if not widgets_result.data:
            logger.warning(f"No reference widgets found for IDs: {widget_ids}")
            return []

        reference_widgets = []
        for widget_data in widgets_result.data:
            reference_widgets.append(
                {
                    "widget_id": widget_data["id"],
                    "title": widget_data.get("title", ""),
                    "description": "",  # Description not in database schema
                    "widget_type": widget_data.get("type", ""),
                    "config": widget_data.get("config", {}),
                    "data": widget_data.get("data", {}),
                    "summary": widget_data.get("summary", ""),
                    "order": widget_data.get("order", 0),
                    "is_configured": widget_data.get("is_configured", False),
                    "dashboard_id": widget_data.get("dashboard_id", ""),
                }
            )

        logger.info(
            f"Successfully fetched {len(reference_widgets)} reference widgets from {len(widget_ids)} requested"
        )
        return reference_widgets

    except Exception as e:
        error_msg = (
            f"Failed to fetch reference widget details for {widget_ids}: {str(e)}"
        )
        logger.error(error_msg)
        raise Exception(error_msg) from e


# Export all public functions
def get_dashboard_widgets_for_ordering(dashboard_id: str) -> List[Dict[str, Any]]:
    """Fetch existing dashboard widgets with only required columns for ordering.

    Args:
        dashboard_id: Dashboard identifier

    Returns:
        List of widget dictionaries with id, widget_type, order, and summary

    Raises:
        Exception: If database query fails
    """
    try:
        supabase = _get_supabase_client()

        # Fetch only the required columns for ordering
        # Note: Using explicit column selection without alias due to Supabase client issue
        response = (
            supabase.table("widgets")
            .select("id, type, order, summary")
            .eq("dashboard_id", dashboard_id)
            .order("order", desc=False)  # Order by current order
            .execute()
        )

        if response.data:
            # Transform the data to use widget_type key as expected by the ordering logic
            transformed_data = []
            for widget in response.data:
                transformed_widget = {
                    "id": widget["id"],
                    "widget_type": widget["type"],  # Rename type to widget_type
                    "order": widget["order"],
                    "summary": widget["summary"]
                }
                transformed_data.append(transformed_widget)
            
            logger.info(
                f"Retrieved {len(transformed_data)} widgets for ordering from dashboard {dashboard_id}"
            )
            return transformed_data
        else:
            logger.info(f"No widgets found for dashboard {dashboard_id}")
            return []

    except Exception as e:
        logger.error(f"Error fetching dashboard widgets for ordering: {e}")
        raise


def update_widgets_order_and_configuration(
    widget_updates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Update widget orders and mark them as configured using optimized bulk operations.

    Args:
        widget_updates: List of dictionaries with 'id', 'order', and 'is_configured' keys

    Returns:
        Dict with success status, updated_count, and message

    Raises:
        Exception: If database update fails
    """
    try:
        supabase = _get_supabase_client()
        
        if not widget_updates:
            return {
                "success": True,
                "updated_count": 0,
                "message": "No widgets to update"
            }

        # Validate and prepare updates
        valid_updates = []
        current_timestamp = datetime.now().isoformat()
        
        for update in widget_updates:
            widget_id = update.get("id")
            new_order = update.get("order")
            is_configured = update.get("is_configured", True)

            if not widget_id or new_order is None:
                logger.warning(f"Invalid update data: {update}")
                continue
                
            valid_updates.append({
                "id": widget_id,
                "order": new_order,
                "is_configured": is_configured,
                "updated_at": current_timestamp
            })

        if not valid_updates:
            return {
                "success": False,
                "updated_count": 0,
                "message": "No valid widget updates provided"
            }

        # Perform individual updates for each widget (safer than upsert for existing records)
        # This ensures we only update existing widgets and don't accidentally create new ones
        updated_count = 0
        
        for update in valid_updates:
            widget_id = update["id"]
            try:
                response = (
                    supabase.table("widgets")
                    .update({
                        "order": update["order"],
                        "is_configured": update["is_configured"],
                        "updated_at": update["updated_at"]
                    })
                    .eq("id", widget_id)
                    .execute()
                )
                
                if response.data:
                    updated_count += 1
                    logger.debug(f"Updated widget {widget_id} with order {update['order']}")
                else:
                    logger.warning(f"Widget {widget_id} not found for update")
                    
            except Exception as widget_error:
                logger.error(f"Failed to update widget {widget_id}: {str(widget_error)}")
                # Continue with other widgets instead of failing completely
                continue
        
        logger.info(
            f"Successfully updated {updated_count}/{len(valid_updates)} widgets with new orders and configuration status"
        )

        return {
            "success": True,
            "updated_count": updated_count,
            "message": f"Successfully updated {updated_count} widgets",
        }

    except Exception as e:
        error_msg = f"Error updating widget orders and configuration: {e}"
        logger.error(error_msg)
        return {"success": False, "updated_count": 0, "message": error_msg}


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
    "update_widgets_configuration_status",
    "delete_widget",
    "get_widget_specs",
    "get_widgets_from_dashboard_id",
    "get_widget_from_widget_id",
    # Reference widget operations
    "fetch_reference_widget_details",
    "fetch_multiple_reference_widget_details",
    # Widget ordering operations
    "get_dashboard_widgets_for_ordering",
    "update_widgets_order_and_configuration",
]
