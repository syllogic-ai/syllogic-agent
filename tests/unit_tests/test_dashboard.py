"""
Unit tests for dashboard.py functions with centralized Supabase client.
"""

import os
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add the src directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src")
)

from actions.dashboard import (
    create_widget,
    delete_widget,
    get_data_from_file,
    get_sample_from_file,
    get_schema_from_file,
    get_widget_from_widget_id,
    get_widget_specs,
    get_widgets_from_dashboard_id,
    update_widget,
)
from agent.models import CreateWidgetInput, UpdateWidgetInput, Widget


class TestGetDataFromFile:
    """Test cases for get_data_from_file function."""

    @patch("requests.get")
    @patch("actions.dashboard._get_supabase_client")
    @patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"})
    def test_get_data_csv_success(
        self,
        mock_get_supabase_client,
        mock_requests,
        sample_file_data,
        sample_csv_content,
    ):
        """Test successfully getting data from CSV file."""
        # Mock Supabase client
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        # Mock database response
        mock_result = Mock()
        mock_result.data = sample_file_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_csv_content
        mock_response.raise_for_status.return_value = None
        mock_requests.return_value = mock_response

        result = get_data_from_file("file-123")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["date", "value", "category", "percentage"]
        assert result.iloc[0]["value"] == 100

    @patch("actions.dashboard._get_supabase_client")
    def test_get_data_file_not_found(self, mock_get_supabase_client):
        """Test error when file is not found in database."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        with pytest.raises(Exception, match="File file-123 not found"):
            get_data_from_file("file-123")

    @patch("actions.dashboard._get_supabase_client")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_data_missing_env_vars(
        self, mock_get_supabase_client, sample_file_data
    ):
        """Test error when environment variables are missing."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = sample_file_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        with pytest.raises(
            Exception,
            match="SUPABASE_STORAGE_BASE_URL or SUPABASE_URL environment variable not set",
        ):
            get_data_from_file("file-123")

    @patch("requests.get")
    @patch("actions.dashboard._get_supabase_client")
    @patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"})
    def test_get_data_http_error(
        self, mock_get_supabase_client, mock_requests, sample_file_data
    ):
        """Test error when HTTP request fails."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = sample_file_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_requests.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 404"):
            get_data_from_file("file-123")


class TestGetSchemaFromFile:
    """Test cases for get_schema_from_file function."""

    @patch("actions.dashboard.get_data_from_file")
    def test_get_schema_success(self, mock_get_data, sample_dataframe):
        """Test successfully getting schema from file."""
        mock_get_data.return_value = sample_dataframe

        result = get_schema_from_file("file-123")

        assert result["total_rows"] == 3
        assert result["total_columns"] == 4
        assert len(result["columns"]) == 4

        # Check first column details
        date_col = result["columns"][0]
        assert date_col["name"] == "date"
        assert date_col["type"] == "object"
        assert date_col["null_count"] == 0
        assert date_col["unique_count"] == 3
        assert "sample_values" in date_col

    @patch("actions.dashboard.get_data_from_file")
    def test_get_schema_with_nulls(self, mock_get_data):
        """Test getting schema from file with null values."""
        df_with_nulls = pd.DataFrame({"col1": [1, 2, None], "col2": ["a", None, "c"]})
        mock_get_data.return_value = df_with_nulls

        result = get_schema_from_file("file-123")

        col1_info = next(col for col in result["columns"] if col["name"] == "col1")
        assert col1_info["null_count"] == 1
        assert col1_info["unique_count"] == 2

    @patch("actions.dashboard.get_data_from_file")
    def test_get_schema_error_propagation(self, mock_get_data):
        """Test that errors from get_data_from_file are propagated."""
        mock_get_data.side_effect = Exception("File error")

        with pytest.raises(Exception, match="File error"):
            get_schema_from_file("file-123")


class TestGetSampleFromFile:
    """Test cases for get_sample_from_file function."""

    @patch("actions.dashboard.get_data_from_file")
    def test_get_sample_default_rows(self, mock_get_data, sample_dataframe):
        """Test getting sample with default number of rows."""
        mock_get_data.return_value = sample_dataframe

        result = get_sample_from_file("file-123")

        assert result["headers"] == ["date", "value", "category", "percentage"]
        assert len(result["rows"]) == 3  # Default is 3 rows
        assert result["total_rows_in_file"] == 3
        assert result["sample_rows_returned"] == 3
        assert result["requested_rows"] == 3

    @patch("actions.dashboard.get_data_from_file")
    def test_get_sample_specific_rows(self, mock_get_data, sample_dataframe):
        """Test getting sample with specific number of rows."""
        mock_get_data.return_value = sample_dataframe

        result = get_sample_from_file("file-123", num_rows=2)

        assert len(result["rows"]) == 2
        assert result["requested_rows"] == 2

    @patch("actions.dashboard.get_data_from_file")
    def test_get_sample_more_rows_than_available(self, mock_get_data):
        """Test getting sample when requesting more rows than available."""
        small_df = pd.DataFrame({"col1": [1, 2]})
        mock_get_data.return_value = small_df

        result = get_sample_from_file("file-123", num_rows=5)

        assert len(result["rows"]) == 2  # Only 2 rows available
        assert result["sample_rows_returned"] == 2
        assert result["requested_rows"] == 5


class TestCreateWidget:
    """Test cases for create_widget function."""

    @patch("uuid.uuid4")
    @patch("actions.dashboard._get_supabase_client")
    def test_create_widget_success(
        self, mock_get_supabase_client, mock_uuid, sample_widget_data
    ):
        """Test successfully creating a widget."""
        mock_uuid.return_value = "widget-123"
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = [sample_widget_data]
        mock_supabase.table().insert().execute.return_value = mock_result

        widget_input = CreateWidgetInput(
            dashboard_id="dashboard-789",
            title="Test Widget",
            widget_type="chart",
            config={"chart_type": "bar"},
        )
        result = create_widget(widget_input)

        assert isinstance(result, Widget)
        assert result.id == "widget-123"
        assert result.title == "Test Widget"

        # Verify insert was called with correct data
        insert_call = mock_supabase.table().insert.call_args[0][0]
        assert insert_call["id"] == "widget-123"
        assert insert_call["dashboard_id"] == "dashboard-789"
        assert insert_call["config"] == {"chart_type": "bar"}

    @patch("actions.dashboard._get_supabase_client")
    def test_create_widget_with_optional_fields(
        self, mock_get_supabase_client, sample_widget_data
    ):
        """Test creating widget with optional fields."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = [sample_widget_data]
        mock_supabase.table().insert().execute.return_value = mock_result

        widget_input = CreateWidgetInput(
            dashboard_id="dashboard-789",
            title="Test Widget",
            widget_type="chart",
            config={"chart_type": "bar"},
            data={"data": "test"},
            chat_id="chat-123",
            order=1,
            summary="Test widget summary",
        )
        create_widget(widget_input)

        insert_call = mock_supabase.table().insert.call_args[0][0]
        assert insert_call["data"] == {"data": "test"}
        assert insert_call["chat_id"] == "chat-123"
        assert insert_call["order"] == 1
        assert insert_call["summary"] == "Test widget summary"

    @patch("actions.dashboard._get_supabase_client")
    def test_create_widget_insert_fails(self, mock_get_supabase_client):
        """Test error when widget insert fails."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().insert().execute.return_value = mock_result

        widget_input = CreateWidgetInput(
            dashboard_id="dashboard-789", title="Test", widget_type="chart", config={}
        )
        with pytest.raises(Exception, match="Failed to create widget"):
            create_widget(widget_input)


class TestUpdateWidget:
    """Test cases for update_widget function."""

    @patch("actions.dashboard._get_supabase_client")
    def test_update_widget_success(self, mock_get_supabase_client, sample_widget_data):
        """Test successfully updating a widget."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = [sample_widget_data]
        mock_supabase.table().update().eq().execute.return_value = mock_result

        update_input = UpdateWidgetInput(
            widget_id="widget-123", title="Updated Title", config={"new_config": True}
        )
        result = update_widget(update_input)

        assert isinstance(result, Widget)
        assert result.id == "widget-123"

        # Verify update call
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["title"] == "Updated Title"
        assert update_call["config"] == {"new_config": True}
        assert "updated_at" in update_call

    @patch("actions.dashboard._get_supabase_client")
    def test_update_widget_not_found(self, mock_get_supabase_client):
        """Test updating widget that doesn't exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().update().eq().execute.return_value = mock_result

        update_input = UpdateWidgetInput(widget_id="widget-123", title="Updated")
        result = update_widget(update_input)

        assert isinstance(result, Widget)
        assert result.id == "widget-123"


class TestDeleteWidget:
    """Test cases for delete_widget function."""

    @patch("actions.dashboard._get_supabase_client")
    def test_delete_widget_success(self, mock_get_supabase_client):
        """Test successfully deleting a widget."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = [{"id": "widget-123"}]
        mock_supabase.table().delete().eq().execute.return_value = mock_result

        result = delete_widget("widget-123")

        assert result is True

    @patch("actions.dashboard._get_supabase_client")
    def test_delete_widget_not_found(self, mock_get_supabase_client):
        """Test deleting widget that doesn't exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().delete().eq().execute.return_value = mock_result

        result = delete_widget("widget-123")

        assert result is False


class TestGetWidgetSpecs:
    """Test cases for get_widget_specs function."""

    @patch("actions.dashboard._get_supabase_client")
    @pytest.mark.asyncio
    async def test_get_widget_specs_success(
        self, mock_get_supabase_client, sample_widget_data
    ):
        """Test successfully getting widget specifications."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = sample_widget_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        result = await get_widget_specs("widget-123")

        assert isinstance(result, Widget)
        assert result.id == "widget-123"
        assert result.title == "Test Widget"

        # Verify database call
        mock_supabase.table.assert_called_with("widgets")

    @patch("actions.dashboard._get_supabase_client")
    @pytest.mark.asyncio
    async def test_get_widget_specs_not_found(self, mock_get_supabase_client):
        """Test error when widget is not found."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        with pytest.raises(Exception, match="Widget widget-123 not found"):
            await get_widget_specs("widget-123")


class TestGetWidgetsFromDashboardId:
    """Test cases for get_widgets_from_dashboard_id function."""

    @patch("actions.dashboard._get_supabase_client")
    def test_get_widgets_success(self, mock_get_supabase_client, sample_widget_data):
        """Test successfully getting widgets from dashboard ID."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = [sample_widget_data, sample_widget_data]
        mock_supabase.table().select().eq().execute.return_value = mock_result

        result = get_widgets_from_dashboard_id("dashboard-123")

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(widget, Widget) for widget in result)
        assert result[0].id == "widget-123"
        assert result[0].dashboard_id == "dashboard-789"

        # Verify database call
        mock_supabase.table.assert_called_with("widgets")
        mock_supabase.table().select.assert_called_with("*")
        mock_supabase.table().select().eq.assert_called_with(
            "dashboard_id", "dashboard-123"
        )

    @patch("actions.dashboard._get_supabase_client")
    def test_get_widgets_empty_result(self, mock_get_supabase_client):
        """Test getting widgets when none exist for dashboard."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().execute.return_value = mock_result

        result = get_widgets_from_dashboard_id("dashboard-123")

        assert isinstance(result, list)
        assert len(result) == 0


class TestGetWidgetFromWidgetId:
    """Test cases for get_widget_from_widget_id function."""

    @patch("actions.dashboard._get_supabase_client")
    def test_get_widget_success(self, mock_get_supabase_client, sample_widget_data):
        """Test successfully getting widget by widget ID."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = sample_widget_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        result = get_widget_from_widget_id("widget-123")

        assert result is not None
        assert isinstance(result, Widget)
        assert result.id == "widget-123"
        assert result.title == "Test Widget"
        assert result.dashboard_id == "dashboard-789"

        # Verify database call
        mock_supabase.table.assert_called_with("widgets")
        mock_supabase.table().select.assert_called_with("*")
        mock_supabase.table().select().eq.assert_called_with("id", "widget-123")

    @patch("actions.dashboard._get_supabase_client")
    def test_get_widget_not_found(self, mock_get_supabase_client):
        """Test getting widget that doesn't exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase

        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result

        result = get_widget_from_widget_id("widget-123")

        assert result is None
