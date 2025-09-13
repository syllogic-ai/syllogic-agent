"""Unit tests for widget cleanup functionality."""

import pytest
from unittest.mock import Mock, patch
from actions.widget_cleanup import (
    cleanup_unconfigured_widgets,
    cleanup_unconfigured_widgets_by_dashboard,
    get_unconfigured_widgets_count
)


class TestCleanupUnconfiguredWidgets:
    """Test cases for cleanup_unconfigured_widgets function."""

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_cleanup_no_unconfigured_widgets(self, mock_get_supabase_client):
        """Test cleanup when no unconfigured widgets exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        # Mock empty result
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table().select().eq().execute.return_value = mock_result
        
        result = cleanup_unconfigured_widgets()
        
        assert result["success"] is True
        assert result["deleted_count"] == 0
        assert result["deleted_widgets"] == []
        assert "No unconfigured widgets found" in result["message"]

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_cleanup_with_unconfigured_widgets(self, mock_get_supabase_client):
        """Test cleanup when unconfigured widgets exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        # Mock unconfigured widgets found
        unconfigured_widgets = [
            {"id": "widget-1", "title": "Test Widget 1", "type": "chart", "dashboard_id": "dashboard-1"},
            {"id": "widget-2", "title": "Test Widget 2", "type": "kpi", "dashboard_id": "dashboard-1"}
        ]
        
        mock_select_result = Mock()
        mock_select_result.data = unconfigured_widgets
        mock_supabase.table().select().eq().execute.return_value = mock_select_result
        
        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.data = unconfigured_widgets
        mock_supabase.table().delete().eq().execute.return_value = mock_delete_result
        
        result = cleanup_unconfigured_widgets()
        
        assert result["success"] is True
        assert result["deleted_count"] == 2
        assert result["deleted_widgets"] == ["widget-1", "widget-2"]
        assert "Successfully cleaned up 2 unconfigured widgets" in result["message"]

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_cleanup_database_error(self, mock_get_supabase_client):
        """Test cleanup when database operation fails."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        mock_supabase.table().select().eq().execute.side_effect = Exception("Database connection failed")
        
        result = cleanup_unconfigured_widgets()
        
        assert result["success"] is False
        assert result["deleted_count"] == 0
        assert result["deleted_widgets"] == []
        assert "Database connection failed" in result["error"]


class TestCleanupUnconfiguredWidgetsByDashboard:
    """Test cases for cleanup_unconfigured_widgets_by_dashboard function."""

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_cleanup_by_dashboard_no_widgets(self, mock_get_supabase_client):
        """Test cleanup for specific dashboard with no unconfigured widgets."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table().select().eq().eq().execute.return_value = mock_result
        
        result = cleanup_unconfigured_widgets_by_dashboard("dashboard-1")
        
        assert result["success"] is True
        assert result["deleted_count"] == 0
        assert "No unconfigured widgets found for dashboard dashboard-1" in result["message"]

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_cleanup_by_dashboard_with_widgets(self, mock_get_supabase_client):
        """Test cleanup for specific dashboard with unconfigured widgets."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        unconfigured_widgets = [
            {"id": "widget-1", "title": "Test Widget 1", "type": "chart"}
        ]
        
        mock_select_result = Mock()
        mock_select_result.data = unconfigured_widgets
        mock_supabase.table().select().eq().eq().execute.return_value = mock_select_result
        
        mock_delete_result = Mock()
        mock_delete_result.data = unconfigured_widgets
        mock_supabase.table().delete().eq().eq().execute.return_value = mock_delete_result
        
        result = cleanup_unconfigured_widgets_by_dashboard("dashboard-1")
        
        assert result["success"] is True
        assert result["deleted_count"] == 1
        assert result["deleted_widgets"] == ["widget-1"]
        assert "Successfully cleaned up 1 unconfigured widgets for dashboard dashboard-1" in result["message"]


class TestGetUnconfiguredWidgetsCount:
    """Test cases for get_unconfigured_widgets_count function."""

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_get_count_no_widgets(self, mock_get_supabase_client):
        """Test getting count when no unconfigured widgets exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        mock_result = Mock()
        mock_result.data = []
        mock_supabase.table().select().eq().execute.return_value = mock_result
        
        result = get_unconfigured_widgets_count()
        
        assert result["count"] == 0
        assert result["widgets"] == []
        assert "No unconfigured widgets found" in result["message"]

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_get_count_with_widgets(self, mock_get_supabase_client):
        """Test getting count when unconfigured widgets exist."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        
        unconfigured_widgets = [
            {"id": "widget-1", "title": "Test Widget 1", "type": "chart", "dashboard_id": "dashboard-1", "created_at": "2024-01-01T00:00:00Z"}
        ]
        
        mock_result = Mock()
        mock_result.data = unconfigured_widgets
        mock_supabase.table().select().eq().execute.return_value = mock_result
        
        result = get_unconfigured_widgets_count()
        
        assert result["count"] == 1
        assert result["widgets"] == unconfigured_widgets
        assert "Found 1 unconfigured widgets" in result["message"]

    @patch("actions.widget_cleanup.get_supabase_client")
    def test_get_count_database_error(self, mock_get_supabase_client):
        """Test getting count when database operation fails."""
        mock_supabase = Mock()
        mock_get_supabase_client.return_value = mock_supabase
        mock_supabase.table().select().eq().execute.side_effect = Exception("Database error")
        
        result = get_unconfigured_widgets_count()
        
        assert result["count"] == 0
        assert result["widgets"] == []
        assert "Database error" in result["error"]
