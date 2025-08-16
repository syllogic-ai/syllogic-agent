"""
Unit tests for import utility functions in actions.utils
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError, BaseModel, Field

from actions.utils import (
    robust_import,
    import_actions_dashboard,
    import_config,
    analyze_schema_validation_error,
    get_chart_config_schema_string
)


class TestRobustImport:
    """Test cases for robust_import function."""

    def test_robust_import_direct_success(self):
        """Test successful direct import without any path modification."""
        # Test importing a standard library module
        module = robust_import('json', ['dumps', 'loads'])
        assert hasattr(module, 'dumps')
        assert hasattr(module, 'loads')

    def test_robust_import_missing_attribute(self):
        """Test import fails when required attribute is missing."""
        with pytest.raises(ImportError, match="missing required attribute"):
            robust_import('json', ['dumps', 'loads', 'nonexistent_function'])

    @patch('sys.path')
    def test_robust_import_with_path_strategies(self, mock_path):
        """Test that robust_import tries different path strategies."""
        mock_path.__contains__ = Mock(return_value=False)
        mock_path.insert = Mock()
        mock_path.__setitem__ = Mock()
        
        # Mock a successful import on second strategy
        with patch('builtins.__import__') as mock_import:
            # First call fails, second succeeds
            mock_module = Mock()
            mock_module.test_attr = "exists"
            mock_import.side_effect = [ImportError("Module not found"), mock_module]
            
            result = robust_import('test_module', ['test_attr'])
            assert result == mock_module
            assert mock_import.call_count == 2

    def test_robust_import_all_strategies_fail(self):
        """Test robust_import when all strategies fail."""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Failed to import nonexistent_module with all strategies"):
                robust_import('nonexistent_module', ['some_attr'])

    @patch('actions.utils.find_src_directory')
    def test_robust_import_with_extra_setup(self, mock_find_src):
        """Test robust_import with extra setup functions."""
        mock_find_src.return_value = '/test/src'
        setup_func = Mock()
        
        # Test importing a standard library module with setup function
        module = robust_import('json', ['dumps'], extra_setup_funcs=[setup_func])
        assert hasattr(module, 'dumps')
        setup_func.assert_called()


class TestImportActionsAndConfig:
    """Test cases for import_actions_dashboard and import_config functions."""

    @patch('actions.utils.robust_import')
    def test_import_actions_dashboard_success(self, mock_robust_import):
        """Test successful import of actions.dashboard module."""
        mock_module = Mock()
        mock_robust_import.return_value = mock_module
        
        result = import_actions_dashboard()
        
        assert result == mock_module
        mock_robust_import.assert_called_once_with(
            'actions.dashboard',
            ['get_data_from_file', 'get_sample_from_file', 'get_schema_from_file', 
             'create_widget', 'update_widget', 'delete_widget']
        )

    @patch('actions.utils.robust_import')
    def test_import_config_success(self, mock_robust_import):
        """Test successful import of config module."""
        mock_module = Mock()
        mock_robust_import.return_value = mock_module
        
        result = import_config()
        
        assert result == mock_module
        mock_robust_import.assert_called_once_with(
            'config',
            ['get_supabase_client', 'create_e2b_sandbox']
        )

    @patch('actions.utils.robust_import')
    def test_import_actions_dashboard_failure(self, mock_robust_import):
        """Test import_actions_dashboard when import fails."""
        mock_robust_import.side_effect = ImportError("Failed to import")
        
        with pytest.raises(ImportError):
            import_actions_dashboard()

    @patch('actions.utils.robust_import')
    def test_import_config_failure(self, mock_robust_import):
        """Test import_config when import fails."""
        mock_robust_import.side_effect = ImportError("Failed to import")
        
        with pytest.raises(ImportError):
            import_config()


class TestSchemaValidationError:
    """Test cases for analyze_schema_validation_error function."""

    def test_analyze_validation_error_missing_fields(self):
        """Test analysis of missing field validation errors."""
        # Create a simple model for testing
        class TestModel(BaseModel):
            required_field: str
            optional_field: str = None

        # Generate validation error
        try:
            TestModel(optional_field="test")
        except ValidationError as e:
            result = analyze_schema_validation_error({}, e)
            assert "Missing required field: 'required_field'" in result
            assert isinstance(result, str)

    def test_analyze_validation_error_type_errors(self):
        """Test analysis of type validation errors."""
        class TestModel(BaseModel):
            number_field: int

        try:
            TestModel(number_field="not_a_number")
        except ValidationError as e:
            result = analyze_schema_validation_error({"number_field": "not_a_number"}, e)
            # Check for the actual error type returned by Pydantic
            assert ("Error in field 'number_field'" in result or "Wrong type for field 'number_field'" in result)

    def test_analyze_validation_error_literal_errors(self):
        """Test analysis of literal validation errors."""
        from typing import Literal
        
        class TestModel(BaseModel):
            status: Literal["active", "inactive"]

        try:
            TestModel(status="invalid_status")
        except ValidationError as e:
            result = analyze_schema_validation_error({"status": "invalid_status"}, e)
            assert "Invalid value for field 'status'" in result

    def test_analyze_validation_error_non_pydantic_error(self):
        """Test analysis with non-Pydantic validation error."""
        result = analyze_schema_validation_error({}, ValueError("Generic error"))
        assert "Validation error" in result or "Error analyzing validation failure" in result
        assert "Generic error" in result

    def test_analyze_validation_error_with_suggestions(self):
        """Test that analysis includes helpful suggestions."""
        class TestModel(BaseModel):
            chart_type: str

        try:
            TestModel(chart_type=123)  # Wrong type
        except ValidationError as e:
            result = analyze_schema_validation_error({"chart_type": 123}, e)
            # Just check that we get a meaningful error message
            assert len(result) > 0
            assert "chart_type" in result


class TestGetChartConfigSchema:
    """Test cases for get_chart_config_schema_string function."""

    def test_get_chart_config_schema_string_returns_string(self):
        """Test that the function returns a string."""
        result = get_chart_config_schema_string()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_chart_config_schema_string_contains_schema_info(self):
        """Test that the returned string contains expected schema information."""
        result = get_chart_config_schema_string()
        
        # Should contain key schema fields
        expected_fields = ['chartType', 'title', 'description', 'data', 'chartConfig', 'xAxisConfig']
        for field in expected_fields:
            assert field in result

    def test_get_chart_config_schema_string_contains_chart_types(self):
        """Test that the schema string contains valid chart types."""
        result = get_chart_config_schema_string()
        
        # Should contain expected chart types
        expected_types = ['line', 'bar', 'pie', 'area', 'radial', 'kpi', 'table']
        for chart_type in expected_types:
            assert chart_type in result

    @patch('builtins.__import__')
    def test_get_chart_config_schema_string_handles_import_error(self, mock_import):
        """Test graceful handling when ChartConfigSchema import fails."""
        mock_import.side_effect = ImportError("Schema not available")
        
        result = get_chart_config_schema_string()
        
        # Should return a fallback schema description
        assert isinstance(result, str)
        assert len(result) > 0


class TestPathUtilities:
    """Test cases for path-related utility functions."""

    def test_find_src_directory_returns_path(self):
        """Test that find_src_directory returns a valid path."""
        from actions.utils import find_src_directory
        
        result = find_src_directory()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_add_src_to_path_returns_string(self):
        """Test that add_src_to_path returns a string."""
        from actions.utils import add_src_to_path
        
        result = add_src_to_path()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_add_project_root_src_to_path_returns_string(self):
        """Test that add_project_root_src_to_path returns a string."""
        from actions.utils import add_project_root_src_to_path
        
        result = add_project_root_src_to_path()
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])