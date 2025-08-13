"""
Unit tests for jobs.py functions.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.jobs import (
    create_job,
    update_job_status,
    processing_job,
    complete_job,
    fail_job,
    get_job,
    get_user_jobs,
    cleanup_old_jobs,
    start_job_processing,
    update_job_progress,
    finish_job_success,
    finish_job_error
)


class TestCreateJob:
    """Test cases for create_job function."""
    
    def test_create_job_success(self, mock_supabase):
        """Test successfully creating a job."""
        mock_result = Mock()
        mock_result.data = [{
            "id": "job-123",
            "userId": "user-456",
            "dashboardId": "dashboard-789",
            "status": "pending",
            "progress": 0
        }]
        mock_supabase.table().insert().execute.return_value = mock_result
        
        result = create_job(
            mock_supabase,
            "job-123",
            "user-456",
            "dashboard-789"
        )
        
        assert result["id"] == "job-123"
        assert result["userId"] == "user-456"
        assert result["dashboardId"] == "dashboard-789"
        assert result["status"] == "pending"
        assert result["progress"] == 0
        
        # Verify database call
        insert_call = mock_supabase.table().insert.call_args[0][0]
        assert insert_call["id"] == "job-123"
        assert insert_call["status"] == "pending"
        assert insert_call["progress"] == 0
        assert "createdAt" in insert_call
        
    def test_create_job_with_custom_status(self, mock_supabase):
        """Test creating job with custom status and progress."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().insert().execute.return_value = mock_result
        
        result = create_job(
            mock_supabase,
            "job-123",
            "user-456", 
            "dashboard-789",
            status="processing",
            progress=25
        )
        
        insert_call = mock_supabase.table().insert.call_args[0][0]
        assert insert_call["status"] == "processing"
        assert insert_call["progress"] == 25
        
    def test_create_job_insert_fails(self, mock_supabase):
        """Test error when job insert fails."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().insert().execute.return_value = mock_result
        
        with pytest.raises(Exception, match="Failed to create job job-123"):
            create_job(mock_supabase, "job-123", "user-456", "dashboard-789")
            
    def test_create_job_database_error(self, mock_supabase):
        """Test handling of database errors during job creation."""
        mock_supabase.table().insert().execute.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception, match="DB Error"):
            create_job(mock_supabase, "job-123", "user-456", "dashboard-789")


class TestUpdateJobStatus:
    """Test cases for update_job_status function."""
    
    def test_update_job_status_basic(self, mock_supabase):
        """Test basic job status update."""
        mock_result = Mock()
        mock_result.data = [{
            "id": "job-123",
            "status": "processing",
            "progress": 50
        }]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        result = update_job_status(
            mock_supabase,
            "job-123",
            "processing",
            progress=50
        )
        
        assert result["status"] == "processing"
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "processing"
        assert update_call["progress"] == 50
        
    def test_update_job_status_with_error(self, mock_supabase):
        """Test updating job status with error message."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        update_job_status(
            mock_supabase,
            "job-123",
            "failed",
            error="Test error message"
        )
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "failed"
        assert update_call["error"] == "Test error message"
        
    def test_update_job_status_processing_sets_started_at(self, mock_supabase):
        """Test that processing status sets startedAt timestamp."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        update_job_status(mock_supabase, "job-123", "processing")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "processing"
        assert "startedAt" in update_call
        
    def test_update_job_status_completed_sets_completed_at(self, mock_supabase):
        """Test that completed status sets completedAt timestamp."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        update_job_status(mock_supabase, "job-123", "completed")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "completed"
        assert "completedAt" in update_call
        
    def test_update_job_status_calculates_processing_time(self, mock_supabase):
        """Test that processing time is calculated when provided."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        started_time = datetime.now() - timedelta(seconds=30)
        
        update_job_status(
            mock_supabase,
            "job-123",
            "completed",
            startedAt=started_time.isoformat()
        )
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert "processingTimeMs" in update_call
        # Should be approximately 30000ms (30 seconds)
        assert update_call["processingTimeMs"] >= 29000
        assert update_call["processingTimeMs"] <= 31000
        
    def test_update_job_status_not_found(self, mock_supabase):
        """Test updating job that doesn't exist."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        result = update_job_status(mock_supabase, "job-123", "processing")
        
        assert result == {}


class TestProcessingJob:
    """Test cases for processing_job function."""
    
    def test_processing_job_default_progress(self, mock_supabase):
        """Test marking job as processing with default progress."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        processing_job(mock_supabase, "job-123")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "processing"
        assert update_call["progress"] == 10
        
    def test_processing_job_custom_progress(self, mock_supabase):
        """Test marking job as processing with custom progress."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        processing_job(mock_supabase, "job-123", progress=25)
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["progress"] == 25


class TestCompleteJob:
    """Test cases for complete_job function."""
    
    def test_complete_job_success(self, mock_supabase):
        """Test marking job as completed."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        complete_job(mock_supabase, "job-123")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "completed"
        assert update_call["progress"] == 100
        
    def test_complete_job_with_result_data(self, mock_supabase):
        """Test completing job with result data (logged)."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        result_data = {"widgets_created": 3}
        
        with patch('actions.jobs.logger') as mock_logger:
            complete_job(mock_supabase, "job-123", result_data)
            
            # Check that logger.info was called with the result data
            logger_calls = mock_logger.info.call_args_list
            assert len(logger_calls) >= 1
            # Should contain a call about the result data
            result_logged = any("result:" in str(call) for call in logger_calls)
            assert result_logged


class TestFailJob:
    """Test cases for fail_job function."""
    
    def test_fail_job_success(self, mock_supabase):
        """Test marking job as failed with error message."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        fail_job(mock_supabase, "job-123", "Test error")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "failed"
        assert update_call["error"] == "Test error"


class TestGetJob:
    """Test cases for get_job function."""
    
    def test_get_job_success(self, mock_supabase, sample_job_data):
        """Test successfully getting a job."""
        mock_result = Mock()
        mock_result.data = sample_job_data
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = get_job(mock_supabase, "job-123")
        
        assert result == sample_job_data
        assert result["id"] == "job-123"
        
    def test_get_job_not_found(self, mock_supabase):
        """Test getting job that doesn't exist."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().single().execute.return_value = mock_result
        
        result = get_job(mock_supabase, "job-123")
        
        assert result is None
        
    def test_get_job_database_error(self, mock_supabase):
        """Test handling database error when getting job."""
        mock_supabase.table().select().eq().single().execute.side_effect = Exception("DB Error")
        
        result = get_job(mock_supabase, "job-123")
        
        assert result is None


class TestGetUserJobs:
    """Test cases for get_user_jobs function."""
    
    def test_get_user_jobs_success(self, mock_supabase, sample_job_data):
        """Test successfully getting user jobs."""
        mock_result = Mock()
        mock_result.data = [sample_job_data]
        mock_supabase.table().select().eq().order().limit().execute.return_value = mock_result
        
        result = get_user_jobs(mock_supabase, "user-456")
        
        assert len(result) == 1
        assert result[0] == sample_job_data
        
    def test_get_user_jobs_with_filters(self, mock_supabase):
        """Test getting user jobs with dashboard and status filters."""
        mock_result = Mock()
        mock_result.data = []
        
        # Setup mock chain for multiple eq() calls
        mock_query = Mock()
        mock_supabase.table().select().eq.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order().limit().execute.return_value = mock_result
        
        result = get_user_jobs(
            mock_supabase,
            "user-456",
            dashboard_id="dashboard-789",
            status="completed"
        )
        
        assert result == []
        
    def test_get_user_jobs_no_data(self, mock_supabase):
        """Test getting user jobs when no data returned."""
        mock_result = Mock()
        mock_result.data = None
        mock_supabase.table().select().eq().order().limit().execute.return_value = mock_result
        
        result = get_user_jobs(mock_supabase, "user-456")
        
        assert result == []
        
    def test_get_user_jobs_database_error(self, mock_supabase):
        """Test handling database error when getting user jobs."""
        mock_supabase.table().select().eq().order().limit().execute.side_effect = Exception("DB Error")
        
        result = get_user_jobs(mock_supabase, "user-456")
        
        assert result == []


class TestCleanupOldJobs:
    """Test cases for cleanup_old_jobs function."""
    
    def test_cleanup_old_jobs_success(self, mock_supabase):
        """Test successfully cleaning up old jobs."""
        # Mock select query to find old jobs
        mock_select_result = Mock()
        mock_select_result.data = [
            {"id": "job-1"},
            {"id": "job-2"}
        ]
        
        # Mock delete result
        mock_delete_result = Mock()
        mock_delete_result.data = [{"id": "job-1"}, {"id": "job-2"}]
        
        # Setup mock query chain
        mock_query = Mock()
        mock_supabase.table().select().lt.return_value = mock_query
        mock_query.in_.return_value = mock_query
        mock_query.execute.return_value = mock_select_result
        
        mock_supabase.table().delete().in_().execute.return_value = mock_delete_result
        
        result = cleanup_old_jobs(mock_supabase, days_old=7)
        
        assert result == 2
        
    def test_cleanup_old_jobs_no_old_jobs(self, mock_supabase):
        """Test cleanup when no old jobs found."""
        mock_result = Mock()
        mock_result.data = None
        
        mock_query = Mock()
        mock_supabase.table().select().lt.return_value = mock_query
        mock_query.in_.return_value = mock_query
        mock_query.execute.return_value = mock_result
        
        result = cleanup_old_jobs(mock_supabase)
        
        assert result == 0
        
    def test_cleanup_old_jobs_custom_parameters(self, mock_supabase):
        """Test cleanup with custom parameters."""
        mock_result = Mock()
        mock_result.data = []
        
        mock_query = Mock()
        mock_supabase.table().select().lt.return_value = mock_query
        mock_query.in_.return_value = mock_query
        mock_query.execute.return_value = mock_result
        
        cleanup_old_jobs(
            mock_supabase,
            days_old=30,
            statuses=["failed"]
        )
        
        # Verify the correct parameters were used in the query
        # The cutoff date should be 30 days ago
        lt_call = mock_supabase.table().select().lt.call_args[0]
        assert lt_call[0] == "createdAt"
        
    def test_cleanup_old_jobs_database_error(self, mock_supabase):
        """Test handling database error during cleanup."""
        mock_supabase.table().select().lt().in_().execute.side_effect = Exception("DB Error")
        
        result = cleanup_old_jobs(mock_supabase)
        
        assert result == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_start_job_processing(self, mock_supabase):
        """Test start_job_processing convenience function."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        start_job_processing(mock_supabase, "job-123", progress=15)
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "processing"
        assert update_call["progress"] == 15
        
    def test_update_job_progress(self, mock_supabase):
        """Test update_job_progress convenience function."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        update_job_progress(mock_supabase, "job-123", 75)
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "processing"
        assert update_call["progress"] == 75
        
    def test_finish_job_success(self, mock_supabase):
        """Test finish_job_success convenience function."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        result_data = {"result": "success"}
        
        with patch('actions.jobs.logger'):
            finish_job_success(mock_supabase, "job-123", result_data)
            
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "completed"
        assert update_call["progress"] == 100
        
    def test_finish_job_error(self, mock_supabase):
        """Test finish_job_error convenience function."""
        mock_result = Mock()
        mock_result.data = [{"id": "job-123"}]
        mock_supabase.table().update().eq().execute.return_value = mock_result
        
        finish_job_error(mock_supabase, "job-123", "Test error")
        
        update_call = mock_supabase.table().update.call_args[0][0]
        assert update_call["status"] == "failed"
        assert update_call["error"] == "Test error"