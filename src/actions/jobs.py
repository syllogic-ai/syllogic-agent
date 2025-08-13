"""
Job management utilities for Supabase jobs table.
Provides functions to create, update, and manage job statuses.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from supabase import Client
import logging

logger = logging.getLogger(__name__)


def create_job(
    supabase: Client, 
    job_id: str, 
    user_id: str, 
    dashboard_id: str,
    status: str = "pending",
    progress: int = 0
) -> Dict[str, Any]:
    """
    Create a new job record in Supabase.
    
    Args:
        supabase: Supabase client instance
        job_id: Unique job identifier
        user_id: User ID who owns the job
        dashboard_id: Dashboard ID associated with the job
        status: Initial job status (default: "pending")
        progress: Initial progress (default: 0)
        
    Returns:
        Dict containing the created job data
        
    Raises:
        Exception: If job creation fails
    """
    try:
        job_data = {
            "id": job_id,
            "userId": user_id,
            "dashboardId": dashboard_id,
            "status": status,
            "progress": progress,
            "createdAt": datetime.now().isoformat(),
        }
        
        result = supabase.table("jobs").insert(job_data).execute()
        
        if result.data:
            logger.info(f"Created job {job_id} for user {user_id}")
            return result.data[0]
        else:
            raise Exception(f"Failed to create job {job_id}: No data returned")
            
    except Exception as e:
        logger.error(f"Error creating job {job_id}: {str(e)}")
        raise


def update_job_status(
    supabase: Client,
    job_id: str,
    status: str,
    progress: Optional[int] = None,
    error: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Update job status and optionally other fields.
    
    Args:
        supabase: Supabase client instance
        job_id: Job identifier to update
        status: New status ("pending", "processing", "completed", "failed")
        progress: Progress percentage (0-100)
        error: Error message if status is "failed"
        **kwargs: Additional fields to update
        
    Returns:
        Dict containing the updated job data
        
    Raises:
        Exception: If job update fails
    """
    try:
        update_data = {"status": status}
        
        if progress is not None:
            update_data["progress"] = progress
            
        if error is not None:
            update_data["error"] = error
            
        # Set timestamps based on status
        if status == "processing":
            update_data["startedAt"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            update_data["completedAt"] = datetime.now().isoformat()
            
            # Calculate processing time if startedAt exists
            if "startedAt" in kwargs:
                started_at = datetime.fromisoformat(kwargs["startedAt"].replace('Z', '+00:00'))
                completed_at = datetime.now()
                processing_time_ms = int((completed_at - started_at).total_seconds() * 1000)
                update_data["processingTimeMs"] = processing_time_ms
        
        # Add any additional fields
        update_data.update(kwargs)
        
        result = supabase.table("jobs").update(update_data).eq("id", job_id).execute()
        
        if result.data:
            logger.info(f"Updated job {job_id} to status {status} (progress: {progress})")
            return result.data[0]
        else:
            logger.warning(f"No job found with ID {job_id} to update")
            return {}
            
    except Exception as e:
        logger.error(f"Error updating job {job_id}: {str(e)}")
        raise


def processing_job(
    supabase: Client,
    job_id: str,
    progress: int = 10
) -> Dict[str, Any]:
    """
    Mark job as processing with optional progress.
    
    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        progress: Progress percentage (default: 10)
        
    Returns:
        Dict containing the updated job data
    """
    return update_job_status(
        supabase=supabase,
        job_id=job_id,
        status="processing",
        progress=progress
    )


def complete_job(
    supabase: Client,
    job_id: str,
    result_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Mark job as completed.
    
    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        result_data: Optional result data to store
        
    Returns:
        Dict containing the updated job data
    """
    update_fields = {
        "progress": 100,
    }
    
    # Add result data if provided
    if result_data:
        # Store result data as JSON in a result field (if your schema has one)
        # or log it for now
        logger.info(f"Job {job_id} completed with result: {result_data}")
    
    return update_job_status(
        supabase=supabase,
        job_id=job_id,
        status="completed",
        **update_fields
    )


def fail_job(
    supabase: Client,
    job_id: str,
    error_message: str
) -> Dict[str, Any]:
    """
    Mark job as failed with error message.
    
    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        error_message: Error description
        
    Returns:
        Dict containing the updated job data
    """
    return update_job_status(
        supabase=supabase,
        job_id=job_id,
        status="failed",
        error=error_message
    )


def get_job(
    supabase: Client,
    job_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get job by ID.
    
    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        
    Returns:
        Dict containing job data or None if not found
    """
    try:
        result = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
        
        if result.data:
            return result.data
        else:
            logger.warning(f"Job {job_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {str(e)}")
        return None


def get_user_jobs(
    supabase: Client,
    user_id: str,
    dashboard_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> list[Dict[str, Any]]:
    """
    Get jobs for a user, optionally filtered by dashboard and status.
    
    Args:
        supabase: Supabase client instance
        user_id: User ID
        dashboard_id: Optional dashboard ID filter
        status: Optional status filter
        limit: Maximum number of jobs to return
        
    Returns:
        List of job dictionaries
    """
    try:
        query = supabase.table("jobs").select("*").eq("userId", user_id)
        
        if dashboard_id:
            query = query.eq("dashboardId", dashboard_id)
            
        if status:
            query = query.eq("status", status)
            
        result = query.order("createdAt", desc=True).limit(limit).execute()
        
        return result.data if result.data else []
        
    except Exception as e:
        logger.error(f"Error fetching jobs for user {user_id}: {str(e)}")
        return []


def cleanup_old_jobs(
    supabase: Client,
    days_old: int = 7,
    statuses: list[str] = ["completed", "failed"]
) -> int:
    """
    Clean up old jobs to prevent database bloat.
    
    Args:
        supabase: Supabase client instance
        days_old: Delete jobs older than this many days
        statuses: Only delete jobs with these statuses
        
    Returns:
        Number of jobs deleted
    """
    try:
        from datetime import timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        # Get jobs to delete
        query = supabase.table("jobs").select("id").lt("createdAt", cutoff_date)
        
        if statuses:
            query = query.in_("status", statuses)
            
        result = query.execute()
        
        if result.data:
            job_ids = [job["id"] for job in result.data]
            
            # Delete jobs
            delete_result = supabase.table("jobs").delete().in_("id", job_ids).execute()
            
            deleted_count = len(delete_result.data) if delete_result.data else 0
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return deleted_count
        
        return 0
        
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {str(e)}")
        return 0


# Convenience functions for common job lifecycle operations
def start_job_processing(supabase: Client, job_id: str, progress: int = 10):
    """Convenience function to mark job as processing."""
    return processing_job(supabase, job_id, progress)


def update_job_progress(supabase: Client, job_id: str, progress: int):
    """Convenience function to update job progress."""
    return update_job_status(supabase, job_id, "processing", progress)


def finish_job_success(supabase: Client, job_id: str, result_data: Optional[Dict] = None):
    """Convenience function to mark job as completed."""
    return complete_job(supabase, job_id, result_data)


def finish_job_error(supabase: Client, job_id: str, error: str):
    """Convenience function to mark job as failed."""
    return fail_job(supabase, job_id, error)
