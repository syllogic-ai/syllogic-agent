"""Job management utilities for Supabase jobs table.
Provides functions to create, update, and manage job statuses.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from supabase import Client

from agent.models import CreateJobInput, Job, UpdateJobInput

logger = logging.getLogger(__name__)


def create_job(supabase: Client, job_input: CreateJobInput) -> Job:
    """Create a new job record in Supabase.

    Args:
        supabase: Supabase client instance
        job_input: CreateJobInput containing job creation data

    Returns:
        Job model containing the created job data

    Raises:
        Exception: If job creation fails
    """
    try:
        job_data = {
            "id": job_input.job_id,
            "user_id": job_input.user_id,
            "dashboard_id": job_input.dashboard_id,
            "status": job_input.status,
            "progress": job_input.progress,
            "created_at": datetime.now().isoformat(),
        }

        result = supabase.table("jobs").insert(job_data).execute()

        if result.data:
            logger.info(f"Created job {job_input.job_id} for user {job_input.user_id}")
            return Job(**result.data[0])
        else:
            raise Exception(
                f"Failed to create job {job_input.job_id}: No data returned"
            )

    except Exception as e:
        logger.error(f"Error creating job {job_input.job_id}: {str(e)}")
        raise


def update_job_status(supabase: Client, update_input: UpdateJobInput) -> Job:
    """Update job status and optionally other fields.

    Args:
        supabase: Supabase client instance
        update_input: UpdateJobInput containing update data

    Returns:
        Job model containing the updated job data

    Raises:
        Exception: If job update fails
    """
    try:
        update_data = {"status": update_input.status}

        if update_input.progress is not None:
            update_data["progress"] = update_input.progress

        if update_input.error is not None:
            update_data["error"] = update_input.error

        # Set timestamps based on status
        if update_input.status == "processing":
            update_data["started_at"] = datetime.now().isoformat()
        elif update_input.status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.now().isoformat()

            # Get current job to calculate processing time
            try:
                current_job = (
                    supabase.table("jobs")
                    .select("started_at")
                    .eq("id", update_input.job_id)
                    .single()
                    .execute()
                )
                if (current_job.data and 
                    hasattr(current_job.data, 'get') and 
                    current_job.data.get("started_at")):
                    started_at = datetime.fromisoformat(
                        current_job.data["started_at"].replace("Z", "+00:00")
                    )
                    completed_at = datetime.now()
                    processing_time_ms = int(
                        (completed_at - started_at).total_seconds() * 1000
                    )
                    update_data["processing_time_ms"] = processing_time_ms
            except Exception as e:
                # Skip processing time calculation if query fails
                logger.debug(f"Could not calculate processing time for job {update_input.job_id}: {str(e)}")

        result = (
            supabase.table("jobs")
            .update(update_data)
            .eq("id", update_input.job_id)
            .execute()
        )

        if result.data:
            logger.info(
                f"Updated job {update_input.job_id} to status {update_input.status} (progress: {update_input.progress})"
            )
            return Job(**result.data[0])
        else:
            logger.warning(f"No job found with ID {update_input.job_id} to update")
            # Return a minimal job object for backwards compatibility
            return Job(
                id=update_input.job_id,
                user_id="",
                dashboard_id="",
                status=update_input.status,
            )

    except Exception as e:
        logger.error(f"Error updating job {update_input.job_id}: {str(e)}")
        raise


def processing_job(supabase: Client, job_id: str, progress: int = 10) -> Job:
    """Mark job as processing with optional progress.

    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        progress: Progress percentage (default: 10)

    Returns:
        Job model containing the updated job data
    """
    return update_job_status(
        supabase=supabase,
        update_input=UpdateJobInput(
            job_id=job_id, status="processing", progress=progress
        ),
    )


def complete_job(
    supabase: Client, job_id: str, result_data: Optional[Dict[str, Any]] = None
) -> Job:
    """Mark job as completed.

    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        result_data: Optional result data to store

    Returns:
        Job model containing the updated job data
    """
    # Add result data if provided
    if result_data:
        # Store result data as JSON in a result field (if your schema has one)
        # or log it for now
        logger.info(f"Job {job_id} completed with result: {result_data}")

    return update_job_status(
        supabase=supabase,
        update_input=UpdateJobInput(job_id=job_id, status="completed", progress=100),
    )


def fail_job(supabase: Client, job_id: str, error_message: str) -> Job:
    """Mark job as failed with error message.

    Args:
        supabase: Supabase client instance
        job_id: Job identifier
        error_message: Error description

    Returns:
        Job model containing the updated job data
    """
    return update_job_status(
        supabase=supabase,
        update_input=UpdateJobInput(
            job_id=job_id, status="failed", error=error_message
        ),
    )


def get_job(supabase: Client, job_id: str) -> Optional[Job]:
    """Get job by ID.

    Args:
        supabase: Supabase client instance
        job_id: Job identifier

    Returns:
        Job model containing job data or None if not found
    """
    try:
        result = supabase.table("jobs").select("*").eq("id", job_id).single().execute()

        if result.data:
            return Job(**result.data)
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
    limit: int = 50,
) -> List[Job]:
    """Get jobs for a user, optionally filtered by dashboard and status.

    Args:
        supabase: Supabase client instance
        user_id: User ID
        dashboard_id: Optional dashboard ID filter
        status: Optional status filter
        limit: Maximum number of jobs to return

    Returns:
        List of Job models
    """
    try:
        query = supabase.table("jobs").select("*").eq("user_id", user_id)

        if dashboard_id:
            query = query.eq("dashboard_id", dashboard_id)

        if status:
            query = query.eq("status", status)

        result = query.order("created_at", desc=True).limit(limit).execute()

        return [Job(**job) for job in result.data] if result.data else []

    except Exception as e:
        logger.error(f"Error fetching jobs for user {user_id}: {str(e)}")
        return []


def cleanup_old_jobs(
    supabase: Client, days_old: int = 7, statuses: list[str] = ["completed", "failed"]
) -> int:
    """Clean up old jobs to prevent database bloat.

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
        query = supabase.table("jobs").select("id").lt("created_at", cutoff_date)

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
def start_job_processing(supabase: Client, job_id: str, progress: int = 10) -> Job:
    """Convenience function to mark job as processing."""
    return processing_job(supabase, job_id, progress)


def update_job_progress(supabase: Client, job_id: str, progress: int) -> Job:
    """Convenience function to update job progress."""
    return update_job_status(
        supabase, UpdateJobInput(job_id=job_id, status="processing", progress=progress)
    )


def finish_job_success(
    supabase: Client, job_id: str, result_data: Optional[Dict] = None
) -> Job:
    """Convenience function to mark job as completed."""
    return complete_job(supabase, job_id, result_data)


def finish_job_error(supabase: Client, job_id: str, error: str) -> Job:
    """Convenience function to mark job as failed."""
    return fail_job(supabase, job_id, error)
