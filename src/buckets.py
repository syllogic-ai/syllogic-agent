"""Bucket configuration for file storage.

This module defines the available storage buckets and their configurations
for the syllogic-agent system.
"""

from enum import Enum
from typing import Dict, Any


class BucketType(Enum):
    """Enumeration of available bucket types."""
    USER_FILES = "user-files"
    TEST_BUCKET = "test-bucket"
    PUBLIC_FILES = "public-files"
    TEMP_FILES = "temp-files"


class BucketConfig:
    """Configuration for storage buckets."""
    
    def __init__(self, name: str, is_public: bool = False, max_file_size: str = "50MiB", 
                 allowed_mime_types: list = None, description: str = ""):
        self.name = name
        self.is_public = is_public
        self.max_file_size = max_file_size
        self.allowed_mime_types = allowed_mime_types or []
        self.description = description


# Bucket configurations
BUCKET_CONFIGS: Dict[BucketType, BucketConfig] = {
    BucketType.USER_FILES: BucketConfig(
        name="user-files",
        is_public=False,
        max_file_size="50MiB",
        allowed_mime_types=[
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/json",
            "text/plain"
        ],
        description="Private bucket for user-uploaded files"
    ),
    
    BucketType.TEST_BUCKET: BucketConfig(
        name="test-bucket",
        is_public=True,
        max_file_size="10MiB",
        allowed_mime_types=[
            "text/csv",
            "application/json"
        ],
        description="Public test bucket for development"
    ),
    
    BucketType.PUBLIC_FILES: BucketConfig(
        name="public-files",
        is_public=True,
        max_file_size="100MiB",
        allowed_mime_types=[
            "image/png",
            "image/jpeg",
            "image/gif",
            "application/pdf"
        ],
        description="Public bucket for shared files"
    ),
    
    BucketType.TEMP_FILES: BucketConfig(
        name="temp-files",
        is_public=False,
        max_file_size="25MiB",
        allowed_mime_types=[
            "text/csv",
            "application/json",
            "text/plain"
        ],
        description="Temporary bucket for processing files"
    )
}


def get_bucket_config(bucket_type: BucketType) -> BucketConfig:
    """Get configuration for a specific bucket type.
    
    Args:
        bucket_type: The bucket type to get configuration for
        
    Returns:
        BucketConfig object for the specified bucket type
        
    Raises:
        KeyError: If bucket type is not found
    """
    if bucket_type not in BUCKET_CONFIGS:
        raise KeyError(f"Bucket type {bucket_type} not found in configurations")
    
    return BUCKET_CONFIGS[bucket_type]


def get_bucket_name(bucket_type: BucketType) -> str:
    """Get the bucket name for a specific bucket type.
    
    Args:
        bucket_type: The bucket type to get name for
        
    Returns:
        String name of the bucket
    """
    return get_bucket_config(bucket_type).name


def is_public_bucket(bucket_type: BucketType) -> bool:
    """Check if a bucket is public.
    
    Args:
        bucket_type: The bucket type to check
        
    Returns:
        True if bucket is public, False otherwise
    """
    return get_bucket_config(bucket_type).is_public


def get_allowed_mime_types(bucket_type: BucketType) -> list:
    """Get allowed MIME types for a bucket.
    
    Args:
        bucket_type: The bucket type to get MIME types for
        
    Returns:
        List of allowed MIME types
    """
    return get_bucket_config(bucket_type).allowed_mime_types


def get_storage_path_with_bucket(bucket_type: BucketType, user_id: str, 
                                dashboard_id: str, filename: str) -> str:
    """Generate storage path with bucket identification.
    
    Args:
        bucket_type: The bucket type to use
        user_id: User identifier
        dashboard_id: Dashboard identifier
        filename: File name
        
    Returns:
        Full storage path including bucket: "bucket_name/user_id/dashboard_id/filename"
    """
    bucket_name = get_bucket_name(bucket_type)
    return f"{bucket_name}/{user_id}/{dashboard_id}/{filename}"


def parse_storage_path_with_bucket(storage_path: str) -> tuple:
    """Parse storage path to extract bucket and file path components.
    
    Handles both new format (with bucket) and legacy format (without bucket).
    
    Args:
        storage_path: Full storage path, either with or without bucket
        
    Returns:
        Tuple of (bucket_name, file_path) where file_path is "user_id/dashboard_id/filename"
    """
    parts = storage_path.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid storage path format: {storage_path}")
    
    potential_bucket = parts[0]
    file_path = parts[1]
    
    # Check if the first part is a known bucket name
    try:
        get_bucket_type_from_name(potential_bucket)
        # If we get here, it's a valid bucket name
        return potential_bucket, file_path
    except KeyError:
        # If not a known bucket, this is legacy format without bucket
        # Return default bucket with the full path as file_path
        return DEFAULT_USER_BUCKET.value, storage_path


def get_bucket_type_from_name(bucket_name: str) -> BucketType:
    """Get bucket type from bucket name.
    
    Args:
        bucket_name: Name of the bucket
        
    Returns:
        BucketType enum value
        
    Raises:
        KeyError: If bucket name is not found
    """
    for bucket_type, config in BUCKET_CONFIGS.items():
        if config.name == bucket_name:
            return bucket_type
    
    raise KeyError(f"Bucket name '{bucket_name}' not found in configurations")


# Default bucket for user files
DEFAULT_USER_BUCKET = BucketType.USER_FILES
DEFAULT_TEST_BUCKET = BucketType.TEST_BUCKET
