"""Bulk database operations for top-level supervisor."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.models import CreateWidgetInput, UpdateWidgetInput
from actions.utils import import_actions_dashboard

logger = logging.getLogger(__name__)


def execute_bulk_database_operations(
    pending_operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute bulk database operations from the pending operations list.
    
    Args:
        pending_operations: List of database operation objects
        
    Returns:
        Dictionary containing results summary and any errors
    """
    if not pending_operations:
        return {
            "success": True,
            "operations_executed": 0,
            "results": [],
            "errors": []
        }
    
    try:
        # Import dashboard functions using robust import
        dashboard_module = import_actions_dashboard()
        create_widget = dashboard_module.create_widget
        update_widget = dashboard_module.update_widget
        delete_widget = dashboard_module.delete_widget
    except ImportError as e:
        error_msg = f"Failed to import actions.dashboard module: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "operations_executed": 0,
            "results": [],
            "errors": [error_msg]
        }
    
    results = []
    errors = []
    successful_operations = 0
    
    for operation in pending_operations:
        try:
            operation_type = operation.get("operation_type")
            task_id = operation.get("task_id")
            
            if operation_type == "CREATE":
                # Execute widget creation
                widget_data = operation.get("widget_data")
                expected_widget_id = operation.get("expected_widget_id")
                
                create_input = CreateWidgetInput(**widget_data)
                created_widget = create_widget(create_input)
                
                results.append({
                    "operation_type": "CREATE",
                    "task_id": task_id,
                    "widget_id": created_widget.id,
                    "expected_widget_id": expected_widget_id,
                    "status": "success",
                    "title": widget_data.get("title", ""),
                    "message": f"Successfully created widget '{widget_data.get('title', '')}' with ID: {created_widget.id}"
                })
                successful_operations += 1
                
            elif operation_type == "UPDATE":
                # Execute widget update
                widget_data = operation.get("widget_data")
                widget_id = operation.get("widget_id")
                
                update_input = UpdateWidgetInput(**widget_data)
                updated_widget = update_widget(update_input)
                
                results.append({
                    "operation_type": "UPDATE",
                    "task_id": task_id,
                    "widget_id": widget_id,
                    "status": "success",
                    "title": widget_data.get("title", ""),
                    "message": f"Successfully updated widget '{widget_data.get('title', '')}' with ID: {widget_id}"
                })
                successful_operations += 1
                
            elif operation_type == "DELETE":
                # Execute widget deletion
                widget_id = operation.get("widget_id")
                widget_title = operation.get("widget_title", "")
                
                deletion_success = delete_widget(widget_id)
                
                if deletion_success:
                    results.append({
                        "operation_type": "DELETE",
                        "task_id": task_id,
                        "widget_id": widget_id,
                        "status": "success",
                        "title": widget_title,
                        "message": f"Successfully deleted widget with ID: {widget_id}"
                    })
                    successful_operations += 1
                else:
                    error_msg = f"Failed to delete widget with ID: {widget_id} (widget not found)"
                    errors.append({
                        "operation_type": "DELETE",
                        "task_id": task_id,
                        "widget_id": widget_id,
                        "error": error_msg
                    })
                    
            else:
                error_msg = f"Invalid operation type '{operation_type}' for task {task_id}"
                errors.append({
                    "task_id": task_id,
                    "error": error_msg
                })
                
        except Exception as e:
            error_msg = f"Failed to execute {operation_type} operation for task {task_id}: {str(e)}"
            logger.error(error_msg)
            errors.append({
                "operation_type": operation_type,
                "task_id": task_id,
                "error": error_msg
            })
    
    return {
        "success": len(errors) == 0,
        "operations_executed": successful_operations,
        "total_operations": len(pending_operations),
        "results": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }


def collect_database_operations_from_tasks(
    delegated_tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Collect database operations from completed widget agent tasks.
    
    Args:
        delegated_tasks: List of delegated tasks that may contain database operations
        
    Returns:
        List of database operation objects ready for execution
    """
    database_operations = []
    
    logger.info(f"🔍 COLLECTING DATABASE OPERATIONS FROM {len(delegated_tasks)} TASKS")
    
    for i, task in enumerate(delegated_tasks):
        # Handle both dict and DelegatedTask object formats
        task_data = task if isinstance(task, dict) else task.__dict__
        
        task_id = task_data.get("task_id", f"unknown_{i}")
        task_status = task_data.get("task_status") or task_data.get("status")
        
        logger.info(f"  📋 Task {i+1} ({task_id}): status={task_status}")
        
        # Debug: Print all available keys in the task data
        logger.info(f"    🔑 Available task data keys: {list(task_data.keys())}")
        
        # Check if task is completed and has database operations
        if task_status == "completed":
            logger.info(f"    ✅ Task {task_id} is completed - checking for database operations")
            
            # Check if database_operation is directly in the task data
            direct_db_op = task_data.get("database_operation")
            if direct_db_op:
                logger.info(f"    🎯 Found DIRECT database_operation in task {task_id}: {type(direct_db_op)}")
                database_operations.append(direct_db_op)
            else:
                logger.info(f"    ❌ No direct database_operation found in task {task_id}")
            
            # Also check the result field for database operations
            task_result = task_data.get("result", {})
            logger.info(f"    🔍 Task {task_id} result type: {type(task_result)}, value preview: {str(task_result)[:100]}...")
            
            if isinstance(task_result, dict):
                # Check if result itself is a database operation
                if "operation_type" in task_result and "widget_data" in task_result:
                    logger.info(f"    🎯 Found database operation AS RESULT in task {task_id}")
                    database_operations.append(task_result)
                # Or check if result contains a nested database_operation
                elif "database_operation" in task_result:
                    db_operation = task_result["database_operation"]
                    if db_operation:
                        logger.info(f"    🎯 Found NESTED database_operation in task {task_id} result")
                        database_operations.append(db_operation)
                    else:
                        logger.info(f"    ❌ Task {task_id} result has database_operation key but value is empty")
                else:
                    logger.info(f"    ❌ Task {task_id} result is dict but doesn't contain database operation")
                    logger.info(f"        Result keys: {list(task_result.keys())}")
            else:
                logger.info(f"    ❌ Task {task_id} result is not a dict: {type(task_result)}")
        else:
            logger.info(f"    ⏭️  Task {task_id} status is {task_status} - skipping")
    
    logger.info(f"🎯 COLLECTION COMPLETE: Found {len(database_operations)} database operations")
    for i, op in enumerate(database_operations):
        op_type = op.get("operation_type", "UNKNOWN")
        widget_title = op.get("widget_data", {}).get("title", "No title") if isinstance(op.get("widget_data"), dict) else "No widget_data"
        logger.info(f"  Operation {i+1}: {op_type} - {widget_title}")
    
    return database_operations