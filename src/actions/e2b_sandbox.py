"""E2B sandbox management helper functions."""

from typing import Optional, Dict, Any, List, Union
import asyncio
from contextlib import asynccontextmanager

from config import get_e2b_api_key

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _extract_log_messages(log_list) -> List[str]:
    """Extract log messages handling different E2B formats.
    
    Args:
        log_list: List of log messages (could be strings or OutputMessage objects)
        
    Returns:
        List of string messages
    """
    if not log_list:
        return []
    
    extracted = []
    for msg in log_list:
        if hasattr(msg, 'line'):
            # OutputMessage object
            extracted.append(msg.line)
        elif isinstance(msg, str):
            # Direct string
            extracted.append(msg)
        else:
            # Convert to string as fallback
            extracted.append(str(msg))
    return extracted


def create_e2b_sandbox():
    """Create a new synchronous E2B sandbox instance.
    
    Returns:
        Sandbox: A new E2B sandbox instance
        
    Raises:
        ValueError: If E2B API key is not available
        ImportError: If E2B code interpreter package is not installed
    """
    try:
        from e2b_code_interpreter import Sandbox
        
        api_key = get_e2b_api_key()
        if not api_key:
            raise ValueError("E2B API key is not available. Please set E2B_SANDBOX_API_KEY environment variable.")
        
        sandbox = Sandbox(api_key=api_key)
        logger.info("Created synchronous E2B sandbox successfully")
        return sandbox
        
    except ImportError as e:
        logger.error(f"Failed to import E2B code interpreter: {e}")
        raise ImportError("e2b_code_interpreter package not installed. Install with: pip install e2b-code-interpreter")
    except Exception as e:
        logger.error(f"Failed to create E2B sandbox: {e}")
        raise


async def create_async_e2b_sandbox():
    """Create a new asynchronous E2B sandbox instance using asyncio.to_thread to avoid blocking calls.
    
    Returns:
        AsyncSandbox: A new E2B async sandbox instance
        
    Raises:
        ValueError: If E2B API key is not available
        ImportError: If E2B code interpreter package is not installed
    """
    def _create_sandbox_sync():
        """Internal sync function to create sandbox that will be run in thread."""
        from e2b_code_interpreter import AsyncSandbox
        import asyncio
        
        api_key = get_e2b_api_key()
        if not api_key:
            raise ValueError("E2B API key is not available. Please set E2B_SANDBOX_API_KEY environment variable.")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create sandbox in the thread's event loop
            sandbox = loop.run_until_complete(AsyncSandbox.create(api_key=api_key))
            logger.info("Created asynchronous E2B sandbox successfully in thread")
            return sandbox
        finally:
            loop.close()
    
    try:
        # Run the blocking sandbox creation in a separate thread
        logger.info("Starting E2B AsyncSandbox creation in separate thread to avoid blocking")
        sandbox = await asyncio.to_thread(_create_sandbox_sync)
        logger.info("E2B AsyncSandbox creation completed successfully")
        return sandbox
        
    except ImportError as e:
        logger.error(f"Failed to import E2B code interpreter: {e}")
        raise ImportError("e2b_code_interpreter package not installed. Install with: pip install e2b-code-interpreter")
    except Exception as e:
        logger.error(f"Failed to create async E2B sandbox: {e}")
        raise


def execute_code_in_sandbox(sandbox, code: str, timeout: Optional[float] = 60.0) -> Dict[str, Any]:
    """Execute Python code in a synchronous E2B sandbox.
    
    Args:
        sandbox: E2B Sandbox instance
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 60.0)
        
    Returns:
        Dict containing execution results with keys:
        - success (bool): Whether execution succeeded
        - result (Any): The execution result if successful
        - error (str): Error message if failed
        - logs (Dict): stdout and stderr logs
        - execution_count (int): Execution count
        
    Raises:
        Exception: If sandbox execution fails
    """
    try:
        logger.info(f"Executing code in sandbox (timeout: {timeout}s)")
        logger.debug(f"Code to execute: {code[:200]}{'...' if len(code) > 200 else ''}")
        
        # Execute the code
        execution = sandbox.run_code(code, timeout=timeout)
        
        # Check for execution errors
        if execution.error:
            logger.error(f"Code execution failed: {execution.error}")
            return {
                "success": False,
                "result": None,
                "error": str(execution.error),
                "logs": {
                    "stdout": _extract_log_messages(execution.logs.stdout if execution.logs else []),
                    "stderr": _extract_log_messages(execution.logs.stderr if execution.logs else [])
                },
                "execution_count": execution.execution_count if hasattr(execution, 'execution_count') else 0
            }
        
        # Extract results and logs with robust format handling
        results = execution.results if execution.results else []
        
        logs = {
            "stdout": _extract_log_messages(execution.logs.stdout if execution.logs else []),
            "stderr": _extract_log_messages(execution.logs.stderr if execution.logs else [])
        }
        
        # Get text representation if available
        text_result = execution.text if hasattr(execution, 'text') else None
        
        logger.info("Code executed successfully")
        logger.debug(f"Results: {results}, Text: {text_result}")
        
        return {
            "success": True,
            "result": text_result or results,
            "error": None,
            "logs": logs,
            "execution_count": execution.execution_count if hasattr(execution, 'execution_count') else 0,
            "full_execution": execution  # Include full execution object for advanced use
        }
        
    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}")
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "logs": {"stdout": [], "stderr": []},
            "execution_count": 0
        }


async def execute_code_in_async_sandbox(sandbox, code: str, timeout: Optional[float] = 60.0) -> Dict[str, Any]:
    """Execute Python code in an asynchronous E2B sandbox using asyncio.to_thread to avoid blocking calls.
    
    Args:
        sandbox: E2B AsyncSandbox instance
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 60.0)
        
    Returns:
        Dict containing execution results with keys:
        - success (bool): Whether execution succeeded
        - result (Any): The execution result if successful
        - error (str): Error message if failed
        - logs (Dict): stdout and stderr logs
        - execution_count (int): Execution count
        
    Raises:
        Exception: If sandbox execution fails
    """
    def _execute_code_sync():
        """Internal sync function to execute code that will be run in thread."""
        import asyncio
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Execute code in the thread's event loop
            execution = loop.run_until_complete(sandbox.run_code(code, timeout=timeout))
            return execution
        finally:
            loop.close()
    
    try:
        logger.info(f"Executing code in async sandbox (timeout: {timeout}s) using separate thread")
        logger.debug(f"Code to execute: {code[:200]}{'...' if len(code) > 200 else ''}")
        
        # Execute the code in a separate thread to avoid blocking
        execution = await asyncio.to_thread(_execute_code_sync)
        
        # Check for execution errors
        if execution.error:
            logger.error(f"Code execution failed: {execution.error}")
            return {
                "success": False,
                "result": None,
                "error": str(execution.error),
                "logs": {
                    "stdout": _extract_log_messages(execution.logs.stdout if execution.logs else []),
                    "stderr": _extract_log_messages(execution.logs.stderr if execution.logs else [])
                },
                "execution_count": execution.execution_count if hasattr(execution, 'execution_count') else 0
            }
        
        # Extract results and logs with robust format handling
        results = execution.results if execution.results else []
        
        logs = {
            "stdout": _extract_log_messages(execution.logs.stdout if execution.logs else []),
            "stderr": _extract_log_messages(execution.logs.stderr if execution.logs else [])
        }
        
        # Get text representation if available
        text_result = execution.text if hasattr(execution, 'text') else None
        
        logger.info("Code executed successfully using thread-safe approach")
        logger.debug(f"Results: {results}, Text: {text_result}")
        
        return {
            "success": True,
            "result": text_result or results,
            "error": None,
            "logs": logs,
            "execution_count": execution.execution_count if hasattr(execution, 'execution_count') else 0,
            "full_execution": execution  # Include full execution object for advanced use
        }
        
    except Exception as e:
        logger.error(f"Async sandbox execution failed: {e}")
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "logs": {"stdout": [], "stderr": []},
            "execution_count": 0
        }


def kill_sandbox(sandbox) -> bool:
    """Safely kill/close a synchronous E2B sandbox.
    
    Args:
        sandbox: E2B Sandbox instance to kill
        
    Returns:
        bool: True if successfully killed, False otherwise
    """
    try:
        if sandbox and hasattr(sandbox, 'kill'):
            sandbox.kill()
            logger.info("Sandbox killed successfully")
            return True
        else:
            logger.warning("Sandbox does not have kill method or is None")
            return False
    except Exception as e:
        logger.error(f"Failed to kill sandbox: {e}")
        return False


async def kill_async_sandbox(sandbox) -> bool:
    """Safely kill/close an asynchronous E2B sandbox using asyncio.to_thread to avoid blocking calls.
    
    Args:
        sandbox: E2B AsyncSandbox instance to kill
        
    Returns:
        bool: True if successfully killed, False otherwise
    """
    def _kill_sandbox_sync():
        """Internal sync function to kill sandbox that will be run in thread."""
        import asyncio
        
        if not sandbox or not hasattr(sandbox, 'kill'):
            return False
            
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Kill sandbox in the thread's event loop
            loop.run_until_complete(sandbox.kill())
            return True
        finally:
            loop.close()
    
    try:
        if sandbox and hasattr(sandbox, 'kill'):
            logger.info("Killing async sandbox using separate thread")
            success = await asyncio.to_thread(_kill_sandbox_sync)
            if success:
                logger.info("Async sandbox killed successfully")
                return True
            else:
                logger.warning("Failed to kill async sandbox")
                return False
        else:
            logger.warning("Async sandbox does not have kill method or is None")
            return False
    except Exception as e:
        logger.error(f"Failed to kill async sandbox: {e}")
        return False


@asynccontextmanager
async def managed_async_sandbox():
    """Context manager for automatic async sandbox creation and cleanup.
    
    Usage:
        async with managed_async_sandbox() as sandbox:
            result = await execute_code_in_async_sandbox(sandbox, "print('hello')")
            
    Yields:
        AsyncSandbox: E2B async sandbox instance
        
    Raises:
        Exception: If sandbox creation fails
    """
    sandbox = None
    try:
        sandbox = await create_async_e2b_sandbox()
        yield sandbox
    finally:
        if sandbox:
            await kill_async_sandbox(sandbox)


def get_sandbox_info(sandbox) -> Dict[str, Any]:
    """Get information about a sandbox instance.
    
    Args:
        sandbox: E2B Sandbox or AsyncSandbox instance
        
    Returns:
        Dict containing sandbox information
    """
    try:
        info = {
            "type": type(sandbox).__name__,
            "id": getattr(sandbox, 'id', 'unknown'),
            "status": "active" if sandbox else "inactive"
        }
        
        # Add any additional sandbox-specific information
        if hasattr(sandbox, 'template'):
            info["template"] = sandbox.template
            
        logger.debug(f"Sandbox info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get sandbox info: {e}")
        return {"error": str(e)}