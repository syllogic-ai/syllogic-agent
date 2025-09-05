"""Widget ordering functionality using LLM-based analysis with Langfuse integration."""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from supabase import Client

from agent.models import WidgetOrderingSchema, WidgetOrder
from .dashboard import (
    get_dashboard_widgets_for_ordering,
    update_widgets_order_and_configuration,
)
from config import get_langfuse_callback_handler, LANGFUSE_AVAILABLE

# Handle imports for different execution contexts
try:
    from actions.prompts import compile_prompt, get_prompt_config
except ImportError:
    import sys
    import os

    # Add the src directory to the path
    src_path = os.path.join(os.path.dirname(__file__), "..")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from actions.prompts import compile_prompt, get_prompt_config

logger = logging.getLogger(__name__)


def analyze_and_order_dashboard_widgets(
    dashboard_id: str, new_widget_ids: List[str], chat_id: str, user_id: str = None
) -> Dict[str, Any]:
    """Analyze dashboard widgets and determine optimal ordering using LLM.

    Args:
        dashboard_id: Dashboard identifier
        new_widget_ids: List of newly created widget IDs to be ordered
        chat_id: Chat session ID for tracing
        user_id: User ID for tracing

    Returns:
        Dict with success status, updated widgets, and reasoning

    Raises:
        Exception: If ordering or database update fails
    """
    try:
        logger.info(
            f"Starting widget ordering analysis for dashboard {dashboard_id} with {len(new_widget_ids)} new widgets"
        )

        # Step 1: Fetch existing widgets from dashboard
        existing_widgets = get_dashboard_widgets_for_ordering(dashboard_id)

        # Step 2: Separate existing widgets from new widgets
        existing_configured = [
            w for w in existing_widgets if w["id"] not in new_widget_ids
        ]
        new_widgets = [w for w in existing_widgets if w["id"] in new_widget_ids]

        if not new_widgets:
            logger.warning(
                f"No new widgets found to order from provided IDs: {new_widget_ids}"
            )
            return {
                "success": False,
                "message": "No new widgets found to order",
                "reasoning": "",
                "updated_count": 0,
            }

        logger.info(
            f"Found {len(existing_configured)} existing configured widgets and {len(new_widgets)} new widgets to order"
        )
        
        # Debug: Log widget data being analyzed
        logger.info("📊 Widget data for LLM analysis:")
        logger.info(f"   Dashboard: {dashboard_id}")
        logger.info(f"   Existing widgets: {existing_configured}")
        logger.info(f"   New widgets: {new_widgets}")
        if new_widgets:
            for widget in new_widgets:
                logger.info(f"      New widget {widget.get('id', 'unknown')}: {widget.get('widget_type', 'unknown')} - {widget.get('summary', 'no summary')[:100]}...")

        # Step 3: Prepare data for LLM analysis
        widget_analysis_data = {
            "dashboard_id": dashboard_id,
            "existing_widgets": existing_configured,
            "new_widgets": new_widgets,
            "total_widgets": len(existing_configured) + len(new_widgets),
            "new_widgets_count": len(new_widgets),
            "existing_widgets_count": len(existing_configured),
        }

        # Step 4: Get LLM ordering recommendations
        ordering_result = _get_llm_widget_ordering(
            widget_analysis_data, chat_id, user_id
        )

        if not ordering_result["success"]:
            return ordering_result

        # Step 5: Apply the ordering recommendations
        update_result = _apply_widget_ordering(ordering_result["widget_orders"])

        if update_result["success"]:
            logger.info(
                f"Successfully ordered and configured {update_result['updated_count']} widgets"
            )
            return {
                "success": True,
                "message": f"Successfully ordered and configured {update_result['updated_count']} widgets",
                "reasoning": ordering_result["reasoning"],
                "updated_count": update_result["updated_count"],
                "widget_orders": ordering_result["widget_orders"],
            }
        else:
            return {
                "success": False,
                "message": f"LLM analysis succeeded but database update failed: {update_result['message']}",
                "reasoning": ordering_result["reasoning"],
                "updated_count": 0,
            }

    except Exception as e:
        error_msg = f"Error in widget ordering analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "reasoning": "",
            "updated_count": 0,
        }


def _get_llm_widget_ordering(
    widget_data: Dict[str, Any], chat_id: str, user_id: str = None
) -> Dict[str, Any]:
    """Use LLM to analyze widgets and determine optimal ordering.

    Args:
        widget_data: Dictionary containing existing and new widgets data
        chat_id: Chat session ID for tracing
        user_id: User ID for tracing

    Returns:
        Dict with success status, reasoning, and widget_orders
    """
    try:
        # Fetch model configuration from Langfuse (REQUIRED)
        logger.info("Fetching model configuration from Langfuse for widget ordering...")
        prompt_config = get_prompt_config("top_level_supervisor/tools/widget_ordering", label="latest")

        # Extract required model and temperature from Langfuse config
        model = prompt_config.get("model")
        temperature = prompt_config.get("temperature")
        reasoning_effort = prompt_config.get("reasoning_effort")

        # Validate required configuration
        if not model:
            raise ValueError(
                "Model configuration is missing or empty in Langfuse prompt config"
            )
        if temperature is None:
            raise ValueError(
                "Temperature configuration is missing in Langfuse prompt config"
            )

        logger.info(
            f"✅ Using Langfuse model config - model: {model}, temperature: {temperature}, reasoning_effort: {reasoning_effort}"
        )

        # Prepare prompt variables
        prompt_variables = {
            "dashboard_id": widget_data["dashboard_id"],
            "existing_widgets": json.dumps(widget_data["existing_widgets"], indent=2),
            "new_widgets": json.dumps(widget_data["new_widgets"], indent=2),
            "total_widgets": widget_data["total_widgets"],
            "new_widgets_count": widget_data["new_widgets_count"],
            "existing_widgets_count": widget_data["existing_widgets_count"],
            "current_timestamp": datetime.now().isoformat(),
        }

        # Compile the prompt with dynamic variables from Langfuse (REQUIRED)
        logger.info("Compiling widget ordering prompt from Langfuse...")
        ordering_prompt = compile_prompt(
            "top_level_supervisor/tools/widget_ordering", prompt_variables, label="latest"
        )

        # Validate compiled prompt
        if not ordering_prompt:
            raise ValueError(
                "Compiled widget ordering prompt from Langfuse is empty or None"
            )

        # Handle chat message format from Langfuse
        if isinstance(ordering_prompt, list) and len(ordering_prompt) > 0:
            # Extract content from chat message format
            first_message = ordering_prompt[0]
            if isinstance(first_message, dict) and 'content' in first_message:
                ordering_prompt_content = first_message['content']
            else:
                ordering_prompt_content = str(first_message)
        else:
            ordering_prompt_content = str(ordering_prompt)
        
        if not ordering_prompt_content or len(ordering_prompt_content.strip()) == 0:
            raise ValueError(
                "Compiled widget ordering prompt from Langfuse is empty or invalid"
            )
        
        logger.info(
            f"✅ Extracted prompt content: {len(ordering_prompt_content)} characters"
        )

        logger.info(
            f"✅ Successfully compiled Langfuse widget ordering prompt with {len(prompt_variables)} variables"
        )

        # Initialize LLM with Langfuse configuration
        llm_params = {"model": model, "temperature": temperature}

        # Add reasoning_effort if provided (for reasoning models like o1, o3, o4-mini)
        if reasoning_effort:
            llm_params["reasoning_effort"] = reasoning_effort

        llm = ChatOpenAI(**llm_params)
        structured_llm = llm.with_structured_output(WidgetOrderingSchema)

        # Create Langfuse callback handler for widget ordering analysis
        ordering_config = {}
        if LANGFUSE_AVAILABLE:
            try:
                langfuse_handler = get_langfuse_callback_handler(
                    trace_name="widget-ordering-analysis",
                    session_id=chat_id,
                    user_id=user_id,
                    tags=["widget-ordering", "llm-analysis", "dashboard"],
                    metadata={
                        "dashboard_id": widget_data["dashboard_id"],
                        "total_widgets": widget_data["total_widgets"],
                        "new_widgets_count": widget_data["new_widgets_count"],
                        "existing_widgets_count": widget_data["existing_widgets_count"],
                    },
                )
                if langfuse_handler:
                    ordering_config = {"callbacks": [langfuse_handler]}
            except Exception as langfuse_error:
                logger.warning(
                    f"Failed to create Langfuse handler for widget ordering: {langfuse_error}"
                )

        # Get structured ordering response from LLM
        try:
            logger.info("🤖 Invoking LLM for widget ordering analysis...")
            logger.info(f"   Model: {model}, Temperature: {temperature}")
            logger.info(f"   Prompt length: {len(ordering_prompt_content)} characters")
            logger.info(f"   Using Langfuse callbacks: {bool(ordering_config)}")
            
            if ordering_config:
                response = structured_llm.invoke(
                    ordering_prompt_content, config=ordering_config
                )
            else:
                response = structured_llm.invoke(ordering_prompt_content)

            logger.info("✅ Successfully received widget ordering analysis from LLM")
            logger.info(f"   Response type: {type(response)}")
            logger.info(f"   Has reasoning: {hasattr(response, 'reasoning')}")
            logger.info(f"   Has widget_orders: {hasattr(response, 'widget_orders')}")
            
            if hasattr(response, 'widget_orders'):
                logger.info(f"   Widget orders count: {len(response.widget_orders) if response.widget_orders else 0}")
                
        except Exception as llm_error:
            logger.error(f"❌ LLM invocation failed: {str(llm_error)}")
            logger.error(f"   Error type: {type(llm_error)}")
            if hasattr(llm_error, 'response'):
                logger.error(f"   LLM response: {llm_error.response}")
            raise llm_error

        return {
            "success": True,
            "reasoning": response.reasoning,
            "widget_orders": response.widget_orders,
        }

    except Exception as e:
        error_msg = f"Error in LLM widget ordering analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "reasoning": "",
            "widget_orders": [],
        }


def _apply_widget_ordering(widget_orders: List[WidgetOrder]) -> Dict[str, Any]:
    """Apply the LLM-recommended widget ordering to the database using bulk operations.

    Args:
        widget_orders: List of WidgetOrder objects with 'id' and 'order' attributes

    Returns:
        Dict with success status and updated count
    """
    try:
        if not widget_orders:
            return {
                "success": True,
                "updated_count": 0,
                "message": "No widget orders to apply"
            }

        # Convert WidgetOrder objects to dict format for bulk update
        widget_updates = []
        for order_obj in widget_orders:
            widget_updates.append({
                "id": order_obj.id,
                "order": order_obj.order,
                "is_configured": True,  # Mark all updated widgets as configured
            })

        logger.info(f"Applying bulk widget ordering for {len(widget_updates)} widgets")

        # Apply updates to database using optimized bulk operation
        update_result = update_widgets_order_and_configuration(widget_updates)

        return update_result

    except Exception as e:
        error_msg = f"Error applying widget ordering: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "updated_count": 0, "message": error_msg}
