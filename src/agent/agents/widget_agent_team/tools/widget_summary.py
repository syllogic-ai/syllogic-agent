"""Widget summary generation tool for creating VLLM-friendly widget descriptions."""

import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from actions.prompts import get_prompt_with_fallback, get_prompt_config

# Get logger that uses Logfire if available
try:
    from config import get_logfire_logger
    logger = get_logfire_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def generate_widget_summary(
    widget_type: str,
    title: str,
    description: str,
    widget_config: Dict[str, Any],
    tool_call_id: str = "widget_summary_generation"
) -> str:
    """
    Generate a brief, accessible summary of a widget for VLLMs.
    
    This function creates a 2-3 sentence summary that describes what the widget
    contains in a way that someone who cannot see the dashboard would understand.
    
    Args:
        widget_type: Type of widget (e.g., 'bar', 'line', 'text_block')
        title: Widget title
        description: Widget description 
        widget_config: Complete widget configuration including data
        tool_call_id: Unique identifier for this tool call
        
    Returns:
        Brief widget summary string
        
    Raises:
        Exception: If summary generation fails
    """
    try:
        # Compile dynamic variables for Langfuse prompt
        dynamic_variables = {
            "widget_type": widget_type,
            "widget_title": title,
            "widget_description": description or "No description provided",
            "widget_config": json.dumps(widget_config, indent=2) if widget_config else "{}",
            "data_preview": _extract_data_preview(widget_config),
            "chart_categories": _extract_chart_categories(widget_config, widget_type),
            "value_ranges": _extract_value_ranges(widget_config, widget_type),
            "content_type": _determine_content_type(widget_type, widget_config),
            "accessibility_context": "This summary will be used by VLLMs and should be understandable to someone who cannot see the visual dashboard"
        }
        
        # Get prompt from Langfuse with fallback
        fallback_prompt = """Create a brief 2-3 sentence summary of this widget that describes what it contains in an accessible way.

Widget Type: {widget_type}
Title: {widget_title}
Description: {widget_description}
Config: {widget_config}

Context: {accessibility_context}

The summary should mention:
- What type of visualization or content this is
- What categories/data points are included
- Key patterns or insights if apparent from the data
- For text blocks, what the main message or content is about

Keep it concise but informative for someone who cannot see the dashboard."""

        compiled_prompt = get_prompt_with_fallback(
            "widget-summary-generation",
            fallback_prompt,
            dynamic_variables
        )
        
        # Get model configuration from Langfuse with fallback defaults
        default_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.3
        }
        prompt_config = get_prompt_config("widget-summary-generation", default_config)
        model_name = prompt_config.get("model", "gpt-4o-mini")
        temperature = prompt_config.get("temperature", 0.3)
        reasoning_effort = prompt_config.get("reasoning_effort")
        
        try:
            # Initialize LLM with Langfuse configuration
            llm_params = {
                "model": model_name,
                "temperature": temperature
            }
            
            # Add reasoning_effort if provided
            if reasoning_effort:
                llm_params["reasoning_effort"] = reasoning_effort
                
            model = ChatOpenAI(**llm_params)
            
        except Exception as e:
            logger.warning(f"Failed to get model config from Langfuse, using default: {e}")
            # Fallback to default model
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Generate summary using the model
        response = model.invoke(compiled_prompt)
        summary = response.content.strip()
        
        # Validate summary length (should be brief)
        if len(summary) > 500:
            logger.warning(f"Generated summary is quite long ({len(summary)} chars), consider shortening")
            
        logger.info(f"Successfully generated widget summary for {widget_type} widget: {title}")
        return summary
        
    except Exception as e:
        error_msg = f"Widget summary generation failed: {str(e)}"
        logger.error(error_msg)
        
        # Fallback to basic summary if LLM fails
        fallback_summary = _create_fallback_summary(widget_type, title, description, widget_config)
        logger.info("Using fallback summary due to generation error")
        return fallback_summary


def _extract_data_preview(widget_config: Dict[str, Any]) -> str:
    """Extract a preview of the data for the prompt."""
    if not widget_config or "data" not in widget_config:
        return "No data available"
    
    data = widget_config.get("data", [])
    if not data:
        return "No data available"
    
    # Show first few rows for context
    if isinstance(data, list) and len(data) > 0:
        preview_count = min(3, len(data))
        preview_data = data[:preview_count]
        return f"First {preview_count} data points: {json.dumps(preview_data, indent=2)}"
    
    return "Data structure not in expected format"


def _extract_chart_categories(widget_config: Dict[str, Any], widget_type: str) -> str:
    """Extract chart categories/labels from the config."""
    if widget_type == "text_block" or not widget_config:
        return "Not applicable for this widget type"
    
    data = widget_config.get("data", [])
    if not data or not isinstance(data, list):
        return "No categories identified"
    
    # Try to find common category fields
    categories = []
    for item in data[:5]:  # Check first 5 items
        if isinstance(item, dict):
            # Look for common category field names
            for key in ["category", "label", "name", "x", "key"]:
                if key in item:
                    categories.append(str(item[key]))
                    break
    
    if categories:
        return f"Categories include: {', '.join(categories[:10])}{'...' if len(categories) > 10 else ''}"
    
    return "Categories not clearly identifiable"


def _extract_value_ranges(widget_config: Dict[str, Any], widget_type: str) -> str:
    """Extract value ranges from numeric data."""
    if widget_type == "text_block" or not widget_config:
        return "Not applicable for this widget type"
    
    data = widget_config.get("data", [])
    if not data or not isinstance(data, list):
        return "No values identified"
    
    # Try to find numeric values
    values = []
    for item in data[:10]:  # Check first 10 items
        if isinstance(item, dict):
            # Look for common value field names
            for key in ["value", "y", "count", "amount", "total"]:
                if key in item and isinstance(item[key], (int, float)):
                    values.append(float(item[key]))
                    break
    
    if values:
        min_val = min(values)
        max_val = max(values)
        return f"Values range from {min_val:,.0f} to {max_val:,.0f}"
    
    return "Value ranges not clearly identifiable"


def _determine_content_type(widget_type: str, widget_config: Dict[str, Any]) -> str:
    """Determine the content type for better description."""
    if widget_type == "text_block":
        # For text blocks, try to identify the content type
        html_content = widget_config.get("html", "") if widget_config else ""
        if "table" in html_content.lower():
            return "tabular content"
        elif "list" in html_content.lower() or "<li>" in html_content:
            return "list content"
        elif len(html_content) > 200:
            return "detailed text content"
        else:
            return "text content"
    
    # For chart types
    chart_type_map = {
        "bar": "bar chart visualization",
        "line": "line chart visualization", 
        "pie": "pie chart visualization",
        "scatter": "scatter plot visualization",
        "area": "area chart visualization",
        "histogram": "histogram visualization"
    }
    
    return chart_type_map.get(widget_type, f"{widget_type} chart visualization")


def _create_fallback_summary(
    widget_type: str,
    title: str,
    description: str,
    widget_config: Dict[str, Any]
) -> str:
    """Create a basic fallback summary when LLM generation fails."""
    content_type = _determine_content_type(widget_type, widget_config)
    
    summary_parts = [f"This is a {content_type} titled '{title}'."]
    
    if description and description.strip():
        summary_parts.append(f"Description: {description[:100]}{'...' if len(description) > 100 else ''}.")
    
    # Add basic data info if available
    if widget_config and "data" in widget_config:
        data = widget_config["data"]
        if isinstance(data, list) and data:
            summary_parts.append(f"Contains {len(data)} data points.")
    
    return " ".join(summary_parts)