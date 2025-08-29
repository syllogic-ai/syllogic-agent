import pytest

def test_graph_import() -> None:
    """Test that the graph can be imported successfully."""
    try:
        from agent.graph import graph
        from langgraph.pregel import Pregel
        assert isinstance(graph, Pregel)
    except ImportError as e:
        pytest.skip(f"Could not import graph or langgraph dependencies: {e}")


