"""
Network Intrusion Detection Environment.

An RL environment where an agent analyzes network traffic logs,
classifies threats, and recommends response actions.

Tools available:
- `get_scenario()`: Get current network log snapshot to analyze
- `submit_analysis(threat_type, response_action)`: Submit verdict and get reward

Example:
    >>> from network_intrusion_env import NetworkIntrusionEnv
    >>>
    >>> with NetworkIntrusionEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     scenario = env.call_tool("get_scenario")
    ...     result = env.call_tool("submit_analysis", threat_type="ddos", response_action="block_ip")
    ...     print(result)
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from .client import NetworkIntrusionEnv

__all__ = ["NetworkIntrusionEnv", "CallToolAction", "ListToolsAction"]