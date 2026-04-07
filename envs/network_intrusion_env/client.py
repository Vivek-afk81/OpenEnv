"""
Network Intrusion Detection Environment Client.

Example:
    >>> with NetworkIntrusionEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     scenario = env.call_tool("get_scenario")
    ...     result = env.call_tool("submit_analysis", threat_type="ddos", response_action="block_ip")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class NetworkIntrusionEnv(MCPToolClient):
    """
    Client for the Network Intrusion Detection Environment.

    Tools available:
    - get_scenario(): Get current network log snapshot to analyze
    - submit_analysis(threat_type, response_action): Submit verdict and get reward
    """
    pass