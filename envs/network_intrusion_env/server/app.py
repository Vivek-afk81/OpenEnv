"""
FastAPI application for the Network Intrusion Detection Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .network_environment import NetworkIntrusionEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.network_environment import NetworkIntrusionEnvironment

app = create_app(
    NetworkIntrusionEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="network_intrusion_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()