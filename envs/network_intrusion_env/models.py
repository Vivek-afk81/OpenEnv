from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class NetworkIntrusionAction(Action):
    """Action submitted by the agent to classify a network threat."""
    tool_name: str = Field(..., description="Tool to call: get_scenario, investigate, or submit_analysis")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool")


class NetworkIntrusionObservation(Observation):
    """Observation returned by the environment after each step."""
    result: dict = Field(default_factory=dict, description="Tool result data")
    success: bool = Field(default=True, description="Whether the tool call succeeded")


class NetworkIntrusionState(State):
    """Current state of the network intrusion environment."""
    scenario_id: str = Field(default="", description="ID of the current scenario")
    investigations_used: int = Field(default=0, description="Number of investigate() calls so far")
    episode_active: bool = Field(default=False, description="Whether an episode is in progress")