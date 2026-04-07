# Network Intrusion Detection Environment
# Multi-step RL environment where agent investigates before classifying

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP


VALID_THREATS = ["ddos", "port_scan", "brute_force", "normal"]
VALID_RESPONSES = ["block_ip", "rate_limit", "alert_only", "ignore"]

REWARD_TABLE = {
    (True, True): 1.0,
    (True, False): 0.6,
    (False, True): 0.3,
    (False, False): 0.05,
}

# rate_limit is now used for normal — high-traffic but legitimate
CORRECT_RESPONSES = {
    "ddos": "block_ip",
    "port_scan": "alert_only",
    "brute_force": "block_ip",
    "normal": "ignore",
}

INTERNAL_IPS = ["10.0.0.", "172.16.0.", "192.168.1.", "10.10.10."]
EXTERNAL_IPS = ["198.51.100.", "203.0.113.", "192.0.2.", "8.8.8.", "1.1.1."]
COMMON_PORTS = [80, 443, 8080, 8443]
AUTH_PORTS = [22, 23, 3389, 5900, 21]
ALL_PORTS = list(range(1, 1025))

MAX_INVESTIGATIONS = 3


def _random_ip(pool):
    prefix = random.choice(pool)
    return prefix + str(random.randint(1, 254))


def generate_scenario():
    threat = random.choice(VALID_THREATS)
    scenario_id = f"{threat}_{random.randint(1000, 9999)}"

    if threat == "ddos":
        rps = random.randint(3000, 25000)
        unique_ips = random.randint(1, 8)          # key signal: very few sources
        ports = random.sample(COMMON_PORTS, random.randint(1, 2))
        packet_size = random.randint(40, 100)
        duration = random.randint(30, 300)         # short burst
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = ["high_request_rate", "concentrated_sources"]
        if packet_size < 70:
            flags.append("small_packets")
        if random.random() < 0.2:
            flags.append("known_bad_asn")

    elif threat == "port_scan":
        rps = random.randint(10, 300)
        unique_ips = 1
        if random.random() < 0.5:
            num_ports = random.randint(500, 1024)
            ports = list(range(1, num_ports + 1))
        else:
            ports = sorted(random.sample(ALL_PORTS, random.randint(8, 50)))
        packet_size = random.randint(36, 60)
        duration = random.randint(15, 120)
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = ["single_source", "probe_pattern"]
        if len(ports) > 100:
            flags.append("wide_port_sweep")
        if random.random() < 0.2:
            flags.append("low_severity_indicator")

    elif threat == "brute_force":
        rps = random.randint(10, 200)
        unique_ips = 1
        ports = [random.choice(AUTH_PORTS)]
        packet_size = random.randint(100, 250)
        duration = random.randint(120, 900)        # sustained attack
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = ["repeated_auth_attempts", "single_port_focus"]
        if ports[0] == 3389:
            flags.append("rdp_targeted")
        elif ports[0] == 22:
            flags.append("ssh_targeted")
        elif ports[0] == 21:
            flags.append("ftp_targeted")
        if random.random() < 0.25:
            flags.append("geoip_suspicious_region")

    else:  # normal
        rps = random.randint(1, 400)               # kept below ddos floor
        unique_ips = random.randint(50, 500)       # key signal: many distributed sources
        ports = random.sample(COMMON_PORTS, random.randint(1, len(COMMON_PORTS)))
        packet_size = random.randint(300, 1500)    # large packets = real sessions
        duration = random.randint(3600, 86400)     # hours-long = legitimate traffic
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = []
        if rps > 250:
            flags.append("elevated_but_distributed")
        if random.random() < 0.3:
            flags.append("cdn_traffic")
        if random.random() < 0.15:
            flags.append("minor_anomaly_detected")

    logs = {
        "source_ip": source_ip,
        "requests_per_second": rps,
        "unique_ips": unique_ips,
        "ports_targeted": ports,
        "packet_size_avg": packet_size,
        "duration_seconds": duration,
        "anomaly_flags": flags,
    }

    return {
        "id": scenario_id,
        "logs": logs,
        "threat_type": threat,
        "correct_response": CORRECT_RESPONSES[threat],
    }


class NetworkIntrusionEnvironment(MCPEnvironment):

    def __init__(self):
        mcp = FastMCP("network_intrusion_env")

        # Instance state — safe for concurrent/repeated use
        self._current_scenario = None
        self._investigation_log = []   # tracks targets already investigated
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_reward = 0.0

        env_self = self  # capture for closures below

        @mcp.tool
        def get_scenario() -> dict:
            """
            Returns a high-level summary of the current network scenario.
            Does NOT include port details, packet info, or anomaly flags.
            Use investigate() to dig deeper before submitting your analysis.
            """
            if env_self._current_scenario is None:
                return {"error": "Call reset first to start a new episode."}

            logs = env_self._current_scenario["logs"]
            return {
                "scenario_id": env_self._current_scenario["id"],
                "source_ip": logs["source_ip"],
                "requests_per_second": logs["requests_per_second"],
                "unique_ips": logs["unique_ips"],
                "duration_seconds": logs["duration_seconds"],
                "available_investigations": ["ports", "packets", "flags"],
                "investigations_remaining": MAX_INVESTIGATIONS - len(env_self._investigation_log),
            }

        @mcp.tool
        def investigate(target: str) -> dict:
            """
            Investigate one specific aspect of the current network scenario.
            target must be one of: 'ports', 'packets', 'flags'
            Maximum 3 investigations per episode. Cannot repeat a target.
            """
            if env_self._current_scenario is None:
                return {"error": "Call reset first to start a new episode."}

            if len(env_self._investigation_log) >= MAX_INVESTIGATIONS:
                return {
                    "error": f"Investigation limit reached ({MAX_INVESTIGATIONS}). You must submit your analysis now.",
                    "investigations_used": env_self._investigation_log,
                }

            if target in env_self._investigation_log:
                return {
                    "error": f"You already investigated '{target}'. Choose a different target.",
                    "already_investigated": env_self._investigation_log,
                    "available": [t for t in ["ports", "packets", "flags"] if t not in env_self._investigation_log],
                }

            if target not in ["ports", "packets", "flags"]:
                return {"error": f"Unknown target '{target}'. Must be one of: ports, packets, flags"}

            env_self._investigation_log.append(target)
            logs = env_self._current_scenario["logs"]

            if target == "ports":
                ports = logs["ports_targeted"]
                port_count = len(ports)
                is_sequential = (
                    port_count > 10
                    and ports == list(range(ports[0], ports[0] + port_count))
                )
                if is_sequential:
                    port_summary = f"sequential sweep from port {ports[0]} to {ports[-1]} ({port_count} ports total)"
                elif port_count <= 10:
                    port_summary = f"specific ports: {ports}"
                else:
                    port_summary = f"{port_count} ports, sample: {ports[:5]} ... {ports[-5:]}"

                return {
                    "ports_targeted_summary": port_summary,
                    "port_count": port_count,
                    "sequential_sweep": is_sequential,
                    "investigations_remaining": MAX_INVESTIGATIONS - len(env_self._investigation_log),
                }

            elif target == "packets":
                return {
                    "packet_size_avg_bytes": logs["packet_size_avg"],
                    "investigations_remaining": MAX_INVESTIGATIONS - len(env_self._investigation_log),
                }

            elif target == "flags":
                return {
                    "anomaly_flags": logs["anomaly_flags"],
                    "flag_count": len(logs["anomaly_flags"]),
                    "investigations_remaining": MAX_INVESTIGATIONS - len(env_self._investigation_log),
                }

        @mcp.tool
        def submit_analysis(threat_type: str, response_action: str) -> dict:
            """
            Submit your threat classification and recommended response.
            threat_type: one of ddos, port_scan, brute_force, normal
            response_action: one of block_ip, rate_limit, alert_only, ignore
            """
            if env_self._current_scenario is None:
                return {"error": "Call reset first to start a new episode."}

            t = threat_type.lower().strip()
            r = response_action.lower().strip()

            if t not in VALID_THREATS:
                return {"error": f"Invalid threat_type '{t}'. Must be one of: {VALID_THREATS}"}
            if r not in VALID_RESPONSES:
                return {"error": f"Invalid response_action '{r}'. Must be one of: {VALID_RESPONSES}"}

            threat_correct = t == env_self._current_scenario["threat_type"]
            response_correct = r == env_self._current_scenario["correct_response"]
            base_reward = REWARD_TABLE[(threat_correct, response_correct)]

            # Efficiency: reward smart investigation (1-2 steps on correct), penalise waste on wrong
            steps = len(env_self._investigation_log)
            efficiency_modifier = 0.0
            if threat_correct and response_correct:
                if 1 <= steps <= 2:
                    efficiency_modifier = 0.05   # bonus for efficient correct analysis
            else:
                if steps == MAX_INVESTIGATIONS:
                    efficiency_modifier = -0.05  # used all steps and still wrong

            final_reward = min(1.0, max(0.0, base_reward + efficiency_modifier))
            env_self._last_reward = final_reward

            # Capture before reset
            correct_threat = env_self._current_scenario["threat_type"]
            correct_response = env_self._current_scenario["correct_response"]
            investigations_used = steps

            # Reset episode state
            env_self._investigation_log = []
            env_self._current_scenario = None

            return {
                "reward": final_reward,
                "threat_correct": threat_correct,
                "response_correct": response_correct,
                "correct_threat": correct_threat,
                "correct_response": correct_response,
                "investigations_used": investigations_used,
            }

        super().__init__(mcp)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        if seed is not None:
            random.seed(seed)
        self._current_scenario = generate_scenario()
        self._investigation_log = []
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_reward = 0.0
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "New episode started. Call get_scenario to begin.",
            },
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools."},
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state