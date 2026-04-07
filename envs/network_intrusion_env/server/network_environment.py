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
    (False, False): 0.05
}

CORRECT_RESPONSES = {
    "ddos": "block_ip",
    "port_scan": "alert_only",
    "brute_force": "block_ip",
    "normal": "ignore"
}

INTERNAL_IPS = ["10.0.0.", "172.16.0.", "192.168.1.", "10.10.10."]
EXTERNAL_IPS = ["198.51.100.", "203.0.113.", "192.0.2.", "8.8.8.", "1.1.1."]
COMMON_PORTS = [80, 443, 8080, 8443]
AUTH_PORTS = [22, 23, 3389, 5900, 21]
ALL_PORTS = list(range(1, 1025))


def _random_ip(pool):
    prefix = random.choice(pool)
    return prefix + str(random.randint(1, 254))


def generate_scenario():
    threat = random.choice(VALID_THREATS)
    scenario_id = f"{threat}_{random.randint(1000, 9999)}"

    if threat == "ddos":
        rps = random.randint(3000, 25000)
        unique_ips = random.randint(1, 8)
        ports = random.sample(COMMON_PORTS, random.randint(1, 2))
        packet_size = random.randint(40, 100)
        duration = random.randint(30, 300)
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = ["high_request_rate"]
        if unique_ips <= 3:
            flags.append("few_sources")
        if packet_size < 70:
            flags.append("small_packets")
        if random.random() < 0.2:
            flags.append("known_cdn_range")

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
        flags = ["single_source"]
        if len(ports) > 100:
            flags.append("sequential_ports")
        else:
            flags.append("targeted_port_probe")
        if packet_size < 50:
            flags.append("low_packet_size")
        if random.random() < 0.2:
            flags.append("low_severity_indicator")

    elif threat == "brute_force":
        rps = random.randint(10, 200)
        unique_ips = 1
        ports = [random.choice(AUTH_PORTS)]
        packet_size = random.randint(100, 250)
        duration = random.randint(120, 900)
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = ["repeated_auth_attempts", "single_port"]
        if ports[0] == 3389:
            flags.append("rdp_port")
        elif ports[0] == 22:
            flags.append("ssh_port")
        elif ports[0] == 21:
            flags.append("ftp_port")
        if random.random() < 0.25:
            flags.append("geoip_suspicious_region")

    else:  # normal
        rps = random.randint(1, 500)
        unique_ips = random.randint(10, 300)
        ports = random.sample(COMMON_PORTS, random.randint(1, len(COMMON_PORTS)))
        packet_size = random.randint(200, 1500)
        duration = random.randint(600, 86400)
        source_ip = _random_ip(EXTERNAL_IPS)
        flags = []
        if rps > 300:
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
        "anomaly_flags": flags
    }

    return {
        "id": scenario_id,
        "logs": logs,
        "threat_type": threat,
        "correct_response": CORRECT_RESPONSES[threat]
    }


# Module-level state
_current_scenario = None
_investigation_log = []


class NetworkIntrusionEnvironment(MCPEnvironment):

    def __init__(self):
        mcp = FastMCP("network_intrusion_env")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_reward = 0.0
        self._last_correct_threat = False
        self._last_correct_response = False

        @mcp.tool
        def get_scenario() -> dict:
            """
            Returns a high-level summary of the current network scenario.
            Does NOT include anomaly_flags or port details.
            Use investigate() with 'ports', 'packets', or 'flags' to dig deeper.
            """
            if _current_scenario is None:
                return {"error": "Call reset first."}
            logs = _current_scenario["logs"]
            return {
                "scenario_id": _current_scenario["id"],
                "source_ip": logs["source_ip"],
                "requests_per_second": logs["requests_per_second"],
                "unique_ips": logs["unique_ips"],
                "duration_seconds": logs["duration_seconds"],
                "hint": "Use investigate() with 'ports', 'packets', or 'flags' to learn more."
            }

        @mcp.tool
        def investigate(target: str) -> dict:
            """
            Investigate a specific aspect of the current network scenario.
            target must be one of: 'ports', 'packets', 'flags'
            Each reveals different clues. Call up to 3 times before submitting.
            """
            global _investigation_log

            if _current_scenario is None:
                return {"error": "Call reset first."}

            logs = _current_scenario["logs"]
            _investigation_log.append(target)

            if target == "ports":
                ports = logs["ports_targeted"]
                port_count = len(ports)
                is_sequential = (
                    port_count > 10 and
                    ports == list(range(ports[0], ports[0] + port_count))
                )
                if is_sequential:
                    port_summary = f"ports {ports[0]} to {ports[-1]} — sequential sweep of {port_count} ports"
                elif port_count <= 10:
                    port_summary = str(ports)
                else:
                    port_summary = f"{port_count} ports including: {ports[:5]} ... {ports[-5:]}"
                return {
                    "ports_targeted_summary": port_summary,
                    "port_count": port_count,
                    "sequential": is_sequential,
                    "note": "SSH=22, RDP=3389, FTP=21 suggest brute_force. Sequential sweep of many ports = port_scan. 80/443 = ddos or normal."
                }
            elif target == "packets":
                return {
                    "packet_size_avg": logs["packet_size_avg"],
                    "note": "Small packets (<100 bytes) typical of ddos/scan. Large packets (>200 bytes) suggest real sessions like brute_force or normal."
                }
            elif target == "flags":
                return {
                    "anomaly_flags": logs["anomaly_flags"],
                    "note": "System-detected indicators. May occasionally include false positives."
                }
            else:
                return {
                    "error": f"Unknown target '{target}'. Use: 'ports', 'packets', or 'flags'"
                }

        @mcp.tool
        def submit_analysis(threat_type: str, response_action: str) -> dict:
            """
            Submit your threat classification and recommended response action.
            threat_type: one of ddos, port_scan, brute_force, normal
            response_action: one of block_ip, rate_limit, alert_only, ignore
            """
            global _investigation_log

            if _current_scenario is None:
                return {"error": "Call reset first."}

            t = threat_type.lower().strip()
            r = response_action.lower().strip()

            if t not in VALID_THREATS:
                return {"error": f"Invalid threat_type. Must be one of: {VALID_THREATS}"}
            if r not in VALID_RESPONSES:
                return {"error": f"Invalid response_action. Must be one of: {VALID_RESPONSES}"}

            threat_correct = t == _current_scenario["threat_type"]
            response_correct = r == _current_scenario["correct_response"]
            base_reward = REWARD_TABLE[(threat_correct, response_correct)]

            # Efficiency bonus/penalty based on investigation steps
            steps = len(_investigation_log)
            efficiency_modifier = 0.0
            if threat_correct and response_correct and 1 <= steps <= 2:
                efficiency_modifier = 0.05
            elif not (threat_correct and response_correct) and steps >= 3:
                efficiency_modifier = -0.05

            final_reward = min(1.0, max(0.0, base_reward + efficiency_modifier))

            # Reset investigation log
            _investigation_log = []

            return {
                "reward": final_reward,
                "threat_correct": threat_correct,
                "response_correct": response_correct,
                "correct_threat": _current_scenario["threat_type"],
                "correct_response": _current_scenario["correct_response"],
                "investigations_used": steps,
                "efficiency_note": f"Used {steps} investigate() call(s)."
            }

        super().__init__(mcp)

    def _get_feedback(self, correct_threat: bool, correct_response: bool) -> str:
        if correct_threat and correct_response:
            return "Perfect analysis."
        elif correct_threat and not correct_response:
            return "Threat correct but response not optimal."
        elif not correct_threat and correct_response:
            return "Response appropriate but threat misclassified."
        else:
            return "Both threat and response incorrect."

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        global _current_scenario, _investigation_log
        if seed is not None:
            random.seed(seed)
        _current_scenario = generate_scenario()
        _investigation_log = []
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._last_reward = 0.0
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "Network intrusion environment ready. Call get_scenario to begin."
            }
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools."}
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