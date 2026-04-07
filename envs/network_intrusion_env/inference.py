"""
Inference script for Network Intrusion Detection Environment.
Multi-step: agent investigates clues before classifying the threat.
"""

import json
import os
import httpx
from openai import OpenAI

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

openai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

VALID_THREATS = ["ddos", "port_scan", "brute_force", "normal"]
VALID_RESPONSES = ["block_ip", "rate_limit", "alert_only", "ignore"]

# Ground-truth mapping the agent must learn from evidence alone
THREAT_RESPONSE_MAP = {
    "ddos": "block_ip",
    "port_scan": "alert_only",
    "brute_force": "block_ip",
    "normal": "ignore",
}


def call_llm(prompt: str, max_tokens: int = 200) -> str:
    """Single LLM call. Returns raw text response."""
    response = openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content


def parse_json_answer(raw: str) -> dict | None:
    """
    Robustly extract JSON from LLM output.
    Handles markdown fences, extra text, etc.
    Returns None if parsing fails completely.
    """
    # Strip markdown fences
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find first JSON object in the string
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def get_investigation_target(scenario_summary: dict, already_investigated: list[str]) -> str:
    """Ask LLM which aspect to investigate next."""
    available = [t for t in ["ports", "packets", "flags"] if t not in already_investigated]
    if not available:
        return None

    prompt = f"""You are a network security analyst reviewing suspicious traffic.

Traffic summary:
{json.dumps(scenario_summary, indent=2)}

Already investigated: {already_investigated if already_investigated else "nothing yet"}

Available investigations (pick one):
- ports: which ports are being targeted
- packets: average size of network packets
- flags: system-detected anomaly indicators

Which would give you the most useful information right now?
Reply with ONLY one word from this list: {", ".join(available)}"""

    answer = call_llm(prompt).strip().lower().split()[0]
    if answer not in available:
        answer = available[0]
    return answer


def decide_continue(scenario_summary: dict, clues: dict, already_investigated: list[str]) -> bool:
    """
    Ask LLM whether to investigate more or submit.
    Returns True if it wants to investigate more.
    """
    remaining = [t for t in ["ports", "packets", "flags"] if t not in already_investigated]
    if not remaining:
        return False

    clues_text = "\n".join(
        f"  {target}: {json.dumps(result)}" for target, result in clues.items()
    )

    prompt = f"""You are a network security analyst.

Traffic summary:
{json.dumps(scenario_summary, indent=2)}

Evidence gathered so far:
{clues_text}

Remaining options to investigate: {remaining}

Are you confident enough to classify the threat, or do you need more evidence?
- If confident: reply exactly with: submit
- If more evidence needed: reply exactly with: investigate

Reply with ONLY one word."""

    answer = call_llm(prompt).strip().lower().split()[0]
    return answer == "investigate"


def classify_threat(scenario_summary: dict, clues: dict) -> tuple[str, str]:
    """
    Ask LLM to classify the threat and pick a response based on all evidence.
    Returns (threat_type, response_action).
    """
    evidence_text = "\n".join(
        f"  {target}: {json.dumps(result)}" for target, result in clues.items()
    )

    prompt = f"""You are a network security analyst. Classify this network event based on all evidence.

=== TRAFFIC SUMMARY ===
{json.dumps(scenario_summary, indent=2)}

=== INVESTIGATION RESULTS ===
{evidence_text}

=== CLASSIFICATION GUIDE ===
Use these signals to decide:

ddos:
  - Very high requests_per_second (typically 3000+)
  - Very few unique_ips (1-8) — concentrated attack from few sources
  - Web ports (80, 443, 8080)
  - Small packets
  - Short duration
  → response: block_ip

port_scan:
  - Single unique_ip (=1)
  - Many ports targeted, often sequential sweep
  - Very small packets (under 65 bytes)
  - Low rps
  → response: alert_only

brute_force:
  - Single unique_ip (=1)
  - Single auth port only (22=SSH, 3389=RDP, 21=FTP, 23=Telnet)
  - Medium-large packets
  - Long duration (sustained attack, often 2+ minutes)
  → response: block_ip

normal:
  - Many unique_ips (typically 50 or more)
  - Large packets (300+ bytes, real user sessions)
  - Very long duration (hours)
  - Reasonable rps (under 500)
  → response: ignore

=== DECISION ===
Pick EXACTLY ONE threat_type and its correct response_action.
Respond with ONLY this JSON and nothing else:
{{"threat_type": "ddos", "response_action": "block_ip"}}"""

    # Try up to 3 times to get a valid answer
    for attempt in range(3):
        raw = call_llm(prompt)
        parsed = parse_json_answer(raw)
        if parsed:
            threat = parsed.get("threat_type", "").lower().strip()
            action = parsed.get("response_action", "").lower().strip()
            if threat in VALID_THREATS and action in VALID_RESPONSES:
                return threat, action
        print(f"  (Parse attempt {attempt + 1} failed, retrying...)")

    # Final fallback: pick most likely from summary alone
    print("  ⚠️  All parse attempts failed. Using conservative default.")
    return "normal", "ignore"


def run_episode() -> float:
    """Run one full episode. Returns the reward."""
    with httpx.Client(base_url=ENV_URL, timeout=30) as client:

        # --- Reset ---
        client.post("/reset")

        # --- Get surface summary ---
        r = client.post("/step", json={"action": {"tool_name": "get_scenario", "arguments": {}}})
        data = r.json()["observation"]["result"]["data"]
        print(f"\nScenario: {data.get('scenario_id')}")
        print(f"Summary: rps={data.get('requests_per_second')}, "
              f"unique_ips={data.get('unique_ips')}, "
              f"duration={data.get('duration_seconds')}s")

        investigated = []   # targets already investigated this episode
        clues = {}          # target -> result dict

        # --- Investigation loop (max 2 rounds) ---
        for round_num in range(2):
            target = get_investigation_target(data, investigated)
            if target is None:
                break

            print(f"\nAgent investigates: {target}")
            r = client.post("/step", json={
                "action": {"tool_name": "investigate", "arguments": {"target": target}}
            })
            result = r.json()["observation"]["result"]["data"]

            # Environment enforces limits — check for errors
            if "error" in result:
                print(f"  Environment blocked: {result['error']}")
                break

            investigated.append(target)
            clues[target] = result
            print(f"Clue {round_num + 1}: {json.dumps(result, indent=2)}")

            # After first clue, ask if we need more
            if round_num == 0:
                want_more = decide_continue(data, clues, investigated)
                if not want_more:
                    print("\nAgent confident after 1 clue, submitting directly.")
                    break

        # --- Final classification ---
        threat, action = classify_threat(data, clues)
        print(f"\nAgent final answer: {{\"threat_type\": \"{threat}\", \"response_action\": \"{action}\"}}")

        # --- Submit ---
        r = client.post("/step", json={
            "action": {
                "tool_name": "submit_analysis",
                "arguments": {"threat_type": threat, "response_action": action},
            }
        })
        result = r.json()["observation"]["result"]["data"]

        if "error" in result:
            print(f"Submit error: {result['error']}")
            return 0.0

        print(f"\nReward: {result['reward']}")
        print(f"Correct threat: {result['threat_correct']}, Correct response: {result['response_correct']}")
        print(f"Investigations used: {result['investigations_used']}")
        return result["reward"]


def main():
    print("=" * 60)
    print("Network Intrusion Detection - Multi-Step Inference")
    print("=" * 60)

    rewards = []
    num_episodes = 10  # More episodes = better average estimate

    for i in range(num_episodes):
        print(f"\n--- Episode {i + 1}/{num_episodes} ---")
        try:
            reward = run_episode()
            rewards.append(reward)
        except Exception as e:
            print(f"Episode failed: {e}")
            rewards.append(0.0)

    avg_reward = sum(rewards) / len(rewards)
    correct = sum(1 for r in rewards if r >= 1.0)

    print("\n" + "=" * 60)
    print(f"Results over {num_episodes} episodes:")
    print(f"Rewards: {[round(r, 2) for r in rewards]}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Perfect episodes (reward=1.0): {correct}/{num_episodes}")
    print("=" * 60)


if __name__ == "__main__":
    main()