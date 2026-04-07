"""
Inference script for Network Intrusion Detection Environment.
Multi-step: agent investigates before classifying.
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


def call_llm(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1
    )
    return response.choices[0].message.content


def run_episode():
    with httpx.Client(base_url=ENV_URL, timeout=30) as client:

        # Step 1: Reset
        client.post("/reset")

        # Step 2: Get surface summary
        r = client.post("/step", json={"action": {"tool_name": "get_scenario", "arguments": {}}})
        scenario_summary = r.json()["observation"]["result"]["data"]
        print(f"\nScenario: {scenario_summary.get('scenario_id')}")
        print(f"Summary: rps={scenario_summary.get('requests_per_second')}, unique_ips={scenario_summary.get('unique_ips')}, duration={scenario_summary.get('duration_seconds')}s")

        # Step 3: Agent picks first investigation
        investigation_prompt = f"""You are a network security analyst.
            Network traffic summary:
            {json.dumps(scenario_summary, indent=2)}

            Available investigations:
            - ports: which ports are being targeted (reveals scan vs auth vs web)
            - packets: average packet size (small=attack, large=normal session)
            - flags: system-detected anomaly indicators

            Which single word would you investigate first? Reply with ONLY one word: ports, packets, or flags."""

        first_target = call_llm(investigation_prompt).strip().lower()
        if first_target not in ["ports", "packets", "flags"]:
            first_target = "flags"

        print(f"\nAgent investigates: {first_target}")
        r = client.post("/step", json={"action": {"tool_name": "investigate", "arguments": {"target": first_target}}})
        clue_1 = r.json()["observation"]["result"]["data"]
        print(f"Clue 1: {json.dumps(clue_1, indent=2)}")

        # Step 4: Decide to investigate more or submit
        decision_prompt = f"""You are a network security analyst.

            Summary: {json.dumps(scenario_summary, indent=2)}
            First investigation ({first_target}): {json.dumps(clue_1, indent=2)}

            Based on this, are you confident enough to classify, or do you need one more clue?

            IMPORTANT: Do NOT investigate the same thing twice. You already investigated '{first_target}'.
            - To investigate more: reply with exactly one of these: investigate ports / investigate packets / investigate flags
            - To classify now: reply with exactly: submit

            Reply with ONLY that one line."""

        decision = call_llm(decision_prompt).strip().lower()

        if decision.startswith("investigate"):
            second_target = decision.split()[-1]
            if second_target not in ["ports", "packets", "flags"]:
                second_target = "packets"
            print(f"\nAgent investigates more: {second_target}")
            r = client.post("/step", json={"action": {"tool_name": "investigate", "arguments": {"target": second_target}}})
            clue_2 = r.json()["observation"]["result"]["data"]
            print(f"Clue 2: {json.dumps(clue_2, indent=2)}")
            all_evidence = f"Summary: {json.dumps(scenario_summary)}\nClue 1 ({first_target}): {json.dumps(clue_1)}\nClue 2 ({second_target}): {json.dumps(clue_2)}"
        else:
            print("\nAgent confident after 1 clue, submitting directly.")
            all_evidence = f"Summary: {json.dumps(scenario_summary)}\nClue 1 ({first_target}): {json.dumps(clue_1)}"

        # Step 5: Final classification

        final_prompt = f"""You are a network security analyst. Classify the network threat based on ALL evidence.

            {all_evidence}

            THREAT TYPES and their CORRECT RESPONSES:
            - ddos (high rps, few IPs, web ports) → correct response: block_ip
            - port_scan (sequential ports, single IP, low rps) → correct response: alert_only
            - brute_force (single auth port like 22/3389/21, repeated attempts) → correct response: block_ip
            - normal (many unique_ips like 50+, very long duration, large packets >200 bytes, reasonable rps <500) → correct response: ignore

            Pick EXACTLY ONE threat_type and its matching correct response from above.

            Respond with ONLY this JSON, no explanation:
            {{"threat_type": "ddos", "response_action": "block_ip"}}"""

        raw_answer = call_llm(final_prompt).strip()
        if "```" in raw_answer:
            raw_answer = raw_answer.split("```")[1]
            if raw_answer.startswith("json"):
                raw_answer = raw_answer[4:]
        print(f"\nAgent final answer: {raw_answer}")

        try:
            parsed = json.loads(raw_answer.strip())
            threat = parsed["threat_type"]
            action = parsed["response_action"]
        except (json.JSONDecodeError, KeyError):
            threat, action = "normal", "ignore"

        # Step 6: Submit
        r = client.post("/step", json={"action": {"tool_name": "submit_analysis", "arguments": {
            "threat_type": threat,
            "response_action": action
        }}})
        result = r.json()["observation"]["result"]["data"]
        print(f"\nReward: {result['reward']}")
        print(f"Correct threat: {result['threat_correct']}, Correct response: {result['response_correct']}")
        print(f"Investigations used: {result['investigations_used']}")
        return result["reward"]


def main():
    print("[START] task=network_intrusion_detection", flush=True)

    rewards = []
    num_episodes = 5

    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---", flush=True)
        try:
            reward = run_episode()
            rewards.append(reward)
            print(f"[STEP] step={i+1} reward={reward}", flush=True)
        except Exception as e:
            print(f"Episode failed: {e}", flush=True)
            print(f"[STEP] step={i+1} reward=0.0", flush=True)
            rewards.append(0.0)

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nResults over {num_episodes} episodes:", flush=True)
    print(f"Rewards: {rewards}", flush=True)
    print(f"Average reward: {avg_reward:.3f}", flush=True)
    print(f"[END] task=network_intrusion_detection score={avg_reward:.3f} steps={num_episodes}", flush=True)