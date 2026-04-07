---
title: Network Intrusion Detection Environment
colorFrom: blue
colorTo: green
sdk: docker
---

# Network Intrusion Detection Environment

A reinforcement learning environment where an agent analyzes network traffic, investigates details, and classifies threats.

---

## Overview

The agent:
1. Gets a partial traffic summary
2. Chooses what to investigate (ports, packets, flags)
3. Submits a threat classification and response

It must balance:
- Getting enough information
- Avoiding unnecessary investigations

---

## Actions

- get_scenario()
  Returns initial traffic summary

- investigate("ports" | "packets" | "flags")
  Reveals more details

- submit_analysis(threat_type, response_action)
  Submits final answer and returns reward

---