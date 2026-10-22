# Nightwatch: Pager Fatigue Agent

An AI agent built for the Strands Decider Hackathon that uses **Strands Decider 2B** (`#policy` pattern) to filter out server alert noise before it wakes up an on-call engineer.

## The Problem
On-call engineers suffer from alert fatigue. Most alerts are noise. However, missing a real outage (like a P1 incident) is disastrous.

## The Solution
1. **Decider 2B (Fast Gate):** Reads the alert state and instantly decides `page`, `queue_for_digest`, or `suppress`. 
2. **Confidence Band Routing:** If Decider 2B is highly confident (<0.85), it acts immediately. If it is unsure, the decision is escalated to **Amazon Bedrock** (the LLM) for deeper reasoning.

## Reusable Pattern: Decider-Gate
This repository includes a reusable `InterventionHandler` in the [`patterns/decider-gate`](./patterns/decider-gate) folder. You can drop this file into your own Strands agents to prevent them from taking low-confidence actions.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Generate the synthetic data: `python generate_alerts.py`
3. Run the agent: `python router.py`