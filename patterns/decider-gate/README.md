# Decider-Gate Pattern

This is a reusable Strands `InterventionHandler` that prevents an agent from paging a human unless a fast, local `strands-decider-2b` model is highly confident that the alert is real.

## The Decision
- **Question:** Should this alert page a human right now?
- **Options:** `page`, `queue_for_digest`, `suppress`
- **Context passed to Decider:** Service name, severity, error rate slope, recent deploy time, duplicate count.

## How it works
1. The agent attempts to call the `page_human` tool.
2. The `before_tool_call` hook intercepts it.
3. It builds a text representation of the alert (the "state").
4. It asks Decider 2B.
5. **The Band Router:**
   - Confidence ≥ 0.85 + Decision `page` → `Proceed()` (Agent pages human).
   - Confidence ≥ 0.85 + Decision `suppress` → `Deny()` (Agent stops).
   - Confidence < 0.85 → `Confirm()` (Agent stops and asks the human for permission).

## How to use it in your agent
```python
from handler import DeciderGate

gate = DeciderGate(
    decider_client=my_onnx_decider, 
    build_state=my_state_builder, 
    threshold=0.85
)

agent = Agent(
    interventions=[gate],
    tools=[page_human, get_weather]
)
