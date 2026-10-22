from strands.interventions import InterventionHandler, Proceed, Deny, Confirm

class DeciderGate(InterventionHandler):
    """
    A reusable Strands InterventionHandler that asks Decider 2B
    before allowing a tool to page a human.
    """
    def __init__(self, decider_client, build_state, threshold=0.85):
        self.decider = decider_client
        self.build = build_state
        self.threshold = threshold

    def before_tool_call(self, call, context):
        # We only intervene on the paging tool
        if call.name != "page_human":
            return Proceed()

        # 1. Build the state from the agent's current context
        state_text = self.build(context)

        # 2. Ask Decider 2B
        decision, confidence = self.decider.ask(state_text)

        # 3. The Confidence Router
        if decision == "page" and confidence >= self.threshold:
            return Proceed()  # Model is confident, let the tool run
            
        elif decision == "suppress" and confidence >= self.threshold:
            return Deny(reason="Decider 2B classified this as noise.")
            
        else:
            # Low confidence! We don't know what to do. Ask the human.
            return Confirm(
                question=f"Decider 2B is unsure (confidence: {confidence:.2f}). Page the on-call engineer?"
            )