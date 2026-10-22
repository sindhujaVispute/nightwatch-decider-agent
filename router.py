import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# --- 1. Setup Decider ---
print("Loading ONNX session...")
tokenizer = AutoTokenizer.from_pretrained("StrandsAgents/strands-decider-2B-hobson-v19")
session = ort.InferenceSession("local_model/model_quantized.onnx", providers=["CPUExecutionProvider"])

options = ["page", "queue_for_digest", "suppress"]
TEMPERATURE = 0.734

def ask_decider(state_text: str):
    instructions = "Should this alert page a human right now?"
    prompt_text = f"State: {state_text}\nQuestion: {instructions}\nOptions:\n" + "\n".join(options)
    
    encoded = tokenizer(prompt_text, return_tensors="np", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    answer_pos = np.array([input_ids.shape[1] - 1], dtype=np.int64)
    option_positions = []
    for opt in options:
        opt_tokens = tokenizer.encode(opt, add_special_tokens=False)
        for i in range(len(input_ids[0]) - len(opt_tokens) + 1):
            if np.array_equal(input_ids[0][i:i+len(opt_tokens)], opt_tokens):
                option_positions.append(i + len(opt_tokens) - 1)
                break
        else:
            option_positions.append(input_ids.shape[1] - 1)
            
    option_pos = np.array([option_positions], dtype=np.int64)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "answer_pos": answer_pos, "option_pos": option_pos}
    
    logits = session.run(None, inputs)[0]
    scaled_logits = logits / TEMPERATURE
    e_x = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probabilities = e_x / e_x.sum(axis=-1, keepdims=True)
    probs = probabilities[0]
    
    best_idx = np.argmax(probs)
    # FIX: cast to standard python float so JSON can save it later
    return options[best_idx], float(probs[best_idx])

# --- 2. Setup Bedrock Fallback ---
def ask_bedrock(alert: dict):
    """
    If you have AWS set up, replace this with boto3.
    For now, this mock represents the 'Frontier LLM' correctly analyzing the raw numbers.
    """
    # Mock logic: If error rate is high and climbing, it's a real incident.
    if alert["error_rate"] > 0.05 and alert["error_rate_slope"] > 0:
        return "page"
    return "suppress"

# --- 3. Build State (Merged) ---
def build_state(alert: dict) -> str:
    deploy = alert.get("recent_deploy_mins_ago")
    deploy_text = f"a deploy happened {deploy} minutes ago" if deploy else "no recent deploys"
    incident_text = "an incident is already open for this" if alert["open_incident"] else "no open incident"
    return (f"Service: {alert['service']}. Severity: {alert['severity']}. "
            f"Error rate is {alert['error_rate']*100:.1f}% and the trend slope is {alert['error_rate_slope']}. "
            f"Context: {deploy_text}, {incident_text}. "
            f"This alert has fired {alert['same_alert_count_24h']} times in the last 24 hours.")

# --- 4. The Band Router ---
CONFIDENCE_THRESHOLD = 0.85

alerts = [json.loads(line) for line in open("data/alerts.jsonl")]
results = []
llm_calls = 0

print(f"Routing {len(alerts)} alerts...\n")

for alert in alerts:
    state_text = build_state(alert)
    
    # 1. Ask the tiny model
    decision, confidence = ask_decider(state_text)
    source = "decider"
    
    # 2. Band Router: If confidence is low, escalate to Bedrock!
    if confidence < CONFIDENCE_THRESHOLD:
        source = "bedrock"
        decision = ask_bedrock(alert)
        llm_calls += 1
        
    truth = alert["ground_truth"]
    is_correct = (truth == "real_incident" and decision == "page") or (truth == "noise" and decision in ["suppress", "queue_for_digest"])
    
    results.append({
        "id": alert["id"],
        "truth": truth,
        "decision": decision,
        "confidence": round(confidence, 3),
        "source": source,
        "correct": is_correct
    })

# --- 5. Results ---
correct_count = sum(1 for r in results if r["correct"])
print(f"--- Results ---")
print(f"Accuracy: {correct_count}/{len(alerts)} ({(correct_count/len(alerts))*100:.1f}%)")
print(f"Bedrock LLM calls used: {llm_calls} out of {len(alerts)} ({(llm_calls/len(alerts))*100:.1f}%)")
print(f"Tokens/Cost saved by Decider: {100 - (llm_calls/len(alerts))*100:.1f}%\n")

print(f"Sample Decisions:")
print(f"{'ID':<15} | {'Truth':<15} | {'Decision':<18} | {'Source':<10} | {'Correct'}")
print("-" * 75)
for r in results[:10]:
    print(f"{r['id']:<15} | {r['truth']:<15} | {r['decision']:<18} | {r['source']:<10} | {r['correct']}")

# Save for your hackathon evaluation notebook
with open("data/routed_results.json", "w") as f:
    json.dump(results, f, indent=2)