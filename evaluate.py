import json
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# --- 1. Builder Logic (Merged inside to avoid import errors) ---
def build_state(alert: dict) -> str:
    deploy = alert.get("recent_deploy_mins_ago")
    deploy_text = f"a deploy happened {deploy} minutes ago" if deploy else "no recent deploys"
    incident_text = "an incident is already open for this" if alert["open_incident"] else "no open incident"
    state = (
        f"Service: {alert['service']}. "
        f"Severity: {alert['severity']}. "
        f"Error rate is {alert['error_rate']*100:.1f}% and the trend slope is {alert['error_rate_slope']}. "
        f"Context: {deploy_text}, {incident_text}. "
        f"This alert has fired {alert['same_alert_count_24h']} times in the last 24 hours."
    )
    return state

# --- 2. Load Model & Tokenizer ---
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
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_pos": answer_pos,
        "option_pos": option_pos,
    }
    
    logits = session.run(None, inputs)[0]
    scaled_logits = logits / TEMPERATURE
    
    e_x = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    probabilities = e_x / e_x.sum(axis=-1, keepdims=True)
    probs = probabilities[0]
    
    best_idx = np.argmax(probs)
    return options[best_idx], probs[best_idx]

# --- 3. Run on all 100 alerts ---
alerts = [json.loads(line) for line in open("data/alerts.jsonl")]
results = []

print(f"Running {len(alerts)} alerts through Decider 2B...\n")

for i, alert in enumerate(alerts):
    state_text = build_state(alert)
    decision, confidence = ask_decider(state_text)
    
    truth = alert["ground_truth"]
    is_correct = False
    
    if truth == "real_incident" and decision == "page":
        is_correct = True
    elif truth == "noise" and decision in ["suppress", "queue_for_digest"]:
        is_correct = True
        
    results.append({
        "id": alert["id"],
        "truth": truth,
        "decision": decision,
        "confidence": round(confidence, 3),
        "correct": is_correct
    })
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(alerts)}...")

# --- 4. Print Results ---
correct_count = sum(1 for r in results if r["correct"])
print(f"\n--- Results ---")
print(f"Accuracy: {correct_count}/{len(alerts)} ({(correct_count/len(alerts))*100:.1f}%)")

print("\nSample Decisions (First 10):")
print(f"{'ID':<15} | {'Truth':<15} | {'Decision':<18} | {'Conf':<6} | {'Correct'}")
print("-" * 70)
for r in results[:10]:
    print(f"{r['id']:<15} | {r['truth']:<15} | {r['decision']:<18} | {r['confidence']:<6} | {r['correct']}")

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)