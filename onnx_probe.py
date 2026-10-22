import json
import os
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# --- 1. Download the quantized model AND its external data file ---
model_id = "onnx-community/strands-decider-2B-hobson-v19-ONNX"

print("Downloading model graph (.onnx)...")
model_file = hf_hub_download(
    repo_id=model_id,
    filename="onnx/model_quantized.onnx"
)

print("Downloading model weights (.onnx_data)...")
# This is the missing piece. We download the companion file.
data_file = hf_hub_download(
    repo_id=model_id,
    filename="onnx/model_quantized.onnx_data"
)

# --- 2. Load the ONNX Runtime session ---
# ONNX Runtime looks for the .onnx_data file in the same directory as the .onnx file.
# hf_hub_download caches files in separate snapshot folders, so we need to copy them together.
import shutil

# Create a local directory for the model files
os.makedirs("local_model", exist_ok=True)
local_onnx_path = os.path.join("local_model", "model_quantized.onnx")
local_data_path = os.path.join("local_model", "model_quantized.onnx_data")

# Copy the downloaded files to the local directory
shutil.copy(model_file, local_onnx_path)
shutil.copy(data_file, local_data_path)

print("Loading ONNX session...")
# Now load the model from the local path
session = ort.InferenceSession(local_onnx_path, providers=["CPUExecutionProvider"])

# --- 3. Define your state and question ---
state = "checkout-api error rate 4.1% and climbing, P1, deploy to checkout-api 18 minutes ago, no open incident, this alert has not fired in the last 24h"

question_type = "choice"
instructions = "Should this alert page a human right now?"
options = ["page", "queue_for_digest", "suppress"]

# Format the prompt
prompt_text = f"State: {state}\nQuestion: {instructions}\nOptions:\n" + "\n".join(options)

# --- 4. Tokenize and find positions ---
tokenizer = AutoTokenizer.from_pretrained("StrandsAgents/strands-decider-2B-hobson-v19")
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

# --- 5. Run the model ---
inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "answer_pos": answer_pos,
    "option_pos": option_pos,
}

print("Running inference...")
logits = session.run(None, inputs)[0]

# --- 6. Apply calibration temperature and softmax ---
TEMPERATURE_BY_KIND = {
    "noul": 0.911,
    "choice": 0.734,
    "score": 1.328,
}

temp = TEMPERATURE_BY_KIND[question_type]
scaled_logits = logits / temp

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

probabilities = softmax(scaled_logits)[0]

# --- 7. Print the result ---
best_idx = np.argmax(probabilities)
print(f"\nDecision: {options[best_idx]}")
print(f"Confidence: {probabilities[best_idx]:.4f}")
print("\nFull distribution:")
for opt, prob in zip(options, probabilities):
    print(f"  {opt}: {prob:.4f}")