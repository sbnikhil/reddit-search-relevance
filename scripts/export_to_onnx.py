import torch
import yaml
import os
from models.arch.relevance_ranker import RedditRelevanceRanker

with open("config/settings.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")

model = RedditRelevanceRanker(
    cfg["training"]["model_name"],
    cfg["training"]["extra_feature_dim"],
    dropout=cfg["model_params"].get("dropout_rate", 0.1),
).to(device)

model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
model_path = os.path.join(cfg["artifacts"]["model_path"], model_file)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Dummy inputs for ONNX export
batch_size = 1
seq_length = cfg["data_processing"]["max_sequence_length"]
feature_dim = cfg["training"]["extra_feature_dim"]

dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length), dtype=torch.long)
dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
dummy_features = torch.randn((batch_size, feature_dim), dtype=torch.float32)

onnx_path = os.path.join(cfg["artifacts"]["model_path"], f"reddit_ranker_v{cfg['artifacts']['version']}.onnx")

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask, dummy_features),
    onnx_path,
    input_names=["input_ids", "attention_mask", "features"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "features": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
    do_constant_folding=True,
)

print(f"Model exported to {onnx_path}")

# Verify the export
try:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)

    onnx_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
        "features": dummy_features.numpy(),
    }

    onnx_output = session.run(None, onnx_inputs)[0]

    with torch.no_grad():
        torch_output = model(dummy_input_ids, dummy_attention_mask, dummy_features).numpy()

    diff = np.abs(onnx_output - torch_output).max()

    if diff < 1e-4:
        print(f"Verification passed (max diff: {diff:.6f})")
    else:
        print(f"Warning: ONNX output differs from PyTorch (max diff: {diff:.6f})")

except ImportError:
    print("onnxruntime not installed, skipping verification")
    print("Install with: pip install onnxruntime")
