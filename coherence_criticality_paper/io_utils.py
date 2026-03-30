import json
import numpy as np

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)