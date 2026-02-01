import pickle
from pathlib import Path
from openpilot.common.swaglog import cloudlog


def load_compiled_model(model_name: str = "student"):
  pkl_path = Path(__file__).parent / "distilled_models" / f"{model_name}_tinygrad.pkl"

  if not pkl_path.exists():
    cloudlog.error(f"Compiled model not found at {pkl_path}")
    return None

  try:
    with open(pkl_path, "rb") as f:
      model_run = pickle.load(f)
    return model_run
  except Exception as e:
    cloudlog.error(f"Failed to load compiled Tinygrad model: {e}")
    return None
