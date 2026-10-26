"""Roda em sequencia a avaliacao do pre-treinado (03) e a do modelo ajustado (05)."""
import subprocess
import sys

for s in ("03_avaliar_coco.py", "05_avaliar_ft.py"):
    print(f"===== {s}", flush=True)
    subprocess.run([sys.executable, f"/w/deteccao/{s}"] + sys.argv[1:] if s.startswith("05") else [sys.executable, f"/w/deteccao/{s}"], check=False)
