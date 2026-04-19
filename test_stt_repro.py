"""Reproduce the main.py Whisper call in isolation to diagnose the hang."""
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Register NVIDIA CUDA DLLs for faster-whisper
if os.name == "nt":
    _venv_root = Path(sys.executable).parent.parent
    _nvidia_root = _venv_root / "Lib" / "site-packages" / "nvidia"
    if _nvidia_root.is_dir():
        _new_paths = []
        for _pkg_dir in _nvidia_root.iterdir():
            _bin = _pkg_dir / "bin"
            if _bin.is_dir():
                os.add_dll_directory(str(_bin))
                _new_paths.append(str(_bin))
        if _new_paths:
            os.environ["PATH"] = os.pathsep.join(_new_paths) + os.pathsep + os.environ.get("PATH", "")
            print(f"[dll] registered {len(_new_paths)} nvidia bin dirs", flush=True)

import numpy as np
from faster_whisper import WhisperModel


def run_test(model_size: str, use_vad: bool):
    print(f"\n===== {model_size} | VAD={use_vad} =====", flush=True)
    t0 = time.time()
    m = WhisperModel(model_size, device="cuda", compute_type="float16")
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    audio = (np.random.randn(24000) * 0.1).astype(np.float32)
    print(f"[audio] shape={audio.shape} range=[{audio.min():.3f},{audio.max():.3f}]", flush=True)

    kwargs = dict(beam_size=1, language="en", initial_prompt="Raiken, Claude, Nakama")
    if use_vad:
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 300}

    t0 = time.time()
    segments, info = m.transcribe(audio, **kwargs)
    print(f"[transcribe()] returned in {time.time()-t0:.2f}s; lang={info.language}", flush=True)

    t0 = time.time()
    texts = []
    for i, s in enumerate(segments):
        print(f"  seg{i}: {s.text!r}", flush=True)
        texts.append(s.text.strip())
    print(f"[iterate] {time.time()-t0:.2f}s; result={' '.join(texts)!r}", flush=True)

    # Free VRAM for the next test
    del m


# Start small, work up.
run_test("tiny", use_vad=True)
run_test("tiny", use_vad=False)
run_test("medium", use_vad=True)
run_test("medium", use_vad=False)

print("\n[DONE]", flush=True)
