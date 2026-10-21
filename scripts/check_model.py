"""Check an ONNX model against docs/MODEL_CONTRACT.md using the service's own loading code.

Usage:
    python scripts/check_model.py model.onnx --labels healthy,rust,blight
    python scripts/check_model.py model.onnx --card model_card.json [--image leaf.jpg] [--router]

The labels come from --labels, or else from the card's "diseases" / "classes" field.
`pip install onnx` additionally enables the opset, IR-version and external-data checks.
Exit code is 0 when every required check passes.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exceptions import ModelLoadError  # noqa: E402
from ood_detector import energy_score, softmax, to_scores  # noqa: E402
from onnx_model import OnnxClassifier  # noqa: E402
from preprocessing import preprocess_image  # noqa: E402

MIN_OPSET, MAX_OPSET = 13, 21
LATENCY_BUDGET_MS = {"router": 200.0, "expert": 1000.0}
TIMED_RUNS = 20

failures: list[str] = []


def report(ok: bool, name: str, detail: str = "", required: bool = True) -> None:
    tag = "PASS" if ok else ("FAIL" if required else "WARN")
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok and required:
        failures.append(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_labels(card: dict) -> list[str] | None:
    for key in ("diseases", "classes", "labels"):
        if isinstance(card.get(key), list):
            return card[key]
        if isinstance(card.get("output"), dict) and isinstance(card["output"].get(key), list):
            return card["output"][key]
    return None


def check_graph(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[SKIP] opset / IR / external-data checks (pip install onnx to enable)")
        return
    model = onnx.load(str(path), load_external_data=False)
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    report(
        opset is not None and MIN_OPSET <= opset <= MAX_OPSET,
        "opset",
        f"{opset} (accepted {MIN_OPSET}-{MAX_OPSET}), IR version {model.ir_version}",
    )
    external = sorted(
        {e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"}
    )
    report(not external, "single file (no external data)", ", ".join(external) or "embedded")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--labels", help="comma-separated labels in output order")
    ap.add_argument("--card", type=Path, help="model card JSON")
    ap.add_argument("--image", type=Path, help="real leaf photo to run (default: random noise)")
    ap.add_argument("--router", action="store_true", help="apply the router latency budget")
    ap.add_argument("--input-size", type=int, default=224)
    args = ap.parse_args()

    card = json.loads(args.card.read_text(encoding="utf-8")) if args.card else {}
    labels = args.labels.split(",") if args.labels else find_labels(card)
    if not labels:
        ap.error("labels needed: pass --labels or a --card containing 'diseases'/'classes'")
    role = "router" if args.router else "expert"

    print(f"onnxruntime {ort.__version__} | model {args.model.name} | {role} | {len(labels)} labels")
    size_mb = args.model.stat().st_size / 1024**2
    digest = sha256(args.model)
    print(f"[INFO] size {size_mb:.1f} MB, sha256 {digest}")
    if card.get("sha256"):
        report(card["sha256"].lower() == digest, "sha256 matches card")

    check_graph(args.model)

    rss_before = psutil.Process().memory_info().rss
    try:
        clf = OnnxClassifier(args.model, len(labels), name=args.model.stem)
    except ModelLoadError as exc:
        report(False, "loads in service (OnnxClassifier)", str(exc))
        return 1
    rss_mb = (psutil.Process().memory_info().rss - rss_before) / 1024**2
    report(True, "loads in service (OnnxClassifier)", f"+{rss_mb:.0f} MB RSS")

    inputs, outputs = clf.session.get_inputs(), clf.session.get_outputs()
    report(len(inputs) == 1 and len(outputs) == 1, "exactly 1 input and 1 output")
    report(
        len(inputs[0].shape) == 4 and 3 in (inputs[0].shape[1], inputs[0].shape[3]),
        "input is a 4-D RGB image tensor",
        f"{inputs[0].shape} -> {'NHWC' if clf.channels_last else 'NCHW'}",
    )
    report(inputs[0].type == "tensor(float)", "input dtype float32", inputs[0].type)

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    batch = preprocess_image(img, (args.input_size, args.input_size))

    raw = clf.predict(batch)
    report(raw.shape == (len(labels),), "output size equals label count", f"{raw.shape[0]} vs {len(labels)}")
    report(bool(np.all(np.isfinite(raw))), "output values finite")
    scores = to_scores(raw, "auto")
    report(scores.logits is not None, "outputs raw logits (no softmax)",
           "looks like probabilities - energy OOD disabled" if scores.logits is None else "")

    clf.predict(batch)  # warm-up
    t = time.perf_counter()
    for _ in range(TIMED_RUNS):
        clf.predict(batch)
    ms = (time.perf_counter() - t) * 1000 / TIMED_RUNS
    budget = LATENCY_BUDGET_MS[role]
    report(ms < budget, "latency (1 thread)", f"{ms:.1f} ms, budget {budget:.0f} ms. "
           "This machine may be faster than the 1-vCPU server", required=False)

    probs = softmax(raw)
    top = int(np.argmax(probs))
    print(f"[INFO] top-1 {labels[top]!r} p={probs[top]:.3f}, energy {energy_score(raw):.3f}")
    print(f"[INFO] logits {np.round(raw, 4).tolist()}")

    print("\nRESULT:", "OK" if not failures else f"FAILED ({', '.join(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
