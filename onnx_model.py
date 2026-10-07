"""Thin wrapper around an ONNX Runtime session, shared by the router and the experts."""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from exceptions import ModelLoadError

logger = logging.getLogger("nomaapp.onnx")


def create_session(path: Path) -> ort.InferenceSession:
    # One vCPU: extra ORT threads only add contention.
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class OnnxClassifier:
    """Single-input, single-output image classifier.

    Preprocessing always yields NCHW. If the exported graph expects NHWC (common for
    Keras/TF exports) the batch is transposed here, so the shared pipeline stays identical.
    """

    def __init__(self, path: Path, num_classes: int, name: str) -> None:
        self.name = name
        try:
            self.session = create_session(path)
        except Exception as exc:  # ORT raises several unrelated exception types
            raise ModelLoadError(f"could not load {name}: {exc}") from exc

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.channels_last = len(shape) == 4 and shape[1] != 3 and shape[3] == 3

        out_dim = self.session.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != num_classes:
            raise ModelLoadError(
                f"{name}: model has {out_dim} outputs but manifest lists {num_classes} classes"
            )
        logger.info(
            "onnx model loaded",
            extra={"model": name, "input_shape": [str(d) for d in shape], "nhwc": self.channels_last},
        )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(1, 3, H, W) float32 -> 1-D raw output (logits or probabilities)."""
        if self.channels_last:
            batch = np.ascontiguousarray(batch.transpose(0, 2, 3, 1))
        outputs = self.session.run(None, {self.input_name: batch})
        return np.asarray(outputs[0]).reshape(-1)
