# NomaApp v1.2: Model Delivery Contract

This is the agreement between the ML team, which builds the models, and the inference service (branch `fastapi-v2`), which serves them. If a model meets every rule here, it can be deployed by uploading it to R2 and updating `manifest.json`. No service code changes are needed.

Items marked **(proposed)** need a small service change before they take effect. Everything else works today.

---

## 1. How the service uses your models

```
photo ─► quality check ─► ROUTER (which crop?) ─► EXPERT for that crop (which disease?) ─► OOD gate ─► result
```

- **Router:** one model that identifies the crop. It stays loaded permanently.
- **Experts:** one model per crop that identifies the disease. They are loaded on demand, and at most 3 are kept in RAM.
- **Runtime:** `onnxruntime` on CPU only (1 vCPU, 4 GB RAM). TensorFlow and PyTorch are **not installed** on the server.
- **Who calls it:** the Express API calls `POST /diagnose` with a single image. **The client does not send a crop type**; the router determines it.

## 2. File format

| Rule | Requirement |
| --- | --- |
| Format | `.onnx`. `.keras`, `.h5` and `.pt` are not accepted. |
| Single file | Weights must be embedded in the `.onnx` file. No `.onnx.data` sidecar. |
| Opset | 13–21. Keras's default exporter output (opset 15, IR 8) is fine. |
| Inputs | Exactly 1 input: float32 image tensor, either `[N, 3, 224, 224]` (NCHW) or `[N, 224, 224, 3]` (NHWC). The layout is detected automatically. A dynamic batch dimension is fine. |
| Outputs | Exactly 1 output, shaped `[N, num_classes]`. |
| Output values | **Raw logits, with no final Softmax.** The OOD gate uses the energy score, which needs logits. A model that outputs probabilities still works, but loses OOD protection and falls back to a weaker confidence check. |
| Size / speed | Router: about 20 MB and under 200 ms. Expert: about 100 MB or less and under 1 s on 1 vCPU. MobileNetV3 or EfficientNet-B0 class models fit comfortably. |

### Keras → ONNX (logits)

```python
import keras
model = keras.models.load_model("maize.keras")
model.layers[-1].activation = keras.activations.linear   # drop softmax -> logits
# Option A (Keras >= 3.8):
model.export("maize_v1.onnx", format="onnx")
# Option B: export SavedModel, then
#   model.export("maize_savedmodel")
#   python -m tf2onnx.convert --saved-model maize_savedmodel --opset 17 --output maize_v1.onnx
```

If softmax is a separate `Activation` layer rather than part of the last `Dense`, build `keras.Model(model.input, model.layers[-2].output)` instead.

### PyTorch → ONNX

```python
torch.onnx.export(model.eval(), torch.randn(1, 3, 224, 224), "cassava_v1.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}, opset_version=17)
# If the file is saved with a .onnx.data sidecar, re-save it as one file:
import onnx; m = onnx.load("cassava_v1.onnx"); onnx.save(m, "cassava_v1.onnx", save_as_external_data=False)
```

### Parity check (required before delivery)

Run the same 50 images through the original framework model and the ONNX file, then compare. **The maximum absolute difference in logits must be below 1e-3**, and the top-1 class must be identical. Include the result in the model card.

## 3. Preprocessing

The service applies **one** preprocessing pipeline to every model:

```
decode → apply EXIF rotation → RGB → resize to 224×224 (bilinear)
→ /255 → (x − [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]   # ImageNet mean/std
→ float32, NCHW (transposed to NHWC automatically if the model expects it)
```

**Models must be trained or fine-tuned with this exact normalisation.** The old `main` branch used `/255` only. A model trained that way gives wrong predictions here, and the failure is silent: no error, just bad answers.

If retraining isn't possible, choose one of these options and tell us which:

- **A.** Put your own normalisation inside the model graph (for example as a Keras `Rescaling`/`Normalization` layer) and declare `"preprocessing": "raw_0_255"`. *(proposed)*
- **B.** Declare `"preprocessing": "unit_0_1"` for `/255`-only models. *(proposed)*

## 4. Labels

- Labels live in `manifest.json` (see section 6). This is the **single source of truth**, so there is no separate `labels.json` at runtime.
- Each expert's label list contains disease names only, because the crop comes from the router: `["northern_blight", "gray_leaf_spot", "common_rust", "healthy"]`.
- Labels must be in **exactly the model's output index order**. The service refuses to load a model whose output size differs from its label count.
- Use lowercase `snake_case`, keep labels unique within each crop, and spell the healthy class exactly `healthy`.
- Crop IDs must be lowercase and identical between the router's `classes` and the `experts` keys. Use `maize`, not `corn`.
- Display names, translations (Hausa, Yoruba, Igbo, Pidgin) and treatment advice are **not** part of the model. The app/Express side maps label IDs to those. Please provide a suggested display name per label in the model card.

## 5. The router

- `classes` must include **every crop that has an expert**. A crop with an expert but no router class can never be reached.
- **Strongly recommended:** add catch-all classes trained on negative examples:
  - `other`: plants we don't support → the service returns `unsupported_crop`.
  - `no_plant`: soil, hands, sky, documents, people → the service returns `no_plant_detected`.

  These are the router's main protection against out-of-scope photos.
- A router class with no expert (for example `pepper` today) returns `unsupported_crop`. That's allowed, but it should be deliberate.

## 6. `manifest.json`

```json
{
  "version": "1.3.0",
  "router": {
    "file": "router/crop_identifier_v1.onnx",
    "version": "1.0.0",
    "input_size": [224, 224],
    "classes": ["cassava", "maize", "rice", "tomato", "other", "no_plant"],
    "sha256": "…",                         // (proposed)
    "preprocessing": "imagenet"            // (proposed) imagenet | unit_0_1 | raw_0_255
  },
  "experts": {
    "maize": {
      "file": "experts/maize_v1.onnx",
      "version": "1.0.0",
      "diseases": ["northern_blight", "gray_leaf_spot", "common_rust", "healthy"],
      "sha256": "…",                       // (proposed)
      "preprocessing": "imagenet",         // (proposed)
      "energy_threshold": -4.2             // optional per-model OOD threshold, see section 8
    }
  }
}
```

## 7. Delivery and versioning

- **Model weights never go in git.** Files go to the R2 bucket under `router/` and `experts/`.
- Filenames follow `{crop}_v{N}.onnx`, using the crop ID (`maize_v1.onnx`, not `corn_model_v1.onnx`), and are **immutable**: never overwrite an existing key. A retrained model gets a new filename (`maize_v2.onnx`) plus a manifest update. Servers pick up the new file automatically on restart.
- Each delivery includes the `.onnx` file, its SHA-256 hash, and a model card. The ML team uploads the file. Whoever owns releases bumps `manifest.json`, because that is the release step.
- Before delivering, run `python scripts/check_model.py <model>.onnx --card <card>.json` from the service repo. It loads the model with the service's own code and checks the file format, input/output shapes, label count, logits and latency. Delivery requires `RESULT: OK`.
- Rollback means pointing the manifest back at the previous filename.

## 8. Model card (one per model)

Deliver this as `{crop}_v{N}.card.json` next to the model in R2, or in a shared folder:

```json
{
  "model": "maize_v1.onnx", "sha256": "…", "date": "2026-10-20", "framework": "keras 3.x / tf 2.21",
  "input": {"shape": [1, 224, 224, 3], "layout": "NHWC", "preprocessing": "imagenet"},
  "output": {"type": "logits", "labels": ["northern_blight", "gray_leaf_spot", "common_rust", "healthy"]},
  "display_names": {"northern_blight": "Northern Leaf Blight", "...": "..."},
  "training_data": "PlantVillage + 1,200 field photos from Kaduna (2026-07)",
  "metrics": {"val_accuracy": 0.94, "per_class_f1": {"...": 0.0}, "confusion_matrix": [[...]]},
  "parity": {"images": 50, "max_abs_logit_diff": 0.0002, "top1_match": "50/50"},
  "ood": {"energy_threshold": -4.2, "tpr_in_dist": 0.95, "fpr_ood": 0.12}
}
```

### Calibrating OOD thresholds (the ML team owns this)

1. Compute the energy score `E = -logsumexp(logits)` on held-out **in-distribution** validation images.
2. Compute it on an **OOD set**: other crops, non-plant photos, and blurry or badly framed field photos.
3. Choose the threshold that keeps about **95%** of in-distribution images. Report the OOD rejection rate at that threshold.
4. Do the same for the router's softmax confidence. The current default is 0.70.

Energy scales differ between models, so put each expert's calibrated value in its manifest entry as `energy_threshold`. If an expert has no value, the service uses the global `ENERGY_OOD_THRESHOLD` env var.

## 9. Golden test set (per model)

Provide 20 images, with their file names and expected top-1 label and logits, from your own pipeline. We add these to the service's CI to prove that our decoding, resizing and normalisation reproduce your numbers. This is the main safeguard against the two sides drifting apart.

## 10. Out of scope for v1.2

- **Severity.** The old service derived severity from confidence, which is wrong: a confident diagnosis of a mild disease came out as "high". Severity is dropped until there is severity-labelled data and a dedicated model or head.
- Treatment recommendations: owned by the app/content side.
