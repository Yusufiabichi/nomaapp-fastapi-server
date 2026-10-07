# NomaApp v1.2 — Inference Service

FastAPI service that diagnoses crop diseases from a farmer's photo. It runs ONNX models on CPU only, loading them from Cloudflare R2. It is built for a single-vCPU, 4 GB Railway VPS.

```
image ─► 1. quality check ─► 2. crop router ─► 3. expert model ─► 4. OOD gate ─► diagnosis
          (heuristic)          (always warm)     (RAM→disk→R2)     (energy score)
          poor_image_quality   crop_not_identified                 uncertain_disease_pattern
                               unsupported_crop / no_plant_detected
```

Uncertain or invalid inputs are rejected early (HTTP 422) with a plain-language message for the farmer.

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | App wiring: lifespan, endpoints, exception handlers |
| `config.py` | `Settings`: every env var with its default |
| `preprocessing.py` | Image decoding plus the preprocessing pipeline shared by the router and experts |
| `quality_check.py` | Stage 1: size, resolution, brightness and blur checks |
| `r2_client.py` | boto3 R2 client, manifest loading and fallback, atomic model downloads |
| `onnx_model.py` | ONNX Runtime session wrapper (single-threaded, NCHW/NHWC aware) |
| `router_model.py` | Stage 2: `CropRouter`, loaded at startup and never evicted |
| `expert_cache.py` | Stage 3: `ExpertCache`, a 3-tier LRU with a per-crop `asyncio.Lock` |
| `ood_detector.py` | Stages 2 and 4: softmax and energy-score gates |
| `schemas.py` | Manifest and HTTP response models |
| `exceptions.py` | `QualityRejection`, `OODRejection`, `UnsupportedCropError`, `ModelLoadError` |
| `logging_setup.py` | JSON logs with a per-request `request_id` |

## Running locally

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
cp .env.example .env            # then fill in the R2 credentials
uvicorn main:app --port 8000 --workers 1
```

At startup the service:

1. Creates `model_cache/router/` and `model_cache/experts/`, then logs the cache path and free disk space.
2. Fetches `manifest.json` from R2 and writes it to `model_cache/manifest.json`. If R2 is unreachable, it falls back to that cached copy. If neither exists, startup fails.
3. Loads the router model from disk, or downloads it from R2 if it isn't cached. Startup fails if the router can't be loaded.

Expert models load lazily on the first request for each crop.

## Docker / Railway

```bash
docker build -t nomaapp-inference .
docker run --env-file .env -p 8000:8000 -v $(pwd)/model_cache:/app/model_cache nomaapp-inference
```

The container listens on `$PORT` (default `8000`). **Always run a single worker.** Each extra worker loads its own copy of the router, races the others for R2 downloads, and pushes a 4 GB box past its memory limit. To keep the disk cache across deploys on Railway, mount a volume at `/app/model_cache`.

## R2 bucket layout

```
<R2_BUCKET_NAME>/
├── manifest.json
├── router/
│   └── crop_identifier_v1.onnx
└── experts/
    ├── maize_v1.onnx
    ├── cassava_v1.onnx
    └── ...
```

`manifest.json`:

```json
{
  "version": "1.2.0",
  "router": {
    "file": "router/crop_identifier_v1.onnx",
    "version": "1.0.0",
    "input_size": [224, 224],
    "classes": ["cassava", "maize", "tomato", "pepper", "yam"]
  },
  "experts": {
    "maize": {
      "file": "experts/maize_v1.onnx",
      "version": "1.0.0",
      "diseases": ["northern_blight", "gray_leaf_spot", "common_rust", "healthy"]
    }
  }
}
```

- Router `classes` and each expert's `diseases` must be in the model's output order.
- Expert keys must match the router's class names (case-insensitive).
- A router class with no expert entry produces `unsupported_crop`.
- Router classes listed in `ROUTER_UNSUPPORTED_CLASSES` or `ROUTER_NO_PLANT_CLASSES` act as catch-all rejections.
- The disk cache mirrors the bucket keys (`model_cache/experts/maize_v1.onnx`). To ship a new model version, use a new filename (`maize_v2.onnx`) and bump the manifest. The new file is fetched automatically on next startup.

### Model requirements

- **Single-file ONNX.** Weights must be embedded, not stored in a `.onnx.data` sidecar, because the service downloads exactly one object per model. When exporting, use `onnx.save(model, path, save_as_external_data=False)`. Models must be under 2 GB.
- Input: one float32 image tensor, either `(N, 3, H, W)` or `(N, H, W, 3)`. The layout is detected automatically.
- Preprocessing is fixed for every model: RGB → resize to `input_size` (bilinear) → `/255` → ImageNet mean/std normalisation. Models must have been trained with the same normalisation.
- Output: one `(N, num_classes)` tensor. **Raw logits are preferred**, because the energy-score OOD gate needs them. If a model ends in Softmax, the service detects this and falls back to the softmax-confidence threshold, and `ood_score` is returned as `null`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `R2_ENDPOINT_URL` | — | `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY` / `R2_SECRET_KEY` | — | R2 API token credentials |
| `R2_BUCKET_NAME` | `nomaapp-models` | Bucket name |
| `R2_CONNECT_TIMEOUT_S` / `R2_READ_TIMEOUT_S` | `5` / `60` | R2 timeouts in seconds |
| `R2_MAX_ATTEMPTS` | `3` | boto3 retry attempts |
| `ROUTER_CONFIDENCE_THRESHOLD` | `0.70` | Router max probability below this → `crop_not_identified` |
| `ROUTER_UNSUPPORTED_CLASSES` | `other,unknown,unsupported` | Catch-all router classes → `unsupported_crop` |
| `ROUTER_NO_PLANT_CLASSES` | `no_plant,not_plant,background` | Router classes → `no_plant_detected` |
| `ENERGY_TEMPERATURE` | `1.0` | `T` in `E(x) = -T·log Σ exp(logit/T)` |
| `ENERGY_OOD_THRESHOLD` | `0.0` | Energy above this → `uncertain_disease_pattern`. **Tune on a validation set.** |
| `DISEASE_CONFIDENCE_THRESHOLD` | `0.50` | Fallback gate used when a model outputs probabilities |
| `MODEL_OUTPUT_TYPE` | `auto` | `auto`, `logits` or `probabilities` |
| `MAX_WARM_EXPERTS` | `3` | Expert models kept in RAM |
| `MODEL_CACHE_DIR` | `model_cache` | Disk cache directory |
| `BLUR_THRESHOLD` | `50.0` | Laplacian variance below this → blurry |
| `MAX_UPLOAD_MB` | `10` | Larger uploads are rejected |
| `MIN_IMAGE_SIDE_PX` | `100` | Shorter side below this → too small |
| `MIN_BRIGHTNESS` / `MAX_BRIGHTNESS` | `20` / `235` | Mean gray-level bounds |
| `QUALITY_ANALYSIS_MAX_SIDE_PX` | `512` | Blur and brightness are measured on a copy downscaled to this size |
| `ADMIN_KEY` | — | Key for `DELETE /cache/{crop}`. If empty, admin endpoints are disabled. |
| `PORT` | `8000` | HTTP port |
| `LOG_LEVEL` | `INFO` | Log level |

## API

### `POST /diagnose`

Takes `multipart/form-data` with a `file` field containing a JPEG or PNG. There is no auth on this endpoint; the Express API authenticates callers before forwarding.

```bash
curl -X POST http://localhost:8000/diagnose -F "file=@leaf.jpg"
```

200, diagnosis:

```json
{
  "status": "identified",
  "crop": "maize",
  "crop_confidence": 0.94,
  "disease": "northern_blight",
  "disease_confidence": 0.87,
  "all_probabilities": {"northern_blight": 0.87, "gray_leaf_spot": 0.08, "healthy": 0.05},
  "model_source": "memory",
  "ood_score": -2.45,
  "stage_timings": {"quality_ms": 3.1, "router_ms": 41.2, "expert_ms": 88.0, "total_ms": 133.9}
}
```

`model_source` is `memory`, `disk` or `bucket`. A `bucket` value means the model was pulled cold from R2, so latency was higher.

422, rejected:

```json
{
  "status": "rejected",
  "rejection_reason": "crop_not_identified",
  "message": "Could not identify the crop. Please retake the photo in clear daylight with the leaf filling the frame.",
  "confidence": 0.42,
  "stage_timings": {"quality_ms": 3.0, "router_ms": 40.1, "expert_ms": null, "total_ms": 43.5}
}
```

`rejection_reason` is one of `poor_image_quality`, `crop_not_identified`, `unsupported_crop`, `no_plant_detected` or `uncertain_disease_pattern`.

Other statuses:

- **400**: the request has no `file` field.
- **503**: a model could not be loaded, for example because R2 was unreachable on a cold load.
- **500**: an unexpected error.

All of these return `{"status": "error", "message": "..."}` and never include a traceback.

### `GET /health`

```json
{"status": "ok", "router_ready": true, "warm_models": ["maize", "tomato"], "manifest_version": "1.2.0"}
```

### `GET /cache/status`

```json
{"warm_models": ["maize", "tomato"], "disk_cached": ["cassava", "maize", "tomato"], "capacity": 3}
```

### `DELETE /cache/{crop}`

Evicts the crop's expert from memory and deletes it from disk, so the next request fetches it fresh from R2. Use this to hot-swap a retrained model that kept the same filename.

```bash
curl -X DELETE http://localhost:8000/cache/maize -H "X-Admin-Key: $ADMIN_KEY"
```

## Logging and threshold tuning

Logs are one JSON object per line. Every `/diagnose` request emits:

- a `quality check` line with `blur_score`, `brightness`, dimensions and `quality_ms`;
- a `diagnosis` line with `request_id`, `status`, `crop`, `disease`, `model_source`, `ood_score`, `ood_method`, `rejection_reason` and `stage_timings`.

Cache loads and evictions log process RSS (`rss_mb`).

Use these logs to tune the thresholds. Plot `ood_score` for known-good diagnoses against out-of-distribution photos, then set `ENERGY_OOD_THRESHOLD` between the two distributions. Tune `BLUR_THRESHOLD` the same way using `blur_score`.
