import boto3
import json
import os
from pathlib import Path

CACHE_DIR = Path("model_cache")
BUCKET_NAME = "nomaapp-models"

r2_client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],  # https://<account>.r2.cloudflarestorage.com
    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    region_name="auto"
)

def fetch_manifest() -> dict:
    """Always fetch fresh manifest from bucket on startup."""
    response = r2_client.get_object(Bucket=BUCKET_NAME, Key="manifest.json")
    manifest = json.loads(response['Body'].read())
    
    # Cache locally
    local_path = CACHE_DIR / "manifest.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(manifest, indent=2))
    
    return manifest

def download_model(bucket_key: str, local_path: Path) -> Path:
    """Download model from R2 to local disk cache."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Registry] Downloading {bucket_key} from R2...")
    r2_client.download_file(BUCKET_NAME, bucket_key, str(local_path))
    print(f"[Registry] Cached to {local_path}")
    return local_path

def get_expert_local_path(crop: str, manifest: dict) -> Path:
    expert_info = manifest["experts"][crop]
    filename = Path(expert_info["file"]).name
    return CACHE_DIR / "experts" / filename