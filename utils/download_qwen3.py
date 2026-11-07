"""Download the pinned Qwen3-0.6B snapshot and verify every file, without a GPU.

Requires Python 3 and curl. Pass --endpoint https://hf-mirror.com
when the official Hugging Face endpoint is unavailable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = (
    "LICENSE", "README.md", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)


def hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest()


def verify(path: Path, entry: dict) -> str:
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"File size mismatch: {path}")
    sha256, git_sha1 = hashes(path)
    lfs_sha = entry.get("lfs", {}).get("sha256")
    if (lfs_sha and sha256 != lfs_sha) or (not lfs_sha and git_sha1 != entry["blobId"]):
        raise ValueError(f"Hub checksum mismatch: {path}")
    return sha256


def download_ranges(url: str, partial: Path, size: int, connections: int) -> None:
    """Fetch disjoint ranges concurrently, keeping a previously downloaded prefix."""
    offset = partial.stat().st_size if partial.exists() else 0
    parts_dir = partial.with_name(partial.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    chunks = [(start, min(start + 64 * 1024**2, size) - 1) for start in range(offset, size, 64 * 1024**2)]

    def fetch(chunk: tuple[int, int]) -> Path:
        start, end = chunk
        part = parts_dir / f"{start}-{end}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(5):
            try:
                result = subprocess.run([
                    "curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "15",
                    "--speed-limit", "1024", "--speed-time", "60", "--range", f"{start}-{end}",
                    "--output", str(part), "--write-out", "%{http_code}", url,
                ], check=True, capture_output=True, text=True)
                if result.stdout != "206" or part.stat().st_size != end - start + 1:
                    raise ValueError(f"Server did not honor byte range {start}-{end}")
                print(f"Range verified by size: {start}-{end}", flush=True)
                return part
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Range retries exhausted")

    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        # Preserve chunk order even when downloads finish out of order.
        parts = list(pool.map(fetch, chunks))
    with partial.open("ab") as stream:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, stream, 8 * 1024**2)
    for part in parts:
        part.unlink()
    # Keep any chunks from an older interrupted merge available for inspection.
    if not any(parts_dir.iterdir()):
        parts_dir.rmdir()


def download_file(url: str, target: Path, entry: dict, connections: int = 4) -> str:
    if target.exists():
        return verify(target, entry)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(5):
        try:
            if connections > 1 and entry["size"] > 64 * 1024**2:
                download_ranges(url, partial, entry["size"], connections)
            elif not partial.exists() or partial.stat().st_size != entry["size"]:
                # curl handles CDN redirects and supports resuming interrupted downloads.
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "15", "--speed-limit", "1024", "--speed-time", "60",
                                "--continue-at", "-", "--output", str(partial), url], check=True)
            checksum = verify(partial, entry)
            partial.replace(target)
            return checksum
        except Exception as error:
            # A complete but invalid file must be fetched again, not resumed.
            if partial.exists() and (partial.stat().st_size >= entry["size"] or
                                    isinstance(error, subprocess.CalledProcessError) and error.returncode == 33):
                partial.unlink()
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}: {target.name}: {error}", flush=True)
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("Download retry loop exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("models/Qwen3-0.6B"))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--connections", type=int, default=4, choices=range(1, 17), metavar="1..16")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    metadata_url = f"{endpoint}/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Unexpected model revision")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    args.destination.mkdir(parents=True, exist_ok=True)
    verified = {}
    for name in FILES:
        print(f"Downloading/verifying {name}", flush=True)
        digest = download_file(f"{endpoint}/{MODEL_ID}/resolve/{REVISION}/{name}", args.destination / name, entries[name], args.connections)
        verified[name] = {"size": entries[name]["size"], "sha256": digest}
        print(f"Verified {name}: {digest}", flush=True)
    manifest = {"status": "completed", "model_id": MODEL_ID, "revision": REVISION,
                "endpoint": endpoint, "files": verified, "gpu_inference": "Not Run"}
    (args.destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "completed", "destination": str(args.destination.resolve()),
                      "revision": REVISION, "bytes": sum(item["size"] for item in verified.values())}), flush=True)


if __name__ == "__main__":
    main()
