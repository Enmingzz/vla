#!/usr/bin/env python3
"""Download the exact public GCS checkpoint; verify size and GCS MD5/CRC32C."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


def download(destination, workers=4):
    prefix = "checkpoints/pi05_libero/"
    base = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?"
    items, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": 1000}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(base + urllib.parse.urlencode(query), timeout=60) as response:
            page = json.load(response)
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            break
    items = [i for i in items if not i["name"].endswith("/")]
    if not items or not any(i["name"].endswith("norm_stats.json") for i in items):
        raise RuntimeError("Incomplete GCS object listing")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def one(item):
        name = item["name"][len(prefix):]
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)

        def valid(path):
            if not path.exists() or path.stat().st_size != int(item["size"]):
                return False
            if "md5Hash" in item:
                h, expected = hashlib.md5(), item["md5Hash"]
            else:
                import google_crc32c
                h, expected = google_crc32c.Checksum(), item["crc32c"]
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(block)
            return base64.b64encode(h.digest()).decode() == expected

        if not valid(target):
            tmp = target.with_suffix(target.suffix + ".partial")
            # Pin generation to avoid mixing objects if a public checkpoint is replaced mid-download.
            if not valid(tmp):
                url = "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/" + urllib.parse.quote(item["name"], safe="")
                url += "?alt=media&generation=" + item["generation"]
                with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as f:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        f.write(block)
            if not valid(tmp):
                raise RuntimeError("Checksum failure: " + name)
            os.replace(tmp, target)
        print("Verified " + name, flush=True)
        return {k: item[k] for k in ["name", "size", "md5Hash", "crc32c", "generation"] if k in item}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(one, items))
    manifest = {"source": "gs://openpi-assets/checkpoints/pi05_libero", "objects": sorted(verified, key=lambda i: i["name"])}
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified checkpoint: {} ({:.2f} GB)".format(destination, sum(int(i["size"]) for i in items) / 1e9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    import fcntl
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        download(args.destination, args.workers)
