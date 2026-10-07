"""Upload the packed training data (training/pack_dataset.py) to the Stratus bucket the cloud trainer reads.

Uses the signed-in Catalyst CLI (`catalyst login`) for access, so nothing secret is stored. The manifest goes
last, so the trainer never sees a half-uploaded dataset. --init uploads starting weights (a YOLO .pt) that the
trainer fine-tunes instead of starting from the generic COCO weights.

Usage: python catalyst/upload_dataset.py [--data data/cloud] [--bucket colony-counter-ml] [--init best.pt]
"""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DOMAINS = {"in": "zohostratus.in", "us": "zohostratus.com", "eu": "zohostratus.eu", "au": "zohostratus.com.au",
           "jp": "zohostratus.jp", "ca": "zohostratus.ca", "sa": "zohostratus.sa"}


def cli_token():
    out = subprocess.run(["node", str(Path(__file__).with_name("cli_token.js"))], capture_output=True, text=True,
                         shell=True)
    if out.returncode:
        raise SystemExit(out.stderr.strip() or "could not get a token from the Catalyst CLI")
    return json.loads(out.stdout)


def put(base, key, data, size):
    for attempt in range(4):
        cred = cli_token()
        req = urllib.request.Request(f"{base}/{key}", data=data() if callable(data) else data, method="PUT", headers={
            "Authorization": f"Zoho-oauthtoken {cred['token']}", "compress": "false", "overwrite": "true",
            "Content-Length": str(size), "Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if e.code < 500 and e.code != 429:
                raise SystemExit(f"upload of {key} refused ({e.code}): {msg}")
            print(f"  {key}: {e.code}, retrying")
        except OSError as e:
            print(f"  {key}: {e}, retrying")
        time.sleep(10 * (attempt + 1))
    raise SystemExit(f"upload of {key} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/cloud"))
    ap.add_argument("--bucket", default="colony-counter-ml")
    ap.add_argument("--env", default="development", choices=["development", "production"])
    ap.add_argument("--prefix", default="dataset/")
    ap.add_argument("--init", type=Path, help="starting weights for the trainer (optional)")
    args = ap.parse_args()

    dc = cli_token()["dc"]
    host = f"{args.bucket}-{args.env}" if args.env == "development" else args.bucket
    base = f"https://{host}.{DOMAINS.get(dc, 'zohostratus.com')}/{args.prefix.strip('/')}"
    manifest = json.loads((args.data / "manifest.json").read_text())
    files = [manifest["val"], manifest["plates_val"], manifest["plates_test"], *manifest["train"]]
    if args.init:
        manifest["init"] = "init.pt"
    uploads = [(f, args.data / f) for f in files] + ([("init.pt", args.init)] if args.init else [])
    total = sum(p.stat().st_size for _, p in uploads)
    done, t0 = 0, time.time()
    for i, (key, path) in enumerate(uploads, 1):
        size = path.stat().st_size
        put(base, key, lambda p=path: open(p, "rb"), size)
        done += size
        print(f"  {i}/{len(uploads)} {key} ({size / 1e6:.0f} MB), {done / 1e6:.0f}/{total / 1e6:.0f} MB, "
              f"{done / 1e6 / max(time.time() - t0, 1):.1f} MB/s", flush=True)
    body = json.dumps(manifest, indent=2).encode()
    put(base, "manifest.json", body, len(body))
    print(f"uploaded {len(uploads) + 1} files to {base}/")


if __name__ == "__main__":
    main()
