#!/usr/bin/env python
import os, sys, json, shlex, subprocess, pathlib, argparse, tempfile

def _python_for_uig(uig_root: pathlib.Path) -> str:
    for cand in (
        uig_root / ".venv/bin/python",
        uig_root / "venv/bin/python",
        uig_root / ".venv/Scripts/python.exe",
        uig_root / "venv/Scripts/python.exe",
    ):
        if cand.exists():
            return str(cand)
    return "python"

def _ingest_supports_config_json(py: str, ingest_py: str) -> bool:
    try:
        out = subprocess.check_output([py, ingest_py, "--help"], text=True, stderr=subprocess.STDOUT)
        return "--config-json" in out
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Ingest a PA folder via UIG")
    ap.add_argument("--uig", required=True, help="Path to UIG repo root (folder with ingest.py)")
    ap.add_argument("--folder", required=True, help="Folder with .txt/.md files to ingest")
    ap.add_argument("--env-file", default=".env.pa", help="Env file path relative to UIG root (default: .env.pa)")
    ap.add_argument("--tags", default="", help="Comma-separated tags to attach as metadata")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("-n", "--dry-run", action="store_true", help="Run without upserting to Pinecone")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    uig_root = pathlib.Path(args.uig).resolve()
    ingest_py = str(uig_root / "ingest.py")
    if not os.path.exists(ingest_py):
        print(f"[PA→UIG] ingest.py not found at: {ingest_py}", file=sys.stderr); sys.exit(2)

    cfg = {
        "project": "pa",
        "pinecone_index": "pa",
        "embedding_model": "openai-3-large",   # 3072-d to match your existing 'pa' index
        "dimension": 3072,
        "env_file": args.env_file,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "input_streams": [{
            "type": "notes",
            "path": str(pathlib.Path(args.folder).resolve()),
            "content_types": ["txt","md"]
        }],
        "metadata": {"project": "PA", "assistant": "pa",
                     "tags": [t.strip() for t in args.tags.split(",") if t.strip()]}
    }

    py = _python_for_uig(uig_root)
    use_inline = _ingest_supports_config_json(py, ingest_py)

    if use_inline:
        cfg_json = json.dumps(cfg)
        cmd = f'{shlex.quote(py)} {shlex.quote(ingest_py)} --config-json {shlex.quote(cfg_json)}'
    else:
        # Fallback: write temp config and call --config
        suffix = ".json"  # your ingest can read YAML or JSON; JSON is fine
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
        json.dump(cfg, tf)
        tmp_path = tf.name

    cmd = f'{shlex.quote(py)} {shlex.quote(ingest_py)} --config {shlex.quote(tmp_path)}'
    if args.verbose: cmd += " -v"
    if args.dry_run: cmd += " --dry-run"
    print("[PA→UIG] Running:", cmd)
    rc = subprocess.call(cmd, shell=True, cwd=str(uig_root), env=os.environ.copy())
    sys.exit(rc)

if __name__ == "__main__":
    main()
