# pa_uig_bridge.py
import os, json, tempfile, subprocess, shlex, pathlib

# Set this once (or export UIG_ROOT in your shell)
UIG_ROOT = os.environ.get("UIG_ROOT", "/home/keizer/projects/UIG/UIG_App")

def _uig_python(uig_root: pathlib.Path) -> str:
    for cand in (
        uig_root / ".venv/bin/python",
        uig_root / "venv/bin/python",
        uig_root / ".venv/Scripts/python.exe",
        uig_root / "venv/Scripts/python.exe",
    ):
        if cand.exists():
            return str(cand)
    return "python"

def _run(cmd: str, cwd: str) -> int:
    print("[PA→UIG] Running:", cmd)
    return subprocess.call(cmd, shell=True, cwd=cwd, env=os.environ.copy())

def trigger_uig_on_folder(folder: str, *, tags=(), dry_run=False, verbose=True) -> int:
    """
    Send a folder of files to UIG ingest (PA → index 'pa', openai-3-large 3072d).
    Uses temp JSON + `--config` so it works with all ingest.py versions.
    """
    uig_root = pathlib.Path(UIG_ROOT).resolve()
    ingest_py = uig_root / "ingest.py"
    if not ingest_py.exists():
        raise FileNotFoundError(f"ingest.py not found at {ingest_py}")

    cfg = {
        "project": "pa",
        "pinecone_index": "pa",
        "embedding_model": "openai-3-large",
        "dimension": 3072,
        "env_file": ".env.pa",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "input_streams": [{
            "type": "notes",
            "path": str(pathlib.Path(folder).resolve()),
            "content_types": ["txt", "md", "pdf", "rtf", "csv", "docx"]
        }],
        "metadata": {"project": "PA", "assistant": "pa",
                     "tags": [t.strip() for t in (tags or []) if t and str(t).strip()],"namespace": "pa"}
    }

    # Write temp JSON and call ingest.py --config <tempfile>
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
        json.dump(cfg, tf)
        tmp_path = tf.name

    py = _uig_python(uig_root)
    cmd = f'{shlex.quote(py)} {shlex.quote(str(ingest_py))} --config {shlex.quote(tmp_path)}'
    if verbose: cmd += " -v"
    if dry_run: cmd += " --dry-run"
    return _run(cmd, cwd=str(uig_root))

def trigger_uig_with_text(title: str, text: str, *, tags=(), meta_type=None,
                          extra_metadata: dict | None = None,
                          dry_run=False, verbose=True):
    """
    Save text to a temp .txt file and call ingest with metadata:
    - tags: list[str]
    - meta_type: e.g. "prefilled_metrics", "symptom_log", "note", ...
    - extra_metadata: dict of structured fields to merge (numbers, dates, sym_*…)
    """
    import tempfile, json, os, pathlib, shlex, subprocess

    staging = tempfile.mkdtemp(prefix="pa_uig_")
    safe = "".join(c for c in (title or "note") if c.isalnum() or c in (" ", "_", "-")).strip() or "note"
    fname = (safe[:60] + ".txt").replace(" ", "_")
    path = os.path.join(staging, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

    # ---- build config
    tags_list = [t.strip() for t in (tags or []) if t and str(t).strip()]
    meta = {"project": "PA", "assistant": "pa", "tags": tags_list, "namespace": "pa"}
    if meta_type:
        meta["type"] = meta_type
    if extra_metadata:
        meta.update(extra_metadata)

    cfg = {
        "project": "pa",
        "pinecone_index": "pa",
        "embedding_model": "openai-3-large",
        "dimension": 3072,
        "env_file": ".env.pa",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "input_streams": [{
            "type": "notes",
            "path": str(pathlib.Path(staging).resolve()),
            "content_types": ["txt","md","pdf","rtf","csv","docx"]
        }],
        "metadata": meta
        # (no namespace key -> default, which you mapped to NS="pa")
    }

    # write temp config and call ingest.py --config
    import tempfile, json
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
        json.dump(cfg, tf)
        tmp_path = tf.name

    uig_root = pathlib.Path(os.environ.get("UIG_ROOT", "/home/keizer/projects/UIG/UIG_App")).resolve()
    ingest_py = uig_root / "ingest.py"
    py = str((uig_root / ".venv/bin/python"))
    cmd = f"{py} {shlex.quote(str(ingest_py))} --config {shlex.quote(tmp_path)}"
    if verbose: cmd += " -v"
    if dry_run: cmd += " --dry-run"
    print("[PA→UIG] Running:", cmd)
    return staging, subprocess.call(cmd, shell=True, cwd=str(uig_root), env=os.environ.copy())


# in save_uploaded_files_to_staging(...): log what's written
def save_uploaded_files_to_staging(uploaded_files, *, subdir_prefix="pa_files_") -> str:
    import os, tempfile
    staging = tempfile.mkdtemp(prefix=subdir_prefix)
    names = []
    for uf in uploaded_files or []:
        out = os.path.join(staging, uf.name)
        with open(out, "wb") as f:
            f.write(uf.getbuffer())
        names.append(uf.name)
    print(f"[PA] Saved {len(names)} file(s) to staging {staging}: {names}")
    return staging
