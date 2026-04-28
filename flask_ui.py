import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

app = Flask(__name__, static_folder="ui", template_folder="templates")


def list_result_files():
    files = []
    candidates = [path for path in RESULTS_DIR.glob("*.json") if path.is_file()]
    candidates.sort(
        key=lambda item: (
            0 if item.name.startswith("acsd_compare") else 1 if "compare" in item.name else 2,
            -item.stat().st_mtime,
            item.name,
        )
    )
    for path in candidates:
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_ts": stat.st_mtime,
            }
        )
    return files


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/results")
def api_results():
    return jsonify({"results": list_result_files()})


@app.get("/api/results/<path:filename>")
def api_result(filename):
    candidate = (RESULTS_DIR / filename).resolve()
    if candidate.parent != RESULTS_DIR.resolve() or not candidate.exists():
        abort(404)
    with open(candidate) as handle:
        payload = json.load(handle)
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
