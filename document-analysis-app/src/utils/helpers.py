def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def save_results(results, output_file):
    """Saves results to a specified output file."""
    import json
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

def save_json_atomic(data, output_path: str, indent: int = 2, ensure_ascii: bool = False):
    """Safely save JSON to disk using a temporary file and atomic replace."""
    import json
    import os
    import tempfile
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            json.dump(data, tmp_file, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

def load_json_if_exists(path: str):
    """Load JSON from path if it exists; return None otherwise."""
    import os
    import json
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

