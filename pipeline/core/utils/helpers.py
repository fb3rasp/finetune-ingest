def log_message(message: str):
    """Logs a message to the console.
    Note: Apps may wrap this to also write to /data/logs.
    """
    print(f"[LOG] {message}")


def save_results(results, output_file: str):
    """Saves results to a specified output file."""
    import json
    import os
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)


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


def load_yaml_if_exists(path: str):
    """Load YAML from path if it exists; return None otherwise."""
    import os
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
    
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_atomic(data, output_path: str):
    """Safely save YAML to disk using a temporary file and atomic replace."""
    import os
    import tempfile
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
    
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".yaml", dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            yaml.safe_dump(data, tmp_file, default_flow_style=False, allow_unicode=True, indent=2)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
