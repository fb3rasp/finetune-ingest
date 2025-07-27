def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def load_config(config_file):
    """Loads configuration from a specified file."""
    import json
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_results(results, output_file):
    """Saves results to a specified output file."""
    import json
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

def validate_data(data):
    """Validates the input data for required fields."""
    if not data:
        raise ValueError("No data provided.")
    # Add more validation rules as needed
    return True