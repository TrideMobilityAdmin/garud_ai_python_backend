import yaml
# Load the YAML config file
def load_config():
    with open("app\core\config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
