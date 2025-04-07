import yaml
import json

# Read YAML file
with open("model.yaml", "r") as yaml_file:
    model_yaml = yaml.safe_load(yaml_file)

# Save as JSON file
with open("model.json", "w") as json_file:
    json.dump(model_yaml, json_file, indent=4)

print("YAML successfully converted to JSON.")
