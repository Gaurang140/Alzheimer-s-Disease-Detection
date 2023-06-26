import os

artifact_folder = "artifacts"

# Retrieve a list of all folders within the artifact folder
folders = [f for f in os.listdir(artifact_folder) if os.path.isdir(os.path.join(artifact_folder, f))]

# Sort the list of folders based on their timestamps in descending order
sorted_folders = sorted(folders, key=lambda x: os.path.getmtime(os.path.join(artifact_folder, x)), reverse=True)

# Select the first folder from the sorted list, which will be the latest folder
latest_folder = sorted_folders[0] if sorted_folders else None

# Construct the yaml_out_path in the desired format
if latest_folder:
    yaml_out_path = os.path.join(artifact_folder, latest_folder, "data_ingestion")
    yaml_out_path = yaml_out_path.replace("\\", "/")  # Convert backslashes to forward slashes

    yaml_content = f"""
stages:
  data_ingestion:
    cmd: python src/alzheimer_disease/components/data_ingestion.py
    deps:
      - src/diseaseClassifier/pipeline/s01_data_ingestion.py
      - config/config.yaml
      - artifacts
    outs:
      - {yaml_out_path}
"""

    # Write the modified YAML content to a file
    with open("your_yaml_file.yaml", "w") as file:
        file.write(yaml_content)
