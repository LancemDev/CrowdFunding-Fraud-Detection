import kagglehub

# Download latest version
path = kagglehub.dataset_download("kemical/kickstarter-projects")

print("Path to dataset files:", path)