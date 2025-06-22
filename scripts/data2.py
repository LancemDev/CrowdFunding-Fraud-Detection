import kagglehub

# Download latest version
path = kagglehub.dataset_download("socathie/kickstarter-project-statistics")

print("Path to dataset files:", path)