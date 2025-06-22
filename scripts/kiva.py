import kagglehub

# Download latest version
path = kagglehub.dataset_download("kiva/data-science-for-good-kiva-crowdfunding")

print("Path to dataset files:", path)