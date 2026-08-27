import kagglehub

# Download latest version
path = kagglehub.dataset_download("aletbm/aerial-imagery-dataset-floodnet-challenge")

print("Path to dataset files:", path)