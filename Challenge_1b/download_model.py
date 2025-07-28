from sentence_transformers import SentenceTransformer

print("Downloading the all-MiniLM-L6-v2 model...")
# This model is small, fast, and effective, perfect for the constraints.
model_name = 'all-MiniLM-L6-v2'
# We explicitly save it to a local folder to be used by Docker.
model_path = f'./models/{model_name}'
model = SentenceTransformer(model_name)
model.save(model_path)
print(f"Model downloaded and saved to {model_path}")