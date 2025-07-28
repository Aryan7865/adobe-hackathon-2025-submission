# Use a lightweight Python base image compatible with amd64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# We use --no-cache-dir to keep the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded model into the container
# This is the key step for offline execution
COPY ./models/ /app/models/

# Copy your Python script into the container
COPY process_1b.py .

# Create the input and output directories as expected by the run command
# The 'docs' subdirectory is where the PDFs will be
RUN mkdir -p /app/input/docs /app/output

# This command will be run when the container starts
CMD ["python", "process_1b.py"]