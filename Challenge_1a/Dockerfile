# Use a lightweight Python base image compatible with amd64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY process_pdfs.py .

# Create the input and output directories as expected by the run command
RUN mkdir -p /app/input /app/output

# This command will be run when the container starts
CMD ["python", "process_pdfs.py"]