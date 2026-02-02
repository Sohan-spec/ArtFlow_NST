# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory to the user's home directory
WORKDIR /code

# Install system dependencies required for OpenCV and other tools
# libgl1-mesa-glx is needed for cv2.imshow alternatives even if headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Perform operations as the non-root user from here on
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
# We need to switch back to root temporarily to copy files and set permissions if we want to be strict,
# but COPY --chown is cleaner.
COPY --chown=user . $HOME/app

# Make sure the data directories exist and are writable
RUN mkdir -p data/content-images \
    data/style-images \
    data/output-images \
    data/temp-styles

# Expose port 7860 for Hugging Face
EXPOSE 7860

# Define the command to run the application
CMD ["python", "app.py"]
