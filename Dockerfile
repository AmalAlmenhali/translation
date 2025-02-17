# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory inside the container
RUN mkdir /translation
WORKDIR /translation

# Copy the requirements file and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files
COPY final_model.h5 ./final_model.h5
COPY final_model.keras ./final_model.keras

# Copy all other project files
COPY . ./

# Expose the Flask API port
EXPOSE 8000

# Set the environment variable for Python path
ENV PYTHONPATH /translation

# Command to run the Flask application
CMD ["python3", "/translation/app.py"]
