# Python runtime as a parent image
FROM python:3.8-slim

# Working directory
WORKDIR /app

# Create the necessary directories
RUN mkdir -p /app/ /app/models  

# Copy the files from your host machine into the container
COPY app.py /app/
COPY requirements.txt /app/
COPY models/ /app/models/


# Update the repositories and install Java for H20.ai AutoML 
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean;


# Create and activate a virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install requirements.txt
RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the default command to run your Streamlit application
CMD ["streamlit", "run", "app.py"]
