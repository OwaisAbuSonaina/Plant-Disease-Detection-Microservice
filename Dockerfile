# 1. Base Image: Lightweight Python environment
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /code

# 3. Copy dependencies and install them

COPY ./requirements.txt /code/requirements.txt

# Install CPU-only torch to keep image size small 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /code/requirements.txt

# 4. Copy the rest of the application
COPY ./app /code/app
COPY ./models /code/models

# 5. Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]