# Use a stable, lightweight Python runtime base image
FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /app

# Copy dependency specifications first to leverage Docker build caching
COPY pyproject.toml /app/

# Install pip updates and application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the codebase into the container
COPY . /app/

# Expose FastAPI server port
EXPOSE 8000

# Start script running migrations and uvicorn server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
