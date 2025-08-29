# Use official Python image
FROM python:3.13-slim

# ------------------ Install system dependencies ------------------
# Required for PyAudio and general build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------ Set working directory ------------------
WORKDIR /app

# ------------------ Copy requirements ------------------
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ------------------ Copy app code ------------------
COPY . .

# ------------------ Expose the port Render will provide ------------------
ENV PORT=5000
EXPOSE 5000

# ------------------ Start the app ------------------
# Using gunicorn for Flask + production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "streaming_app:app"]
