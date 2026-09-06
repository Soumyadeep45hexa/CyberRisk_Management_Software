# Dockerfile - builds a container image for the Cyber Risk Dashboard
#
# Build:  docker build -t cyber-risk-dashboard .
# Run:    docker run -p 8501:8501 cyber-risk-dashboard

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer = faster rebuilds when only
# your code changes, not your requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Streamlit's default port
EXPOSE 8501

# --server.address=0.0.0.0 is required so the app is reachable from
# outside the container, not just from inside it
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]