# ===========================================================================
# Dockerfile  -  builds the container that runs on Hugging Face Spaces
# ===========================================================================
#
# WHAT A DOCKER IMAGE IS
# ----------------------
# Think of it as a sealed lunchbox containing the operating system, Python,
# every library, and our code - all frozen at exact versions. Hugging Face
# runs that lunchbox. Because everything inside is pinned, the app behaves
# identically on a laptop and in the cloud. "It works on my machine" stops
# being a problem, because the machine travels with the app.
#
# WHY TWO STAGES
# --------------
# Building the React app needs Node.js and ~150 MB of npm packages. RUNNING it
# needs none of that - just the compiled HTML/CSS/JS files.
#
# So stage 1 uses a Node image purely to compile the frontend, and stage 2
# starts from a clean Python image and copies ONLY the finished files across.
# Node and node_modules never reach the final image, which keeps it small.
# ===========================================================================


# ---------------------------------------------------------------------------
# STAGE 1  -  compile the React frontend
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /app/frontend

# Copy ONLY the dependency files first. Docker caches each step, so as long as
# package.json has not changed it will reuse the previously installed
# node_modules instead of re-downloading everything - turning a 40-second step
# into an instant one on most rebuilds.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Now copy the source and build. Vite writes its output to ../static,
# i.e. /app/static, which stage 2 picks up.
COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# STAGE 2  -  the actual runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Hugging Face Spaces run containers as user ID 1000, never as root. Creating
# that user ourselves and switching to it early avoids permission errors on
# every file we copy in afterwards.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install Python dependencies before copying the application code. Same caching
# logic as above: requirements.txt changes rarely, app.py changes constantly.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# The backend, the trained model, and the two small data files it serves.
COPY --chown=user app.py ./
COPY --chown=user model.joblib model_metadata.json app_config.json ./
COPY --chown=user dashboard_data.parquet ./

# The compiled React app from stage 1.
COPY --chown=user --from=frontend /app/static ./static

# 7860 is the port Hugging Face Spaces expects. It must match app_port in
# README.md, and the server must bind to 0.0.0.0 (not 127.0.0.1) so that
# traffic from outside the container can reach it.
EXPOSE 7860

# A quick self-check so the platform can tell a hung container from a healthy
# one. It calls the /api/health endpoint we built into app.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
