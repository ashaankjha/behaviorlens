# BehaviorLens container image (local, single-user use).
#
#   docker build -t behaviorlens .
#   docker run --rm -p 127.0.0.1:8501:8501 -v behaviorlens-data:/data behaviorlens
#
# Always publish to 127.0.0.1 as above. Inside the container Streamlit has to listen on 0.0.0.0
# (Docker forwards published ports to the container's own interface), so the host-side
# `-p 127.0.0.1:...` binding is what keeps the app off the LAN (docs/AUDIT.md D1). A plain
# `-p 8501:8501` would expose uploaded videos and review data to the whole network.
FROM python:3.11-slim

# MediaPipe needs EGL/GLES even for CPU inference. mediapipe also installs opencv-contrib-python
# (non-headless), which needs libGL and glib if its cv2 wins the install race (AUDIT D7).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BEHAVIORLENS_DATA_DIR=/data \
    PYTHONPATH=/app/src

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin behaviorlens
WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY --chown=behaviorlens:behaviorlens . .
# Model weights are fetched at build time from Google's bucket and checked against the pinned
# SHA-256 (config.MODEL_SHA256); the build fails if they don't match.
RUN python scripts/download_model.py \
    && chown -R behaviorlens:behaviorlens assets/models \
    && mkdir -p /data && chown behaviorlens:behaviorlens /data

USER behaviorlens
VOLUME ["/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# 0.0.0.0 *inside* the container only; see the note at the top about publishing to 127.0.0.1.
# With a wildcard address Streamlit 1.66 GETs checkip.amazonaws.com to print an "External URL"
# (bootstrap._print_url); hideWelcomeMessage returns before that call, keeping the app offline.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false", "--logger.hideWelcomeMessage=true"]
