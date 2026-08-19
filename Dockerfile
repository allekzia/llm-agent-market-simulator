# Packages the dashboard into a self-contained image: anyone with Docker
# can run this identically, regardless of what's installed on their own
# machine. Deliberately does NOT bake in any API key, the dashboard
# itself never makes live LLM calls (see dashboard/app.py), so none is
# needed here either.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from the rest of the code.
# Docker caches each instruction as a layer: as long as requirements.txt
# doesn't change, this layer is reused on every future build instead of
# reinstalling everything from scratch, which makes rebuilds during
# development much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project.
COPY core/ ./core/
COPY dashboard/ ./dashboard/
COPY experiments/ ./experiments/
COPY analysis/ ./analysis/
COPY .streamlit/ ./.streamlit/

# Hugging Face Spaces expects the app to listen on port 7860 by default.
EXPOSE 7860

# --server.address=0.0.0.0 is required inside a container: Streamlit's
# default (localhost only) would make the app unreachable from outside
# the container, since "localhost" inside a container refers only to
# the container itself, not the host machine or the outside world.
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
