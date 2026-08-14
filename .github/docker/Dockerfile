FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy="http://proxy-dmz.intel.com:912"
ENV https_proxy="http://proxy-dmz.intel.com:912"

# Install build tools and C++ standard library (needed by icpx)
# Proxy is passed inline so it is NOT persisted in image metadata/history
RUN apt-get update && \
    apt-get install -y \
    cmake \
    ninja-build \
    git \
    python3 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory — all CI operations live under /workspace
RUN mkdir -p /workspace/source /workspace/output /workspace/bin /workspace/tmp

# Source code mounted at /workspace/source (read-only)
# Build output goes to /workspace/output (mounted to /shared/builds/<run-id>)
# oneAPI mounted at /opt/intel/oneapi (read-only)

WORKDIR /workspace

# entrypoint.sh is mounted at runtime from the source tree (not baked into the image)
# so it always matches the current branch's version.
ENTRYPOINT ["/workspace/bin/entrypoint.sh"]
