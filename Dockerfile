# Stage 1: Builder
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy Eigen3 local copy (not downloaded)
COPY external/eigen-3.4.0/ external/eigen-3.4.0/

# Copy build system and source files
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/

# Configure and build only the fasttracker binary
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    && cmake --build build --target fasttracker --parallel $(nproc)


# Stage 2: Runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from builder stage
COPY --from=builder /src/build/fasttracker /app/fasttracker

# Copy the web application
COPY python/webapp/ /app/python/webapp/

# Install Flask
RUN pip3 install flask

# Environment variable pointing to the fasttracker binary
ENV FASTTRACKER_EXE=/app/fasttracker

EXPOSE 5000

CMD ["python3", "python/webapp/app.py"]
