# Define the image name and container name
IMAGE_NAME = openems-image
CONTAINER_NAME = openems-container
LOCAL_DIR=$(shell pwd)
CONTAINER_WORKDIR = /app
# Get current user and group IDs
USER_ID = $(shell id -u)
GROUP_ID = $(shell id -g)
USER_NAME = $(shell id -un)

# Declare phony targets
.PHONY: all build run start stop rm rmi clean help

# Default target: Build and run the Docker container
all: build run

# Build the Docker image
build:
	@echo "Building Docker image $(IMAGE_NAME)..."
	docker build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) --build-arg USER_NAME=$(USER_NAME) -t $(IMAGE_NAME) .

# Run the Docker container with volume mounts for local development
run:
	@echo "Running Docker container $(CONTAINER_NAME)..."
	docker run -it --rm --name $(CONTAINER_NAME) \
		-e DISPLAY=$(DISPLAY) \
		-v "/tmp/.X11-unix:/tmp/.X11-unix" \
		-v "$(LOCAL_DIR):$(CONTAINER_WORKDIR)" \
		-u $(USER_ID):$(GROUP_ID) \
		$(IMAGE_NAME) bash

# Start the Docker container (if it's stopped)
# Note: This only works if container was created without --rm flag
start:
	@echo "Starting Docker container..."
	@docker start $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) not found or already running"

# Stop the Docker container
stop:
	@echo "Stopping Docker container..."
	@docker stop $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) not found or already stopped"

# Remove the Docker container
rm:
	@echo "Removing Docker container..."
	@docker rm $(CONTAINER_NAME) || echo "Container $(CONTAINER_NAME) not found"

# Remove the Docker image
rmi:
	@echo "Removing Docker image..."
	@docker rmi $(IMAGE_NAME) || echo "Image $(IMAGE_NAME) not found"

# Clean up: Stop and remove the container, then remove the image
clean: stop rm rmi

# Show help message
help:
	@echo "Available targets:"
	@echo "  make all     - Build and run the container (default)"
	@echo "  make build   - Build the Docker image"
	@echo "  make run     - Run the Docker container"
	@echo "  make start   - Start a stopped container"
	@echo "  make stop    - Stop a running container"
	@echo "  make rm      - Remove the container"
	@echo "  make rmi     - Remove the image"
	@echo "  make clean   - Stop, remove container, and remove image"
	@echo "  make help    - Show this help message"
