IMAGE_NAME=python_machine
CONTAINER_NAME=python_machine_container

HOST_PORT=8000
CONTAINER_PORT=8000

MODELS_PATH =./models/
CONTAINER_MODELS_PATH=/app/models/

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --name $(CONTAINER_NAME) -p $(HOST_PORT):$(CONTAINER_PORT) -v $(MODELS_PATH):$(CONTAINER_MODELS_PATH) $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)

rm:
	docker rm $(CONTAINER_NAME)

all: build run

clean:
	docker rmi $(IMAGE_NAME) 
