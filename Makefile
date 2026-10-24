PY ?= python
.PHONY: install data train test lint demo eval serve export-sft docker-up docker-down clean

install:
	$(PY) -m pip install -e ".[dev]"

data:
	modelroute data

train:
	modelroute train

test:
	pytest -q

lint:
	ruff check .

demo:
	modelroute demo

eval:
	modelroute eval

serve:
	modelroute serve --port 8000

export-sft:
	modelroute export-sft

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf artifacts data/generated .pytest_cache .ruff_cache
