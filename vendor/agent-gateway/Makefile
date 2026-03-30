.PHONY: install lint format test type-check pre-commit serve smoke docker-build docker-up docker-down start stop restart

install:
	poetry install

lint:
	poetry run ruff check src tests

format:
	poetry run black src tests && poetry run ruff check --fix src tests

pre-commit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest

type-check:
	poetry run mypy src

serve:
	poetry run gateway --reload

start: docker-up

stop: docker-down

restart: stop start

smoke:
	poetry run python scripts/smoke_tests.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

docker-down:
	docker compose -f docker/docker-compose.yml down
