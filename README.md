# Python Fantasy Map Generator (py-fmg)

Python port of the Fantasy Map Generator (FMG) with PostGIS backend and FastAPI REST interface.

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Configure environment:
Copy `.env.example` to `.env` and update database credentials.

3. Start PostgreSQL with PostGIS extension.

4. Run the API:
```bash
poetry run uvicorn py_fmg.api.main:app --reload
```

## Development

- Run tests: `poetry run pytest`
- Format code: `poetry run black . && poetry run isort .`
- Type check: `poetry run mypy py_fmg`
- Lint: `poetry run ruff check py_fmg`