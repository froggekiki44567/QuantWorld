[project]
name = "quantlab"
version = "0.1.0"
description = "Quantitative research platform - Module 1: data layer"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27", "polars>=1.0", "duckdb>=1.0", "pyarrow>=16"]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.5", "mypy>=1.10"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
