---
description: How to correctly run commands in this project using .venv and uv
---

To ensure all dependencies are correctly handled and the project environment is isolated, always follow these steps:

1. **Activate the virtual environment**:
   Before running any python scripts or installation commands, ensure the `.venv` is activated.
   - On Windows: `.\.venv\Scripts\activate`
   - On Unix/macOS: `source .venv/bin/activate`

2. **Use `uv` for dependency management**:
   This project uses `uv` for fast and reliable dependency management.
   - To add a new dependency: `uv add <package_name>`
   - To install all dependencies: `uv sync`

3. **Running Scripts**:
   Always run scripts while the virtual environment is active.
   // turbo
   `python scripts/<script_name>.py`
