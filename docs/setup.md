# Setup

## Python Environment

```bash
cd /home/rlab/raghavendra/april_chal
source .venv/bin/activate
```

The virtual environment uses Python 3.12 and includes the challenge libraries.
The Jupyter kernel is registered as `Python (april_chal)`.

## Notebook

```bash
jupyter lab
```

Open `04_2026/starter_notebook.ipynb` and select the `Python (april_chal)`
kernel.

## Result Book

Preview the MkDocs book locally:

```bash
source .venv/bin/activate
mkdocs serve --dev-addr 127.0.0.1:8000
```

Build the static site:

```bash
source .venv/bin/activate
mkdocs build --strict
```

## Optional Public Preview

Use the helper only when a public ngrok preview is needed:

```bash
source ~/.bashrc
source .venv/bin/activate
python scripts/serve_docs_ngrok.py
```

The helper reads `NGROK_TOKEN` from the environment. Do not commit the token.

