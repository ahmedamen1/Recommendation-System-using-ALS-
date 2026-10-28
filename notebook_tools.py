"""Build, validate, or execute the notebook from its readable cell source."""

import argparse
import ast
import os
from pathlib import Path

import nbformat
import pycodestyle
from nbclient import NotebookClient

SOURCE = Path("train.py")
NOTEBOOK = Path("movie_recommender.ipynb")


def build_notebook():
    cells = []
    for block in SOURCE.read_text(encoding="utf-8").split("# %%")[1:]:
        header, _, body = block.partition("\n")
        if "[markdown]" in header:
            text = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in body.splitlines()
            ).strip()
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body.strip()))
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook.metadata.language_info = {"name": "python"}
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    return notebook


def check_notebook(notebook):
    nbformat.validate(notebook)
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell_{index}")
            count += 1
    style = pycodestyle.StyleGuide()
    result = style.check_files([str(SOURCE), __file__, "test_train.py"])
    if result.total_errors:
        raise SystemExit(f"PEP 8 errors: {result.total_errors}")
    print(f"Notebook valid: {count} Python cells; PEP 8 check passed.")


def execute_notebook(notebook):
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    local_jupyter = Path(".venv/share/jupyter").resolve()
    os.environ["JUPYTER_PATH"] = str(local_jupyter)
    os.environ["IPYTHONDIR"] = str(Path(".review/ipython").resolve())
    os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(".review/runtime").resolve())

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            first_line = cell.source.splitlines()[0]
            print(f"Executing cell {cell_index}: {first_line}", flush=True)

    client = NotebookClient(
        notebook, timeout=3600, kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
        on_cell_start=progress,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, NOTEBOOK)
    print(f"Executed and saved {NOTEBOOK}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build_notebook()
    check_notebook(notebook)
    if args.execute:
        execute_notebook(notebook)
