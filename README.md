# Movie Recommender

A movie recommendation project using ALS and MovieLens 32M. Search for movies,
rate a few, and get recommendations from the saved model.

## Run

Use Python 3.12. From this folder, run:

```sh
python -m pip install -r dashboard/requirements.txt
python dashboard/server.py
```

Open http://127.0.0.1:8000. On Windows, you can also use `start_dashboard.bat`.
The model is included in `inference_bundle/`. No training is needed to use it.

## Train

Open `movie_recommender.ipynb` in Jupyter or Google Colab and run the cells in order.
For a quick test, set `DATASET="ml-latest-small"` and `MAX_SWEEPS=2` in setup.

To run the training code locally:

```sh
python -m pip install -r requirements.txt
python train.py
```

## Paper

The paper is [aamin_AMLAS_draft.pdf](report/aamin_AMLAS_draft.pdf).
The repository is [Recommendation System using ALS](https://github.com/ahmedamen1/Recommendation-System-using-ALS-).
Results are in `results/`.

The LaTeX entry point is `report/main.tex`. To rebuild, run
`latexmk -pdf main.tex` inside `report/`, then copy the generated `main.pdf`
to `aamin_AMLAS_draft.pdf`. Check placement with
`python verify_layout.py aamin_AMLAS_draft.pdf`.

## Upload to GitHub

Follow [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md) for the prepared upload folder,
Windows PowerShell commands, and the files to include.

Downloaded data, caches, local archives, and large checkpoints are ignored by Git.
