# %% [markdown]
# # MovieLens 32M: Practicals 1?5


# %% [markdown]
# ## Setup


# %%
# Install dependencies before importing numerical libraries.
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy>=1.26", "pandas>=2.2", "scipy>=1.12", "numba>=0.61",
        "matplotlib>=3.8", "threadpoolctl>=3.5", "tqdm", "psutil",
    ])
print("Running in Colab:", IN_COLAB)


# %%
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from IPython.display import display  # noqa: E402
from numba import njit, prange  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
import psutil  # noqa: E402


# %% [markdown]
# ### Settings


# %%
DATASET = os.environ.get("MOVIELENS_DATASET", "ml-32m")
SEED = 42
CHUNK_SIZE = 500_000
MAX_SWEEPS = int(os.environ.get("ALS_SWEEPS", "20"))
NUM_THREADS = min(8, numba.config.NUMBA_NUM_THREADS)
REFIT_ALL = True
ALS_SETTING = "als_k10"  # Fixed choice for Practical 5 and full-data refits.
USE_GENRES = True  # Fixed choice for demonstration recommendations.
FIGURE_FORMAT = "pdf"  # Vector output: choose "pdf" or "eps".

if DATASET not in {"ml-latest-small", "ml-32m"}:
    raise ValueError("Choose ml-latest-small or ml-32m.")
if FIGURE_FORMAT not in {"pdf", "eps"}:
    raise ValueError("FIGURE_FORMAT must be pdf or eps.")
if MAX_SWEEPS < 1:
    raise ValueError("ALS_SWEEPS must be positive.")


# %%
# Colab uses its local disk for fast CSV/sparse-array operations.
WORK_DIR = Path(os.environ.get(
    "MOVIELENS_WORK_DIR", "/content/movielens_practicals" if IN_COLAB else "."
))
data_dir = WORK_DIR / "data" / DATASET
cache_dir = WORK_DIR / "cache" / DATASET
output_dir = WORK_DIR / "artifacts" / DATASET / "train_test"
for folder in (data_dir, cache_dir, output_dir / "models"):
    folder.mkdir(parents=True, exist_ok=True)
numba.set_num_threads(NUM_THREADS)
plt.rcParams.update({"font.size": 10, "figure.dpi": 110,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
print("Dataset:", DATASET, "| Numba CPU threads:", NUM_THREADS)
print(f"Available RAM: {psutil.virtual_memory().available / 2**30:.1f} GiB")
print(f"Free disk: {shutil.disk_usage(WORK_DIR).free / 2**30:.1f} GiB")
print("Results folder:", output_dir.resolve())


# %%


def save_figure(fig, name):
    """Save a scalable vector figure for the report (PDF or EPS only)."""
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.{FIGURE_FORMAT}",
                format=FIGURE_FORMAT, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %%


def save_table(table, name):
    """Keep full numeric precision in CSV; round only the display."""
    table.to_csv(output_dir / f"{name}.csv", index=False)
    display(table.round(4))


# %% [markdown]
# ### Download MovieLens


# %%
url = f"https://files.grouplens.org/datasets/movielens/{DATASET}.zip"
archive_path = data_dir.parent / f"{DATASET}.zip"
required_files = ("ratings.csv", "movies.csv")
if not all((data_dir / name).exists() for name in required_files):
    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".zip.part")
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with temporary_path.open("wb") as destination, tqdm(
                total=total, unit="B", unit_scale=True, desc="Download"
            ) as progress:
                while block := response.read(1024 * 1024):
                    destination.write(block)
                    progress.update(len(block))
        temporary_path.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        for name in required_files:
            target = data_dir / name
            if target.exists():
                continue
            partial = target.with_suffix(".csv.part")
            print("Extracting", name, flush=True)
            with archive.open(f"{DATASET}/{name}") as source_file:
                with partial.open("wb") as destination:
                    shutil.copyfileobj(source_file, destination)
            partial.replace(target)
print("Data folder:", data_dir.resolve())


# %% [markdown]
# ## Practical 1: Data Loading and Indexing


# %%
movies = pd.read_csv(data_dir / "movies.csv").sort_values("movieId")
movies = movies.reset_index(drop=True)
movie_ids = movies["movieId"].to_numpy(dtype=np.int32)
if movies["movieId"].duplicated().any():
    raise ValueError("Movie IDs must be unique.")


# %%
# Read ratings in small chunks.
csv_options = {
    "usecols": ["userId", "movieId", "rating"],
    "dtype": {"userId": "int32", "movieId": "int32", "rating": "float32"},
    "chunksize": CHUNK_SIZE,
}
user_ids = np.empty(0, dtype=np.int32)
number_of_rows = 0
for chunk in tqdm(pd.read_csv(data_dir / "ratings.csv", **csv_options),
                  desc="Pass 1: count all ratings", unit="chunk"):
    user_ids = np.union1d(user_ids, chunk["userId"].unique())
    number_of_rows += len(chunk)
if number_of_rows == 0:
    raise ValueError("The ratings file is empty.")
if DATASET == "ml-32m" and number_of_rows != 32_000_204:
    raise ValueError("The ml-32m ratings file must contain 32,000,204 rows.")

print(f"Verified {number_of_rows:,} ratings in {DATASET}.")


# %%
# Allocate compact arrays on disk.
n_users, n_movies = len(user_ids), len(movie_ids)
arrays = {}
for name, dtype in (("users", "int32"), ("items", "int32"),
                    ("ratings", "float32"), ("split", "uint8")):
    arrays[name] = np.lib.format.open_memmap(
        cache_dir / f"{name}.npy", mode="w+", dtype=dtype,
        shape=(number_of_rows,),
    )


# %%
# Fill arrays using every CSV chunk.
rng = np.random.default_rng(SEED)
offset = 0
for chunk in tqdm(pd.read_csv(data_dir / "ratings.csv", **csv_options),
                  total=(number_of_rows + CHUNK_SIZE - 1) // CHUNK_SIZE,
                  desc="Pass 2: index and split", unit="chunk"):
    stop = offset + len(chunk)
    raw_items = chunk["movieId"].to_numpy()
    item_indices = np.searchsorted(movie_ids, raw_items)
    if (item_indices >= n_movies).any():
        raise ValueError("A rated movie is missing from movies.csv.")
    if not np.array_equal(movie_ids[item_indices], raw_items):
        raise ValueError("A rated movie is missing from movies.csv.")
    ratings = chunk["rating"].to_numpy()
    if not np.all(np.isfinite(ratings) & (ratings >= 0.5)
                  & (ratings <= 5.0)):
        raise ValueError("Ratings must be finite and within [0.5, 5].")
    arrays["users"][offset:stop] = np.searchsorted(
        user_ids, chunk["userId"].to_numpy()
    )
    arrays["items"][offset:stop] = item_indices
    arrays["ratings"][offset:stop] = ratings
    draws = rng.random(len(chunk))
    arrays["split"][offset:stop] = (draws >= 0.8).astype(np.uint8)
    offset = stop
for array in arrays.values():
    array.flush()
del chunk, ratings, draws, item_indices, raw_items


# %%
# Inspect the indexed data.
m_idx = arrays["users"]
n_idx = arrays["items"]
r = arrays["ratings"]
split = arrays["split"]
print(f"{n_users:,} users | {n_movies:,} catalog movies")
print(f"{number_of_rows:,} ratings; no row limit applied")


# %%


def make_sparse(mask):
    """Index the selected ratings by user and by movie."""
    by_user = csr_matrix(
        (r[mask], (m_idx[mask], n_idx[mask])),
        shape=(n_users, n_movies), dtype=np.float32,
    )
    if by_user.nnz != np.count_nonzero(mask):
        raise ValueError("Duplicate user/movie pairs need a data policy.")
    by_movie = by_user.T.tocsr()
    return by_user, by_movie


# %%


def heldout_rows(code):
    """Store held-out observations once, without a dense rating matrix."""
    mask = split == code
    return m_idx[mask], n_idx[mask], r[mask]


# %%
# Build the training and held-out data.
data_by_user_train, data_by_movie_train = make_sparse(split == 0)
test_rows = heldout_rows(1)
if min(data_by_user_train.nnz, len(test_rows[2])) == 0:
    raise ValueError("Each split must contain at least one observation.")


# %%
# Check split counts add up to the full dataset.
split_table = pd.DataFrame({
    "split": ["train", "test"],
    "ratings": [data_by_user_train.nnz, len(test_rows[2])],
})
assert split_table["ratings"].sum() == number_of_rows
save_table(split_table, "split_counts")
user_count = np.diff(data_by_user_train.indptr)
movie_count = np.diff(data_by_movie_train.indptr)
print("Users without training ratings:", np.count_nonzero(user_count == 0))
print("Movies without training ratings:", np.count_nonzero(movie_count == 0))


# %%
rating_counts = np.zeros(10, dtype=np.int64)
for start in range(0, len(r), CHUNK_SIZE):
    bins = (r[start:start + CHUNK_SIZE] * 2).astype(np.int32) - 1
    rating_counts += np.bincount(bins, minlength=10)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(np.arange(0.5, 5.1, 0.5), rating_counts, width=0.4)
ax.set(xlabel="Rating (stars)", ylabel="Number of ratings")
save_figure(fig, "rating_distribution")


# %%


def degree_distribution(degrees):
    """Count each degree once, instead of repeated list.count calls."""
    values, counts = np.unique(degrees, return_counts=True)
    positive = values > 0
    return values[positive], counts[positive] / len(degrees)


# %%
# Plot user and movie activity.
fig, ax = plt.subplots(figsize=(5.5, 3.2))
for indices, size, label in ((m_idx, n_users, "Users"),
                             (n_idx, n_movies, "Movies")):
    degrees = np.bincount(indices, minlength=size)
    values, probabilities = degree_distribution(degrees)
    ax.loglog(values, probabilities, ".", label=label, markersize=3)
ax.set(xlabel="Number of ratings", ylabel="Fraction of catalog entities")
ax.legend()
save_figure(fig, "degree_distribution")
del degrees, bins


# %% [markdown]
# ## Practical 2: Bias Model


# %%


@njit
def update_biases(indptr, indices, ratings, own_bias, other_bias, lam, gam):
    for row in range(len(own_bias)):
        start, stop = indptr[row], indptr[row + 1]
        total = 0.0
        for position in range(start, stop):
            total += ratings[position] - other_bias[indices[position]]
        own_bias[row] = lam * total / (lam * (stop - start) + gam)


# %%


@njit(parallel=True)
def squared_error_sparse(indptr, indices, ratings, u, v, b_u, b_i):
    total = 0.0
    for user in prange(len(b_u)):
        local_total = 0.0
        for position in range(indptr[user], indptr[user + 1]):
            item = indices[position]
            prediction = b_u[user] + b_i[item]
            for factor in range(u.shape[1]):
                prediction += u[user, factor] * v[item, factor]
            local_total += (ratings[position] - prediction) ** 2
        total += local_total
    return total


# %%


def squared_errors(rows, model):
    """Yield small batches; never allocate a ratings-by-factors array."""
    users, items, ratings = rows
    for start in range(0, len(ratings), 100_000):
        stop = start + 100_000
        user, item = users[start:stop], items[start:stop]
        prediction = model["b_u"][user] + model["b_i"][item]
        prediction += np.einsum(
            "ij,ij->i", model["u"][user], model["v"][item]
        )
        yield (ratings[start:stop] - prediction) ** 2


# %%


def rmse(rows, model):
    if len(rows[2]) == 0:
        return np.nan
    return np.sqrt(sum(errors.sum() for errors in squared_errors(rows, model))
                   / len(rows[2]))


# %%


def initialize_model(n_user, n_item, factors, seed):
    rng = np.random.default_rng(seed)
    return {
        "u": rng.normal(0.0, 0.1, (n_user, factors)),
        "v": rng.normal(0.0, 0.1, (n_item, factors)),
        "b_u": np.zeros(n_user), "b_i": np.zeros(n_item),
    }


# %%


def training_sse(data, model):
    return squared_error_sparse(
        data.indptr, data.indices, data.data,
        model["u"], model["v"], model["b_u"], model["b_i"],
    )


# %%


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(history["sweep"], history["loss"], "o-", markersize=3)
    axes[0].set(xlabel="ALS sweep", ylabel="Regularized objective")
    for column, label in (("train_rmse", "Train"),
                          ("test_rmse", "Test")):
        axes[1].plot(history["sweep"], history[column], label=label)
    axes[1].set(xlabel="ALS sweep", ylabel="RMSE (stars)")
    axes[1].legend()
    save_figure(fig, name)


# %%
# Initialize the bias model.
lam, gam, tau = 0.01, 0.001, 0.1
bias_model = initialize_model(n_users, n_movies, 0, SEED)
bias_records = []


# %%
# Train biases and monitor test.
bias_model = initialize_model(n_users, n_movies, 0, SEED)
bias_records = []
for sweep in range(1, MAX_SWEEPS + 1):
    for data, own, other in (
        (data_by_user_train, "b_u", "b_i"),
        (data_by_movie_train, "b_i", "b_u"),
    ):
        update_biases(data.indptr, data.indices, data.data,
                      bias_model[own], bias_model[other], lam, gam)
    sse = training_sse(data_by_user_train, bias_model)
    penalty = sum(np.sum(bias_model[key] ** 2) for key in ("b_u", "b_i"))
    score = rmse(test_rows, bias_model)
    bias_records.append({
        "sweep": sweep, "loss": 0.5 * (lam * sse + gam * penalty),
        "train_rmse": np.sqrt(sse / data_by_user_train.nnz),
        "test_rmse": score,
    })
    print(f"Bias sweep {sweep}/{MAX_SWEEPS}: test RMSE={score:.4f}",
          flush=True)


# %%
# Inspect the bias learning curves.
bias_history = pd.DataFrame(bias_records)
bias_history.to_csv(output_dir / "bias_history.csv", index=False)
assert np.all(np.diff(bias_history["loss"]) <= 1e-6)
plot_history(bias_history, "bias_learning_curves")
save_table(bias_history.iloc[[0, -1]], "bias_first_last")


# %% [markdown]
# ### Bias Update Example


# %%
demo_user_bias = 0.5 * (5.0 + 3.0) / (0.5 * 2 + 0.1)
demo_item_bias = 0.5 * (5.0 - demo_user_bias) / (0.5 + 0.1)
print("New user bias:", demo_user_bias)
print("New movie-0 bias:", demo_item_bias)
transposed_sse = squared_error_sparse(
    data_by_movie_train.indptr, data_by_movie_train.indices,
    data_by_movie_train.data, bias_model["v"], bias_model["u"],
    bias_model["b_i"], bias_model["b_u"],
)
assert np.isclose(transposed_sse, training_sse(
    data_by_user_train, bias_model
))
print("The two sparse views give the same squared error.")


# %% [markdown]
# ## Practical 3: Latent Factors


# %%


def update_side_reference(data, own, other, own_bias, other_bias,
                          lam, gam, tau, prior):
    """Readable version of the original two-loop update."""
    factors = own.shape[1]
    for row in range(data.shape[0]):
        start, stop = data.indptr[row:row + 2]
        total = 0.0
        for position in range(start, stop):
            column = data.indices[position]
            total += (data.data[position] - other_bias[column]
                      - own[row] @ other[column])
        own_bias[row] = lam * total / (lam * (stop - start) + gam)
        matrix = tau * np.eye(factors)
        target = tau * prior[row].copy()
        for position in range(start, stop):
            column = data.indices[position]
            vector = other[column]
            residual = (data.data[position] - own_bias[row]
                        - other_bias[column])
            matrix += lam * np.outer(vector, vector)
            target += lam * vector * residual
        own[row] = np.linalg.solve(matrix, target)


# %% [markdown]
# ## Practical 4: Efficient ALS


# %%


def update_side_kernel(indptr, indices, ratings, own, other,
                       own_bias, other_bias, lam, gam, tau, prior):
    factors = own.shape[1]
    for row in prange(len(own_bias)):
        start, stop = indptr[row], indptr[row + 1]
        total = 0.0
        for position in range(start, stop):
            column = indices[position]
            interaction = 0.0
            for factor in range(factors):
                interaction += own[row, factor] * other[column, factor]
            total += ratings[position] - other_bias[column] - interaction
        own_bias[row] = lam * total / (lam * (stop - start) + gam)

        matrix = tau * np.eye(factors)
        target = tau * prior[row].copy()
        for position in range(start, stop):
            column = indices[position]
            residual = (ratings[position] - own_bias[row]
                        - other_bias[column])
            for first in range(factors):
                value = other[column, first]
                target[first] += lam * value * residual
                for second in range(first + 1):
                    matrix[first, second] += (
                        lam * value * other[column, second]
                    )
        for first in range(factors):
            for second in range(first):
                matrix[second, first] = matrix[first, second]
        own[row] = np.linalg.solve(matrix, target)


# %%
update_side_serial = njit(update_side_kernel)
update_side_parallel = njit(parallel=True)(update_side_kernel)


# %%


def update_side(data, own, other, own_bias, other_bias,
                lam, gam, tau, prior):
    update_side_parallel(
        data.indptr, data.indices, data.data, own, other,
        own_bias, other_bias, lam, gam, tau, prior,
    )


# %% [markdown]
# ### Correctness and Speed Benchmark


# %%
benchmark_data = data_by_user_train[:min(500, n_users)].copy()
if benchmark_data.nnz > 100_000:
    last_row = np.searchsorted(benchmark_data.indptr, 100_000, side="right")
    benchmark_data = benchmark_data[:max(1, last_row - 1)].copy()
bench_rng = np.random.default_rng(SEED)
bench_own = bench_rng.normal(0, 0.1, (benchmark_data.shape[0], 10))
bench_other = bench_rng.normal(0, 0.1, (n_movies, 10))
bench_bias = np.zeros(benchmark_data.shape[0])
bench_other_bias = np.zeros(n_movies)
bench_prior = np.zeros_like(bench_own)


# %%


def benchmark_update(method):
    own, bias = bench_own.copy(), bench_bias.copy()
    start = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        if method == "Python reference":
            update_side_reference(
                benchmark_data, own, bench_other, bias, bench_other_bias,
                lam, gam, tau, bench_prior,
            )
        else:
            function = (update_side_serial if method == "Numba serial"
                        else update_side_parallel)
            function(
                benchmark_data.indptr, benchmark_data.indices,
                benchmark_data.data, own, bench_other, bias,
                bench_other_bias, lam, gam, tau, bench_prior,
            )
    return time.perf_counter() - start, own, bias


# %%
# Run the bounded correctness benchmark.
reference_seconds, reference_u, reference_b = benchmark_update(
    "Python reference"
)
benchmark_records = []
for method in ("Python reference", "Numba serial", "Numba parallel"):
    first_seconds, own, bias = benchmark_update(method)
    np.testing.assert_allclose(own, reference_u, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(bias, reference_b, rtol=1e-9, atol=1e-10)
    times = [benchmark_update(method)[0] for _ in range(3)]
    benchmark_records.append({
        "method": method, "median_seconds": float(np.median(times)),
        "first_call_seconds": first_seconds,
        "ratings": benchmark_data.nnz, "factors": 10,
    })
benchmark_table = pd.DataFrame(benchmark_records)
save_table(benchmark_table, "update_benchmark")


# %% [markdown]
# ### Training Loop


# %%


def fit_als(by_user, by_movie, test=None, factors=10, lam=0.01,
            gam=0.001, tau=0.1, sweeps=20, seed=42,
            features=None, eta=0.1):
    if min(lam, gam, tau, eta) <= 0 or factors < 1 or sweeps < 1:
        raise ValueError("Use positive penalties, factors, and sweeps.")
    model = initialize_model(*by_user.shape, factors, seed)
    warm_items = np.diff(by_movie.indptr) > 0
    zero_users = np.zeros_like(model["u"])
    prior = np.zeros_like(model["v"])
    if features is not None:
        model["features"] = np.zeros((features.shape[1], factors))
        warm_features = features[warm_items]
        feature_matrix = (tau * (warm_features.T @ warm_features)
                          + eta * np.eye(features.shape[1]))
    records = []
    start_time = time.perf_counter()
    previous_loss = np.inf
    with threadpool_limits(limits=1, user_api="blas"):
        for sweep in range(1, sweeps + 1):
            update_side(by_user, model["u"], model["v"],
                        model["b_u"], model["b_i"],
                        lam, gam, tau, zero_users)
            update_side(by_movie, model["v"], model["u"],
                        model["b_i"], model["b_u"],
                        lam, gam, tau, prior)
            if features is not None:
                model["features"] = np.linalg.solve(
                    feature_matrix,
                    tau * warm_features.T @ model["v"][warm_items],
                )
                prior = features @ model["features"]
                model["v"][~warm_items] = prior[~warm_items]
            sse = training_sse(by_user, model)
            bias_penalty = sum(np.sum(model[key] ** 2)
                               for key in ("b_u", "b_i"))
            factor_penalty = np.sum(model["u"] ** 2)
            factor_penalty += np.sum(
                (model["v"][warm_items] - prior[warm_items]) ** 2
            )
            feature_penalty = (np.sum(model["features"] ** 2)
                               if features is not None else 0.0)
            loss = 0.5 * (lam * sse + gam * bias_penalty
                          + tau * factor_penalty + eta * feature_penalty)
            if not np.isfinite(loss):
                raise ArithmeticError("The training loss is not finite.")
            if loss > previous_loss + 1e-7 * max(1.0, abs(previous_loss)):
                raise ArithmeticError("ALS increased the objective.")
            previous_loss = loss
            score = rmse(test, model) if test is not None else np.nan
            records.append({
                "sweep": sweep, "loss": loss,
                "train_rmse": np.sqrt(sse / by_user.nnz),
                "test_rmse": score,
                "elapsed_seconds": time.perf_counter() - start_time,
            })
            print(f"K={factors:2d} sweep={sweep:2d}/{sweeps} "
                  f"train={records[-1]['train_rmse']:.4f} "
                  f"test={records[-1]['test_rmse']:.4f} "
                  f"elapsed={time.perf_counter() - start_time:.0f}s",
                  flush=True)
    return model, pd.DataFrame(records), sweeps


# %% [markdown]
# ### ALS Experiments


# %%


def run_experiment(name, factors=10, lam=0.01, tau=0.1):
    """Train one setting and save it immediately; return a small summary."""
    print("Starting", name, "on", data_by_user_train.nnz, "training ratings",
          flush=True)
    model, history, completed_sweeps = fit_als(
        data_by_user_train, data_by_movie_train, test_rows,
        factors=factors, lam=lam, gam=gam, tau=tau,
        sweeps=MAX_SWEEPS, seed=SEED,
    )
    history.to_csv(output_dir / f"{name}_history.csv", index=False)
    np.savez(output_dir / "models" / f"{name}.npz", **model)
    final_row = history.iloc[-1]
    record = {
        "name": name, "factors": factors, "lam": lam, "tau": tau,
        "gamma": gam, "completed_sweeps": completed_sweeps,
        "test_rmse": final_row["test_rmse"],
        "train_rmse": final_row["train_rmse"],
        "seconds": history["elapsed_seconds"].iloc[-1],
    }
    return record, history


# %%
# Initialize once. Rerunning one experiment replaces its dictionary entry.
experiments = {}


# %% [markdown]
# ### Baseline: K = 10


# %%
experiments["als_k10"] = run_experiment("als_k10", factors=10)


# %%
plot_history(experiments["als_k10"][1], "practical3_learning_curves")


# %% [markdown]
# ### Capacity: K = 20


# %%
experiments["als_k20"] = run_experiment("als_k20", factors=20)


# %% [markdown]
# ### Weak Regularization


# %%
experiments["als_weak_tau"] = run_experiment("als_weak_tau", tau=0.01)


# %% [markdown]
# ### Strong Regularization


# %%
experiments["als_strong_tau"] = run_experiment("als_strong_tau", tau=1.0)


# %% [markdown]
# ### Higher Rating-Error Weight


# %%
experiments["als_larger_lambda"] = run_experiment(
    "als_larger_lambda", lam=0.1
)


# %% [markdown]
# ### Model Comparison


# %%
comparison = pd.DataFrame([record for record, _ in experiments.values()])
histories = {name: history for name, (_, history) in experiments.items()}
save_table(comparison, "als_test_comparison")
# ALS_SETTING is fixed in setup, not chosen using the test errors.
configured_record = experiments[ALS_SETTING][0]
als_setting = {key: configured_record[key]
               for key in ("name", "factors", "lam", "tau")}
als_sweeps = MAX_SWEEPS
with np.load(output_dir / "models" / f"{als_setting['name']}.npz") as saved:
    als_model = {key: saved[key] for key in saved.files}


# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for name in ("als_k10", "als_k20"):
    history = histories[name]
    axes[0].plot(history["sweep"], history["train_rmse"], label=name)
    axes[1].plot(history["sweep"], history["test_rmse"], label=name)
axes[0].set(xlabel="ALS sweep", ylabel="Training RMSE")
axes[1].set(xlabel="ALS sweep", ylabel="Test RMSE")
for ax in axes:
    ax.legend()
save_figure(fig, "capacity_comparison")
print("Predefined ALS settings:", als_setting)
print("Fixed training sweeps:", als_sweeps)


# %% [markdown]
# ### Error by User Activity


# %%


def group_errors(rows, model, groups):
    """Aggregate squared errors by group without retaining predictions."""
    totals = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    offset = 0
    for errors in squared_errors(rows, model):
        group = groups[offset:offset + len(errors)]
        totals += np.bincount(group, weights=errors, minlength=4)
        counts += np.bincount(group, minlength=4)
        offset += len(errors)
    values = np.full(4, np.nan)
    np.divide(totals, counts, out=values, where=counts > 0)
    return counts, np.sqrt(values)


# %%
# Compare test error by user activity.
test_groups = np.digitize(
    user_count[test_rows[0]], bins=[1, 20, 100]
)
group_records = []
for name in ("als_k10", "als_k20"):
    with np.load(output_dir / "models" / f"{name}.npz") as saved:
        group_model = {key: saved[key] for key in saved.files}
    counts, errors = group_errors(
        test_rows, group_model, test_groups
    )
    for label, count, error in zip(
        ["0", "1-19", "20-99", "100+"], counts, errors
    ):
        group_records.append({"model": name, "user_train_ratings": label,
                              "test_ratings": count, "rmse": error})
save_table(pd.DataFrame(group_records), "user_activity_errors")


# %% [markdown]
# ## Practical 5: Genre Features


# %%
genre_lists = movies["genres"].fillna("(no genres listed)").str.split("|")
genre_names = sorted({genre for genres in genre_lists for genre in genres
                      if genre != "(no genres listed)"})
genre_to_index = {genre: index for index, genre in enumerate(genre_names)}
genre_features = np.zeros((n_movies, len(genre_names)))
for item, genres in enumerate(genre_lists):
    columns = [genre_to_index[genre] for genre in genres
               if genre in genre_to_index]
    if columns:
        genre_features[item, columns] = 1.0 / np.sqrt(len(columns))
print("Genre feature matrix:", genre_features.shape)


# %%
# Train Practical 5 with the predefined ALS settings.
chosen_parameters = {key: value for key, value in als_setting.items()
                     if key != "name"}
genre_model, genre_history, genre_sweep = fit_als(
    data_by_user_train, data_by_movie_train, test_rows,
    **chosen_parameters, gam=gam, sweeps=MAX_SWEEPS, seed=SEED,
    features=genre_features, eta=0.1,
)


# %%
# Save and plot the genre model.
np.savez(output_dir / "models" / "genre_model.npz", **genre_model,
         user_ids=user_ids, movie_ids=movie_ids, genre_names=genre_names)
genre_history.to_csv(output_dir / "genre_history.csv", index=False)
plot_history(genre_history, "genre_learning_curves")


# %% [markdown]
# ### Test Results


# %%
training_mean = float(data_by_user_train.data.mean(dtype=np.float64))
mean_test_sse = 0.0
for start in range(0, len(test_rows[2]), CHUNK_SIZE):
    values = test_rows[2][start:start + CHUNK_SIZE].astype(np.float64)
    mean_test_sse += np.sum((values - training_mean) ** 2)
final_records = [{"model": "Training mean", "completed_sweeps": 0,
                  "test_rmse": np.sqrt(mean_test_sse / len(test_rows[2]))}]
frozen_models = [("Biases only", bias_model, MAX_SWEEPS),
                 ("Configured ALS", als_model, als_sweeps),
                 ("Genre ALS", genre_model, genre_sweep)]
for name, model, sweep in frozen_models:
    final_records.append({"model": name, "completed_sweeps": sweep,
                          "test_rmse": rmse(test_rows, model)})
final_results = pd.DataFrame(final_records)
save_table(final_results, "final_test_results")


# %%
# Export the final results as a LaTeX table.
latex_lines = [r"\begin{tabular}{lrr}", r"\toprule",
               r"Model & Sweep & Test RMSE \\", r"\midrule"]
for row in final_results.itertuples(index=False):
    latex_lines.append(
        f"{row.model} & {row.completed_sweeps} & {row.test_rmse:.4f} " + r"\\"
    )
latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
(output_dir / "final_test_results.tex").write_text(
    "\n".join(latex_lines), encoding="utf-8"
)


# %%
# Inspect natural cold-start performance.
test_cold_groups = (
    (user_count[test_rows[0]] == 0).astype(np.int32) * 2
    + (movie_count[test_rows[1]] == 0).astype(np.int32)
)
cold_records = []
for name, model, _ in frozen_models[1:]:
    counts, errors = group_errors(test_rows, model, test_cold_groups)
    for label, count, error in zip(
        ["warm user / warm item", "warm user / cold item",
         "cold user / warm item", "cold user / cold item"], counts, errors
    ):
        cold_records.append({"model": name, "group": label,
                             "test_ratings": count, "rmse": error})
save_table(pd.DataFrame(cold_records), "natural_cold_start_errors")


# %% [markdown]
# ### Cold-Start Experiment


# %%
candidates = np.flatnonzero(movie_count >= 20)
if len(candidates) >= 2:
    rng = np.random.default_rng(SEED + 1)
    cold_items = rng.choice(candidates, max(1, len(candidates) // 20),
                            replace=False)
    cold_mask = np.zeros(n_movies, dtype=bool)
    cold_mask[cold_items] = True
    cold_train, cold_train_items = make_sparse(
        (split == 0) & ~cold_mask[n_idx]
    )
    warm_user = np.diff(cold_train.indptr) > 0
    test_mask = cold_mask[test_rows[1]] & warm_user[test_rows[0]]
    cold_test = tuple(values[test_mask] for values in test_rows)
    cold_base, _, _ = fit_als(
        cold_train, cold_train_items, None, **chosen_parameters,
        gam=gam, sweeps=als_sweeps, seed=SEED,
    )
    cold_genre, _, _ = fit_als(
        cold_train, cold_train_items, None, **chosen_parameters,
        gam=gam, sweeps=genre_sweep, seed=SEED,
        features=genre_features, eta=0.1,
    )
    assert np.all(np.diff(cold_train_items.indptr)[cold_items] == 0)
    cold_results = pd.DataFrame({
        "model": ["ALS without genres", "Genre ALS"],
        "held_out_movies": len(cold_items),
        "test_ratings": len(cold_test[2]),
        "test_rmse": [rmse(cold_test, cold_base), rmse(cold_test, cold_genre)],
    })
    save_table(cold_results, "controlled_cold_start")
    np.save(output_dir / "cold_movie_ids.npy", movie_ids[cold_items])
    del cold_train, cold_train_items, cold_base, cold_genre
else:
    print("Too few supported movies for the controlled cold-start study.")


# %% [markdown]
# ### 2D Embeddings


# %%
embedding_models = {}
for label, features in (("movies_2d", None),
                        ("genres_2d", genre_features)):
    embedding, history, _ = fit_als(
        data_by_user_train, data_by_movie_train, test_rows,
        factors=2, lam=0.01, gam=gam, tau=0.1,
        sweeps=MAX_SWEEPS, seed=SEED, features=features,
    )
    embedding_models[label] = embedding
    history.to_csv(output_dir / f"{label}_history.csv", index=False)
    fig, (ax, key_ax) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    if features is None:
        selected = np.argsort(movie_count)[-12:]
        points = embedding["v"][selected]
        labels = movies.iloc[selected]["title"].tolist()
    else:
        points = embedding["features"]
        labels = genre_names
    ax.scatter(points[:, 0], points[:, 1], s=20)
    key_labels = []
    for number, (point, label_text) in enumerate(zip(points, labels), 1):
        offset = (5, 5) if number % 2 else (5, -10)
        ax.annotate(str(number), point, xytext=offset,
                    textcoords="offset points", fontsize=8)
        key_labels.append(textwrap.fill(f"{number}. {label_text}", width=38,
                                        subsequent_indent="    "))
    key_ax.axis("off")
    key_ax.text(0, 0.98, "\n".join(key_labels), va="top", fontsize=8,
                transform=key_ax.transAxes, linespacing=1.3)
    ax.margins(0.12)
    ax.set(xlabel="Latent coordinate 1", ylabel="Latent coordinate 2")
    save_figure(fig, label)
    coordinates = pd.DataFrame({
        "label": labels, "coordinate_1": points[:, 0],
        "coordinate_2": points[:, 1],
    })
    coordinates.to_csv(output_dir / f"{label}_coordinates.csv", index=False)


# %% [markdown]
# ## Full-Data Refits


# %%
# Build the full sparse matrices once for both models.
if REFIT_ALL:
    full_by_user, full_by_movie = make_sparse(np.ones(len(r), dtype=bool))
    assert full_by_user.nnz == number_of_rows
    assert full_by_movie.nnz == number_of_rows
    print(f"Both refits will use {full_by_user.nnz:,} ratings.")


# %% [markdown]
# ### Practical 4: ALS


# %%
if REFIT_ALL:
    full_als, full_als_history, _ = fit_als(
        full_by_user, full_by_movie, None, **chosen_parameters,
        gam=gam, sweeps=als_sweeps, seed=SEED,
    )
    np.savez(output_dir / "models" / "full_als.npz", **full_als,
             user_ids=user_ids, movie_ids=movie_ids)
    full_als_history.to_csv(output_dir / "full_als_history.csv", index=False)
    print("Saved full-data Practical 4 model.")


# %% [markdown]
# ### Practical 5: Genre ALS


# %%
if REFIT_ALL:
    full_genre, full_genre_history, _ = fit_als(
        full_by_user, full_by_movie, None, **chosen_parameters,
        gam=gam, sweeps=genre_sweep, seed=SEED,
        features=genre_features, eta=0.1,
    )
    np.savez(output_dir / "models" / "full_genre.npz", **full_genre,
             user_ids=user_ids, movie_ids=movie_ids, genre_names=genre_names)
    full_genre_history.to_csv(
        output_dir / "full_genre_history.csv", index=False
    )
    print("Saved full-data Practical 5 model.")


# %%
# Use the recommendation model specified in setup.
use_genres = USE_GENRES
recommendation_sweeps = genre_sweep if use_genres else als_sweeps
recommendation_model = genre_model if use_genres else als_model
recommendation_count = movie_count
if REFIT_ALL:
    recommendation_model = full_genre if use_genres else full_als
    recommendation_count = np.diff(full_by_movie.indptr).copy()
    full_history = full_genre_history if use_genres else full_als_history
    full_history.to_csv(output_dir / "full_refit_history.csv", index=False)
np.savez(output_dir / "models" / "recommendation_model.npz",
         **recommendation_model, user_ids=user_ids, movie_ids=movie_ids)


# %% [markdown]
# ### Dashboard Model Export


# %%


def export_inference_bundle(folder, model, catalog, ids, counts, metadata):
    """Write a portable, non-pickle bundle for new-user recommendations."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not np.array_equal(catalog["movieId"].to_numpy(), ids):
        raise ValueError("Movie catalog order must match the model IDs.")
    if model["v"].shape[0] != len(ids) or len(counts) != len(ids):
        raise ValueError("Model, catalog, and counts must have equal lengths.")
    np.savez_compressed(
        folder / "model.npz", item_factors=model["v"],
        item_bias=model["b_i"], movie_ids=ids, movie_counts=counts,
    )
    catalog[["movieId", "title", "genres"]].to_csv(
        folder / "movies.csv", index=False
    )
    information = {
        **metadata, "schema_version": 1,
        "factors": int(model["v"].shape[1]), "movies": len(ids),
        "prediction": "user_bias + item_bias + user_vector @ item_vector",
    }
    (folder / "metadata.json").write_text(
        json.dumps(information, indent=2), encoding="utf-8"
    )
    return shutil.make_archive(str(folder), "zip", folder)


# %%
inference_zip = export_inference_bundle(
    output_dir / "inference_bundle", recommendation_model, movies,
    movie_ids, recommendation_count,
    {
        "dataset": DATASET, "dataset_ratings": number_of_rows,
        "training_ratings": int(recommendation_count.sum()),
        "model_type": "Genre ALS" if use_genres else "ALS",
        "full_refit": REFIT_ALL, "sweeps": recommendation_sweeps,
        "lam": chosen_parameters["lam"], "gam": gam,
        "tau": chosen_parameters["tau"], "seed": SEED,
    },
)
print("Local inference download:", inference_zip)
print("Contains model.npz, movies.csv, metadata.json; no raw ratings needed.")


# %% [markdown]
# ### New-User Recommendations


# %%


def recommend_for_profile(profile, model, counts, bias_weight=1.0,
                          minimum_ratings=20, top_n=10):
    """Fit a new user to {original_movie_id: rating} and rank candidates."""
    if not profile:
        raise ValueError("Give at least one movie rating.")
    raw_ids = np.array(list(profile), dtype=np.int32)
    items = np.searchsorted(movie_ids, raw_ids)
    if (items >= n_movies).any():
        raise ValueError("The profile contains an unknown movie ID.")
    if not np.array_equal(movie_ids[items], raw_ids):
        raise ValueError("The profile contains an unknown movie ID.")
    values = np.array(list(profile.values()), dtype=np.float64)
    if not np.all(np.isfinite(values) & (values >= 0.5) & (values <= 5.0)):
        raise ValueError("Profile ratings must be in [0.5, 5.0].")
    vectors = model["v"][items]
    design = np.column_stack([np.ones(len(items)), vectors])
    penalties = np.diag([gam] + [chosen_parameters["tau"]] * vectors.shape[1])
    weight = chosen_parameters["lam"]
    coefficients = np.linalg.solve(
        weight * design.T @ design + penalties,
        weight * design.T @ (values - model["b_i"][items]),
    )
    interaction = model["v"] @ coefficients[1:]
    predictions = coefficients[0] + interaction + model["b_i"]
    scores = interaction + bias_weight * model["b_i"]
    eligible = counts >= minimum_ratings
    eligible[items] = False
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(-scores[ranked], kind="stable")[:top_n]]
    result = movies.iloc[ranked][["movieId", "title", "genres"]].copy()
    result["ranking_score"] = scores[ranked]
    result["predicted_rating"] = predictions[ranked]
    result["observed_ratings"] = counts[ranked]
    result["bias_weight"] = bias_weight
    result["minimum_ratings"] = minimum_ratings
    return result


# %%
# Choose a movie for the dummy user.
matches = movies.index[movies["title"].str.contains(
    "Lord of the Rings: The Fellowship", case=False, regex=False
)]
seed_item = int(matches[0]) if len(matches) else int(np.argmax(movie_count))
profile = {int(movie_ids[seed_item]): 5.0}
print("Dummy user likes:", movies.iloc[seed_item]["title"])


# %%
# Show recommendations with different ranking weights.
for bias_weight in (1.0, 0.05):
    for threshold in (1, 100):
        recommendations = recommend_for_profile(
            profile, recommendation_model, recommendation_count,
            bias_weight=bias_weight, minimum_ratings=threshold,
        )
        name = f"recommendations_{bias_weight}_{threshold}"
        save_table(recommendations, name)


# %% [markdown]
# ### Polarizing Movies


# %%
counts = np.zeros(n_movies, dtype=np.int64)
rating_sum = np.zeros(n_movies)
rating_square_sum = np.zeros(n_movies)
for start in range(0, len(r), CHUNK_SIZE):
    items = n_idx[start:start + CHUNK_SIZE]
    ratings = r[start:start + CHUNK_SIZE].astype(np.float64)
    counts += np.bincount(items, minlength=n_movies)
    rating_sum += np.bincount(items, weights=ratings, minlength=n_movies)
    rating_square_sum += np.bincount(
        items, weights=ratings ** 2, minlength=n_movies
    )


# %%
# Rank polarizing movies.
means = rating_sum / np.maximum(counts, 1)
variances = np.maximum(
    0, rating_square_sum / np.maximum(counts, 1) - means ** 2
)
polarizing = movies.copy()
polarizing["ratings"] = counts
polarizing["rating_std"] = np.sqrt(variances)
polarizing["vector_norm"] = np.linalg.norm(recommendation_model["v"], axis=1)
polarizing = polarizing[polarizing["ratings"] >= 100]
save_table(polarizing.nlargest(15, "vector_norm"), "polarizing_movies")


# %% [markdown]
# ## Run Summary


# %%


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# %%
# Save dataset and experiment provenance.
manifest = {
    "dataset": DATASET, "ratings": number_of_rows,
    "users": n_users, "catalog_movies": n_movies,
    "seed": SEED, "split": "seeded random rating split 80/20",
    "split_counts": split_table.to_dict(orient="records"),
    "maximum_sweeps": MAX_SWEEPS, "configured_als": als_setting,
    "als_sweeps": als_sweeps, "genre_sweep": genre_sweep,
    "gamma": gam, "eta": 0.1, "numba_threads": NUM_THREADS,
    "refit_all": REFIT_ALL, "refit_genres": bool(use_genres),
    "full_refit_models": ["ALS", "Genre ALS"] if REFIT_ALL else [],
    "full_refit_ratings": number_of_rows if REFIT_ALL else 0,
    "source_url": url, "python": platform.python_version(),
    "platform": platform.platform(), "processor": platform.processor(),
    "versions": {name: version(name) for name in
                 ("numpy", "pandas", "scipy", "numba", "matplotlib")},
    "sha256": {name: file_sha256(data_dir / name)
               for name in ("ratings.csv", "movies.csv")},
    "test_used_for_selection": False,
    "test_monitored_each_sweep": True, "figure_format": FIGURE_FORMAT,
    "checkpoint_policy": "final sweep",
}
(output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)


# %%
# Report storage and results location.
memory_bytes = sum(array.nbytes for array in arrays.values())
for sparse_data in (data_by_user_train, data_by_movie_train):
    memory_bytes += sum(
        array.nbytes for array in
        (sparse_data.data, sparse_data.indices, sparse_data.indptr)
    )
memory_bytes += sum(array.nbytes for array in test_rows)
print(f"Core data-array storage: {memory_bytes / 1024 ** 3:.3f} GiB")
print("This includes disk-backed arrays; it is not peak process RAM.")
print("Artifacts saved to:", output_dir.resolve())
print("Final test results describe evaluation models, not the full refit.")


# %% [markdown]
# ## Download Results


# %%
# Download the compact inference bundle automatically at the end in Colab.
DOWNLOAD_INFERENCE = True
if IN_COLAB and DOWNLOAD_INFERENCE:
    from google.colab import files
    files.download(inference_zip)
else:
    print("Inference bundle ready:", inference_zip)


# %%
results_zip = shutil.make_archive(
    str(output_dir.parent / f"{DATASET}_train_test_results"), "zip", output_dir
)
print("Results ZIP:", results_zip)


# %%
# Optional: change to True and run this cell to download the results.
DOWNLOAD_RESULTS = False
if IN_COLAB and DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(results_zip)


# %%
# Optional: change to True and run this cell to save the ZIP to Google Drive.
SAVE_TO_DRIVE = False
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    drive_folder = Path("/content/drive/MyDrive/movielens_practicals")
    drive_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_zip, drive_folder / Path(results_zip).name)
    print("Saved:", drive_folder / Path(results_zip).name)
