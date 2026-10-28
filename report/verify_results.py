"""Independently reproduce the uploaded run's metrics from official ratings."""
import gc
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / ".build" / "32m_audit"
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((RESULTS / "run_manifest.json").read_text())
names = ["als_k10", "als_k20", "als_weak_tau", "als_strong_tau",
         "als_larger_lambda", "genre_model", "full_als", "full_genre"]
models = {}
for name in names:
    with np.load(RESULTS / "models" / f"{name}.npz", allow_pickle=False) as data:
        models[name] = {k: data[k] for k in ("u", "v", "b_u", "b_i", "features") if k in data}
        for value in models[name].values():
            assert np.isfinite(value).all(), name
with np.load(RESULTS / "models" / "recommendation_model.npz") as data:
    user_ids, movie_ids = data["user_ids"], data["movie_ids"]
    for key in models["full_genre"]:
        np.testing.assert_array_equal(data[key], models["full_genre"][key])

bundles = [ROOT / "inference_bundle"]
hashes = {}
for folder in bundles:
    hashes[str(folder.relative_to(ROOT))] = {
        f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
        for f in ("metadata.json", "movies.csv", "model.npz")}
with np.load(bundles[0] / "model.npz") as data:
    np.testing.assert_array_equal(data["item_factors"], models["full_genre"]["v"])
    np.testing.assert_array_equal(data["item_bias"], models["full_genre"]["b_i"])
    np.testing.assert_array_equal(data["movie_ids"], movie_ids)
    saved_counts = data["movie_counts"]

n_users, n_movies = len(user_ids), len(movie_ids)
total_rows = manifest["ratings"]
assert (total_rows, n_users, n_movies) == (32000204, 200948, 87585)
user_count = np.zeros(n_users, np.int64)
movie_count = np.zeros(n_movies, np.int64)
train_users = np.zeros(n_users, np.int64)
train_movies = np.zeros(n_movies, np.int64)
test_users = np.zeros(n_users, np.int64)
test_movies = np.zeros(n_movies, np.int64)
rating_counts = np.zeros(10, np.int64)
rating_sum = np.zeros(n_movies)
rating_squares = np.zeros(n_movies)
errors = {name: np.zeros(2) for name in names}
user_errors = {name: np.zeros(n_users) for name in names[:6]}
movie_errors = {name: np.zeros(n_movies) for name in names[:6]}
cached = {name: np.lib.format.open_memmap(OUT / f"raw_{name}.npy", mode="w+", dtype=dtype, shape=(total_rows,))
          for name, dtype in (("users", "int32"), ("items", "int32"), ("ratings", "float32"), ("test", "bool"))}

class HashReader(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
    def readable(self): return True
    def readinto(self, target):
        data = self.source.read(len(target))
        self.sha256.update(data); self.md5.update(data)
        target[:len(data)] = data
        return len(data)

rng = np.random.default_rng(42)
offset = 0
train_sum = test_sum = test_square_sum = 0.0
with zipfile.ZipFile(ROOT / "data" / "ml-32m.zip") as archive:
    movies_bytes = archive.read("ml-32m/movies.csv")
    assert hashlib.md5(movies_bytes).hexdigest() == "0df90835c19151f9d819d0822e190797"
    assert hashlib.sha256(movies_bytes).hexdigest() == manifest["sha256"]["movies.csv"]
    catalog = pd.read_csv(io.BytesIO(movies_bytes)).sort_values("movieId").reset_index(drop=True)
    pd.testing.assert_frame_equal(catalog, pd.read_csv(bundles[0] / "movies.csv"))
    with archive.open("ml-32m/ratings.csv") as raw:
        reader = HashReader(raw)
        stream = io.BufferedReader(reader, buffer_size=1024*1024)
        for chunk in pd.read_csv(stream, usecols=["userId", "movieId", "rating"],
                                 dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}, chunksize=100_000):
            users = np.searchsorted(user_ids, chunk.userId.to_numpy())
            items = np.searchsorted(movie_ids, chunk.movieId.to_numpy())
            np.testing.assert_array_equal(user_ids[users], chunk.userId.to_numpy())
            np.testing.assert_array_equal(movie_ids[items], chunk.movieId.to_numpy())
            ratings = chunk.rating.to_numpy().astype(np.float64)
            test = rng.random(len(chunk)) >= .8
            stop = offset + len(chunk)
            for name, value in (("users", users), ("items", items), ("ratings", ratings), ("test", test)):
                cached[name][offset:stop] = value
            user_count += np.bincount(users, minlength=n_users)
            movie_count += np.bincount(items, minlength=n_movies)
            train_users += np.bincount(users[~test], minlength=n_users)
            train_movies += np.bincount(items[~test], minlength=n_movies)
            test_users += np.bincount(users[test], minlength=n_users)
            test_movies += np.bincount(items[test], minlength=n_movies)
            rating_counts += np.bincount((ratings*2).astype(int)-1, minlength=10)
            rating_sum += np.bincount(items, weights=ratings, minlength=n_movies)
            rating_squares += np.bincount(items, weights=ratings**2, minlength=n_movies)
            train_sum += ratings[~test].sum()
            test_sum += ratings[test].sum()
            test_square_sum += (ratings[test]**2).sum()
            for name, model in models.items():
                predictions = model["b_u"][users] + model["b_i"][items]
                predictions += np.einsum("ij,ij->i", model["u"][users], model["v"][items])
                square_error = (ratings-predictions)**2
                errors[name] += [square_error[~test].sum(), square_error[test].sum()]
                if name in user_errors:
                    user_errors[name] += np.bincount(users[test], weights=square_error[test], minlength=n_users)
                    movie_errors[name] += np.bincount(items[test], weights=square_error[test], minlength=n_movies)
            offset = stop
            if offset % 2_000_000 == 0: print(f"Verified {offset:,}/{total_rows:,} ratings", flush=True)
        assert reader.md5.hexdigest() == "cf12b74f9ad4b94a011f079e26d4270a"
        assert reader.sha256.hexdigest() == manifest["sha256"]["ratings.csv"]
assert offset == total_rows
for array in cached.values(): array.flush()
assert int(train_users.sum()) == 25601250 and int(test_users.sum()) == 6398954
np.testing.assert_array_equal(saved_counts, movie_count)
assert train_users.min() > 0

genres = catalog.genres.fillna("(no genres listed)").str.split("|")
genre_names = sorted({g for values in genres for g in values if g != "(no genres listed)"})
genre_indices = {g: i for i,g in enumerate(genre_names)}
features = np.zeros((n_movies,len(genre_names)))
for i, values in enumerate(genres):
    columns = [genre_indices[g] for g in values if g in genre_indices]
    if columns: features[i,columns] = 1/np.sqrt(len(columns))
comparison = pd.read_csv(RESULTS / "als_test_comparison.csv").set_index("name")
metrics=[]
for name,model in models.items():
    full = name.startswith("full_")
    observed = movie_count > 0 if full else train_movies > 0
    nnz = total_rows if full else int(train_users.sum())
    train_sse = errors[name].sum() if full else errors[name][0]
    test_rmse = None if full else float(np.sqrt(errors[name][1]/test_users.sum()))
    lam,tau = (comparison.loc[name,["lam","tau"]] if name in comparison.index else (.01,.1))
    prior = features @ model["features"] if "features" in model else np.zeros_like(model["v"])
    penalty=.001*((model["b_u"]**2).sum()+(model["b_i"]**2).sum())
    penalty+=tau*((model["u"]**2).sum()+((model["v"][observed]-prior[observed])**2).sum())
    if "features" in model:
        penalty+=.1*(model["features"]**2).sum()
        np.testing.assert_allclose(model["v"][~observed], prior[~observed], atol=1e-9)
    objective=.5*(lam*train_sse+penalty)
    history_name = "genre_history" if name=="genre_model" else name+"_history"
    recorded = pd.read_csv(RESULTS/(history_name+".csv")).iloc[-1]
    np.testing.assert_allclose(np.sqrt(train_sse/nnz),recorded.train_rmse,atol=1e-9)
    np.testing.assert_allclose(objective,recorded.loss,rtol=1e-9)
    if test_rmse is not None: np.testing.assert_allclose(test_rmse,recorded.test_rmse,atol=1e-9)
    metrics.append({"model":name,"train_rmse":float(np.sqrt(train_sse/nnz)),"test_rmse":test_rmse,"objective":float(objective)})

mean = train_sum/train_users.sum()
mean_rmse = float(np.sqrt((test_square_sum-2*mean*test_sum+len(cached['ratings'][cached['test']])*mean**2)/test_users.sum()))
final = pd.read_csv(RESULTS/"final_test_results.csv").set_index("model")
np.testing.assert_allclose(mean_rmse,final.loc["Training mean","test_rmse"],atol=1e-9)
groups = pd.read_csv(RESULTS/"user_activity_errors.csv")
for row in groups.itertuples():
    label = str(row.user_train_ratings)
    mask = {"0":train_users==0,"1-19":(train_users>=1)&(train_users<20),"20-99":(train_users>=20)&(train_users<100),"100+":train_users>=100}[label]
    count = int(test_users[mask].sum())
    assert count == row.test_ratings
    if count: np.testing.assert_allclose(np.sqrt(user_errors[row.model][mask].sum()/count),row.rmse,atol=1e-9)
cold = pd.read_csv(RESULTS/"natural_cold_start_errors.csv")
for row in cold.itertuples():
    if row.group.startswith('cold user'): assert row.test_ratings==0; continue
    mask = train_movies==0 if row.group.endswith('cold item') else train_movies>0
    model = 'als_k10' if row.model=='Configured ALS' else 'genre_model'
    count=int(test_movies[mask].sum()); assert count==row.test_ratings
    np.testing.assert_allclose(np.sqrt(movie_errors[model][mask].sum()/count),row.rmse,atol=1e-9)

def verify_controlled_cohort(cached, movie_ids, train_movies, n_users, results):
    """Reconstruct held-out movies and eligible test rows without refitting."""
    candidates = np.flatnonzero(train_movies >= 20)
    selected = np.random.default_rng(43).choice(
        candidates, max(1, len(candidates) // 20), replace=False)
    np.testing.assert_array_equal(
        movie_ids[selected], np.load(results / 'cold_movie_ids.npy'))
    cold = np.zeros(len(movie_ids), dtype=bool)
    cold[selected] = True
    remaining = np.zeros(n_users, dtype=np.int64)
    for start in range(0, len(cached['test']), 500000):
        section = slice(start, start + 500000)
        keep = ~cached['test'][section] & ~cold[cached['items'][section]]
        remaining += np.bincount(cached['users'][section][keep], minlength=n_users)
    count = 0
    for start in range(0, len(cached['test']), 500000):
        section = slice(start, start + 500000)
        keep = (cached['test'][section] & cold[cached['items'][section]]
                & (remaining[cached['users'][section]] > 0))
        count += int(keep.sum())
    recorded = pd.read_csv(results / 'controlled_cold_start.csv')
    assert (recorded.held_out_movies == len(selected)).all()
    assert (recorded.test_ratings == count).all()
    return {'movies': len(selected), 'test_ratings': count}


controlled_cohort = verify_controlled_cohort(
    cached, movie_ids, train_movies, n_users, RESULTS)
np.savez(OUT/'distribution_arrays.npz',rating_counts=rating_counts,user_count=user_count,movie_count=movie_count,train_users=train_users,train_movies=train_movies,test_users=test_users,test_movies=test_movies)
pd.DataFrame(metrics).to_csv(OUT/'verified_metrics.csv',index=False)
pd.DataFrame({'rating':np.arange(.5,5.1,.5),'count':rating_counts}).to_csv(OUT/'rating_counts.csv',index=False)
for label,counts in [('users',user_count),('movies',movie_count)]:
    values,counts_of_values=np.unique(counts,return_counts=True)
    pd.DataFrame({'degree':values,'entities':counts_of_values,'fraction':counts_of_values/len(counts)}).to_csv(OUT/f'{label}_degrees.csv',index=False)

# Check every recorded recommendation directly from the full-data movie model.
import sys
sys.path.insert(0,str(ROOT))
from test_train import FUNCTIONS
FUNCTIONS.update(movie_ids=movie_ids,n_movies=n_movies,movies=catalog,gam=.001,chosen_parameters={'lam':.01,'tau':.1})
for weight in (1.,.05):
    for support in (1,100):
        actual=FUNCTIONS['recommend_for_profile']({4993:5.},models['full_genre'],movie_count,bias_weight=weight,minimum_ratings=support)
        expected=pd.read_csv(RESULTS/f'recommendations_{weight}_{support}.csv')
        np.testing.assert_array_equal(actual.movieId,expected.movieId)
        np.testing.assert_allclose(actual.ranking_score,expected.ranking_score,atol=1e-10)
        np.testing.assert_allclose(actual.predicted_rating,expected.predicted_rating,atol=1e-10)
polarizing=pd.read_csv(RESULTS/'polarizing_movies.csv')
indices=np.searchsorted(movie_ids,polarizing.movieId)
np.testing.assert_allclose(np.linalg.norm(models['full_genre']['v'][indices],axis=1),polarizing.vector_norm,atol=1e-10)
std=np.sqrt(np.maximum(0,rating_squares/np.maximum(movie_count,1)-(rating_sum/np.maximum(movie_count,1))**2))
np.testing.assert_allclose(std[indices],polarizing.rating_std,atol=1e-9)

# Reproduce the complete 28-update bias trace; the uploaded counter restarted at 8.
del models,features,user_errors,movie_errors
gc.collect()
from scipy.sparse import csr_matrix
train=~cached['test']
by_user=csr_matrix((cached['ratings'][train],(cached['users'][train],cached['items'][train])),shape=(n_users,n_movies))
by_item=by_user.T.tocsr()
del train
bias_model=FUNCTIONS['initialize_model'](n_users,n_movies,0,42)
bias=pd.read_csv(RESULTS/'bias_history.csv')
bias_records=[]
for position in range(len(bias)):
    for data,own,other in [(by_user,'b_u','b_i'),(by_item,'b_i','b_u')]:
        FUNCTIONS['update_biases'](data.indptr,data.indices,data.data,bias_model[own],bias_model[other],.01,.001)
    sse=FUNCTIONS['training_sse'](by_user,bias_model)
    test_sse=0.
    for start in range(0,total_rows,500000):
        section=slice(start,start+500000); mask=cached['test'][section]
        prediction=bias_model['b_u'][cached['users'][section][mask]]+bias_model['b_i'][cached['items'][section][mask]]
        test_sse+=((cached['ratings'][section][mask]-prediction)**2).sum()
    penalty=sum((bias_model[k]**2).sum() for k in ['b_u','b_i'])
    values={'sweep':position+1,'loss':.5*(.01*sse+.001*penalty),'train_rmse':np.sqrt(sse/by_user.nnz),'test_rmse':np.sqrt(test_sse/test_users.sum())}
    np.testing.assert_allclose([values[k] for k in ['loss','train_rmse','test_rmse']],bias.iloc[position][['loss','train_rmse','test_rmse']].to_numpy(float),rtol=1e-9,atol=1e-9)
    bias_records.append(values)
    if (position+1)%7==0: print('Verified bias update',position+1,flush=True)
pd.DataFrame(bias_records).to_csv(OUT/'bias_history_corrected.csv',index=False)

audit={'status':'passed','ratings':total_rows,'train':int(train_users.sum()),'test':int(test_users.sum()),'official_md5_verified':True,'manifest_sha256_verified':True,'bundle_copies_identical':True,'bundle_matches_full_genre':True,'bundle_hashes':hashes,'independent_final_model_metrics':metrics,'mean_test_rmse':mean_rmse,'bias_updates_verified':len(bias),'recommendation_tables_verified':4,'natural_cold_start_verified':True,'user_activity_errors_verified':True,'polarization_statistics_verified':True,'limitations':['Controlled cold-start RMSE cannot be recomputed from saved weights: those two model checkpoints were not exported.','Two-dimensional coordinates can be checked against figures but their fitted models were not exported.','Single seed, test monitored each sweep; no validation selection or statistical significance claim.']}
audit['controlled_cold_cohort_verified'] = controlled_cohort
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('PASS: independent 32M audit complete.',flush=True)
