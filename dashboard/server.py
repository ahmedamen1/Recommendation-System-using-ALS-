"""Serve the local movie dashboard and run inference with notebook weights.

Usage: python dashboard/server.py --bundle path/to/inference_bundle.zip
Only NumPy is required; the raw ratings and notebook are not loaded.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import zipfile

import numpy as np


STATIC = Path(__file__).resolve().parent / "static"
BUNDLE_FILES = ("metadata.json", "movies.csv", "model.npz")
# Catalog grouping for the requested sequel demonstration, separate from ALS.
COLLECTIONS = {"Toy Story": (1, 3114, 78499, 201588)}


def default_bundle():
    """Prefer the user's downloaded model; retain the small demo fallback."""
    root = Path(__file__).resolve().parent.parent
    uploaded = root / "inference_bundle"
    if all((uploaded / name).is_file() for name in BUNDLE_FILES):
        return uploaded
    return Path(__file__).resolve().parent / "demo_model"


class Recommender:
    """Immutable movie parameters; each request fits its own new user."""

    def __init__(self, bundle):
        bundle = Path(bundle)
        if bundle.is_dir():
            payload = {name: (bundle / name).read_bytes() for name in BUNDLE_FILES}
        else:
            # Read three known members without extracting arbitrary ZIP paths.
            with zipfile.ZipFile(bundle) as archive:
                payload = {}
                for name in BUNDLE_FILES:
                    if archive.getinfo(name).file_size > 256 * 1024**2:
                        raise ValueError(f"Bundle member too large: {name}")
                    payload[name] = archive.read(name)
        self.metadata = json.loads(payload["metadata.json"])
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Export a schema-version-1 inference bundle from the notebook.")
        with np.load(io.BytesIO(payload["model.npz"]), allow_pickle=False) as data:
            self.v = np.asarray(data["item_factors"], dtype=np.float64)
            self.bias = np.asarray(data["item_bias"], dtype=np.float64)
            raw_ids, raw_counts = data["movie_ids"], data["movie_counts"]
            if raw_ids.dtype.kind not in "iu" or raw_counts.dtype.kind not in "iu":
                raise ValueError("Movie IDs and counts must be integer arrays.")
            self.ids = raw_ids.astype(np.int64)
            self.counts = raw_counts.astype(np.int64)
        rows = list(csv.DictReader(io.StringIO(payload["movies.csv"].decode("utf-8-sig"))))
        size = len(rows)
        if (not size or self.v.ndim != 2 or self.v.shape[0] != size
                or self.v.shape[1] < 1 or self.bias.shape != (size,)
                or self.ids.shape != (size,) or self.counts.shape != (size,)):
            raise ValueError("Bundle array shapes do not match the movie catalog.")
        if not np.isfinite(self.v).all() or not np.isfinite(self.bias).all():
            raise ValueError("Model parameters must be finite.")
        if len(set(self.ids)) != size or (self.counts < 0).any():
            raise ValueError("Invalid movie IDs or counts.")
        if [int(row["movieId"]) for row in rows] != self.ids.tolist():
            raise ValueError("Movie catalog order does not match model IDs.")
        for key in ("lam", "gam", "tau"):
            value = self.metadata.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Missing numeric model setting: {key}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid model setting: {key}")
        if self.metadata.get("training_ratings") != int(self.counts.sum()):
            raise ValueError("Metadata training count disagrees with movie counts.")
        self.movies = [{"movieId": int(self.ids[i]), "title": row["title"],
                        "genres": [g for g in row["genres"].split("|")
                                   if g and g != "(no genres listed)"],
                        "rating_count": int(self.counts[i])}
                       for i, row in enumerate(rows)]
        self.index = {int(movie_id): i for i, movie_id in enumerate(self.ids)}
        for movie in self.movies:
            match = re.search(r"\((\d{4})\)\s*$", movie["title"])
            movie["year"] = int(match[1]) if match else None
        self.years = np.array([movie["year"] or 0 for movie in self.movies])
        self.genre_sets = [set(movie["genres"]) for movie in self.movies]
        poster_manifest = STATIC / "posters" / "sources.json"
        posters = json.loads(poster_manifest.read_text(encoding="utf-8")) if poster_manifest.exists() else {}
        for movie in self.movies:
            poster = posters.get(str(movie["movieId"]), {})
            filename = poster.get("file", "")
            if filename and Path(filename).name == filename and (STATIC / "posters" / filename).is_file():
                movie["poster_url"] = "/posters/" + filename
        self.poster_files = {movie["poster_url"] for movie in self.movies if "poster_url" in movie}
        self.search_titles = [movie["title"].casefold() for movie in self.movies]
        self.popular = np.argsort(-self.counts, kind="stable")
        self.genres = sorted({g for movie in self.movies for g in movie["genres"]})
        fingerprint = hashlib.sha256()
        for name in BUNDLE_FILES:
            fingerprint.update(payload[name])
        self.bundle_id = fingerprint.hexdigest()[:16]
        for array in (self.v, self.bias, self.ids, self.counts):
            array.flags.writeable = False

    def info(self):
        return {"bundle_id": self.bundle_id, "dataset": self.metadata["dataset"],
                "dataset_ratings": self.metadata["dataset_ratings"],
                "training_ratings": self.metadata["training_ratings"],
                "full_refit": self.metadata["full_refit"],
                "model_type": self.metadata["model_type"],
                "sweeps": self.metadata["sweeps"], "factors": self.v.shape[1],
                "movies": len(self.movies), "genres": self.genres,
                "year_min": int(self.years[self.years > 0].min()) if (self.years > 0).any() else None,
                "year_max": int(self.years.max()) or None,
                "small_data": self.metadata["dataset"] != "ml-32m"}

    def candidate_mask(self, query="", genre="", minimum_ratings=0,
                       year_from=None, year_to=None):
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Search must be text of at most 200 characters.")
        if not isinstance(genre, str) or (genre and genre not in self.genres):
            raise ValueError("Unknown genre.")
        minimum_ratings = bounded_int(minimum_ratings, 0, 1000000, "minimum ratings")
        if year_from not in (None, ""):
            year_from = bounded_int(year_from, 1800, 2100, "start year")
        if year_to not in (None, ""):
            year_to = bounded_int(year_to, 1800, 2100, "end year")
        if year_from and year_to and year_from > year_to:
            raise ValueError("Start year must be no later than end year.")
        words = query.casefold().split()
        mask = self.counts >= minimum_ratings
        if words:
            mask &= np.array([all(word in title for word in words) for title in self.search_titles])
        if genre:
            mask &= np.array([genre in genres for genres in self.genre_sets])
        if year_from or year_to:
            mask &= self.years > 0
        if year_from:
            mask &= self.years >= year_from
        if year_to:
            mask &= self.years <= year_to
        return mask

    def catalog(self, query="", genre="", offset=0, limit=12,
                minimum_ratings=0, year_from=None, year_to=None, sort="popular"):
        if sort not in ("popular", "newest", "oldest", "title"):
            raise ValueError("Unknown catalog sort order.")
        mask = self.candidate_mask(query, genre, minimum_ratings, year_from, year_to)
        matches = [int(i) for i in self.popular if mask[i]]
        if sort == "newest":
            matches.sort(key=lambda i: -self.years[i])
        elif sort == "oldest":
            matches.sort(key=lambda i: self.years[i] or 9999)
        elif sort == "title":
            matches.sort(key=lambda i: self.search_titles[i])
        return {"movies": [self.movies[i] for i in matches[offset:offset + limit]],
                "total": len(matches), "offset": offset}

    def recommend(self, ratings, minimum_ratings=20, genre="", limit=12,
                  query="", year_from=None, year_to=None, diversity=0):
        if (isinstance(diversity, bool) or not isinstance(diversity, (int, float))
                or not math.isfinite(diversity) or not 0 <= diversity <= 0.5):
            raise ValueError("Diversity must be a number between 0 and 0.5.")
        if not isinstance(ratings, dict) or not 1 <= len(ratings) <= 200:
            raise ValueError("Rate between 1 and 200 movies to get recommendations.")
        profile = {}
        for raw_id, rating in ratings.items():
            try:
                movie_id = int(raw_id)
            except (ValueError, TypeError):
                raise ValueError("Movie IDs must be integers.") from None
            if movie_id not in self.index or str(movie_id) != str(raw_id):
                raise ValueError("Your profile contains an unknown movie ID.")
            if (isinstance(rating, bool) or not isinstance(rating, (int, float))
                    or not math.isfinite(rating) or not 0.5 <= rating <= 5.0):
                raise ValueError("Ratings must be numbers between 0.5 and 5.")
            profile[self.index[movie_id]] = float(rating)
        indices = np.array(list(profile), dtype=int)
        values = np.array(list(profile.values()), dtype=np.float64)
        # Same regularized new-user solve as recommend_for_profile in the notebook.
        design = np.column_stack([np.ones(len(indices)), self.v[indices]])
        lam, gam, tau = (self.metadata[key] for key in ("lam", "gam", "tau"))
        penalty = np.diag([gam] + [tau] * self.v.shape[1])
        coefficients = np.linalg.solve(
            lam * design.T @ design + penalty,
            lam * design.T @ (values - self.bias[indices]),
        )
        interaction = self.v @ coefficients[1:]
        scores = interaction + self.bias
        predictions = coefficients[0] + scores
        eligible = self.candidate_mask(query, genre, minimum_ratings, year_from, year_to)
        eligible[indices] = False
        ranked = np.flatnonzero(eligible)
        ranked = ranked[np.argsort(-scores[ranked], kind="stable")]
        selected = self.diversify(ranked, scores, limit, diversity)
        liked_genres = {g for i, rating in profile.items() if rating >= 3.5
                        for g in self.movies[i]["genres"]}
        def recommendation(i):
            movie = self.movies[i]
            overlap = [g for g in movie["genres"] if g in liked_genres]
            reason = ("Shares " + " & ".join(overlap[:2]) + " with your liked films"
                      if overlap else "Recommended from your rating profile")
            return {**movie, "score": float(scores[i]),
                    "score_parts": {"user_bias": float(coefficients[0]),
                                    "movie_bias": float(self.bias[i]),
                                    "taste_match": float(interaction[i])},
                                    "raw_prediction": float(predictions[i]),
                                    "predicted_rating": float(np.clip(predictions[i], 0.5, 5)),
                    "reason": reason}
        recommendations = [recommendation(i) for i in selected]
        series = []
        for title, members in COLLECTIONS.items():
            if not any(self.ids[i] in members and rating >= 3.5 for i, rating in profile.items()):
                continue
            matches = []
            for position, i in enumerate(ranked, 1):
                if self.ids[i] in members:
                    matches.append({**recommendation(i), "overall_rank": position})
            if matches:
                series.append({"title": title, "movies": matches})
        return {"movies": recommendations, "profile_size": len(profile),
                "series": series,
                "ranking": "diverse" if diversity else "model",
                "diversity": diversity,
                "eligible_movies": int(eligible.sum()),
                "note": "Predicted stars are clipped to 0.5–5 for display. "
                        + ("Genre variety reranks the top 200 raw-score candidates."
                           if diversity else "Ranking uses raw model scores.")}

    def diversify(self, ranked, scores, limit, strength):
        """Greedy relevance/genre-diversity reranking of at most 200 candidates.

        Base predictions remain untouched. Zero strength preserves notebook parity.
        """
        if not strength or len(ranked) < 2:
            return ranked[:limit]
        pool = list(ranked[:200])
        raw = scores[pool]
        relevance = (raw - raw.min()) / max(float(np.ptp(raw)), 1e-12)
        selected, remaining = [], list(range(len(pool)))
        while remaining and len(selected) < limit:
            def value(position):
                genres = self.genre_sets[pool[position]]
                similarity = max((len(genres & self.genre_sets[i]) /
                                  max(1, len(genres | self.genre_sets[i]))
                                  for i in selected), default=0)
                return (1 - strength) * relevance[position] - strength * similarity
            best = max(remaining, key=value)
            selected.append(pool[best])
            remaining.remove(best)
        return selected


def bounded_int(value, minimum, maximum, name):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {name}.")
    parsed = int(value)
    if str(parsed) != str(value) or not minimum <= parsed <= maximum:
        raise ValueError(f"Invalid {name}.")
    return parsed


def handler_for(model):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, status=200, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}",
                       f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self.respond({"error": "Use the localhost URL printed by the server."}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                self.respond({"error": "Only this local dashboard can call the API."}, 403)
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/model":
                    self.respond(model.info())
                elif parsed.path == "/api/movies":
                    query = parse_qs(parsed.query)
                    get = lambda key, default: query.get(key, [default])[0]
                    self.respond(model.catalog(
                        query=get("q", "")[:200], genre=get("genre", ""),
                        offset=bounded_int(get("offset", "0"), 0, len(model.movies), "offset"),
                        limit=bounded_int(get("limit", "12"), 1, 48, "limit"),
                        minimum_ratings=get("minimum_ratings", "0"),
                        year_from=get("year_from", ""), year_to=get("year_to", ""),
                        sort=get("sort", "popular"),
                    ))
                elif parsed.path in model.poster_files or parsed.path in {"/", "/index.html", "/app.js", "/style.css", "/favicon.svg", "/poster-credits.html"}:
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    if name.endswith(".js"):
                        content_type = "text/javascript"
                    self.respond((STATIC / name).read_bytes(), content_type=content_type)
                else:
                    self.respond({"error": "Not found."}, 404)
            except (ValueError, TypeError) as error:
                self.respond({"error": str(error)}, 400)

        def do_POST(self):
            if not self.local_request():
                return
            if self.path != "/api/recommend":
                self.respond({"error": "Not found."}, 404)
                return
            try:
                length = bounded_int(self.headers.get("Content-Length", "0"), 1, 32768, "request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                genre = body.get("genre", "")
                if not isinstance(genre, str) or (genre and genre not in model.genres):
                    raise ValueError("Unknown genre.")
                self.respond(model.recommend(
                    body.get("ratings"),
                    minimum_ratings=bounded_int(body.get("minimum_ratings", 20), 0, 1000000, "minimum ratings"),
                    genre=genre, limit=bounded_int(body.get("limit", 12), 1, 48, "limit"),
                    query=body.get("q", ""), year_from=body.get("year_from"),
                    year_to=body.get("year_to"), diversity=body.get("diversity", 0),
                ))
            except (ValueError, TypeError, OverflowError) as error:
                self.respond({"error": str(error)}, 400)

        def log_message(self, format, *args):
            # Keep the terminal useful while still showing request failures.
            if len(args) > 1 and str(args[1]) not in ("200", "304"):
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=default_bundle())
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    try:
        model = Recommender(args.bundle)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Cannot load model: {error}\nDownload inference_bundle.zip from the updated notebook and pass --bundle.\n")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(model))
    except OSError as error:
        parser.exit(1, f"Cannot start server: {error}\nIf Movie Recommender is already running, open http://127.0.0.1:{args.port}. Otherwise try --port 8001.\n")
    print(f"Movie Recommender is ready: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"{model.metadata['model_type']} | {model.metadata['dataset']} | "
          f"{model.metadata['training_ratings']:,} training ratings | "
          f"{model.metadata['sweeps']} sweeps", flush=True)
    print("Press Ctrl+C to stop. No training runs when this app starts.", flush=True)
    if args.open_browser:
        opener = threading.Timer(0.5, webbrowser.open, args=(f"http://127.0.0.1:{server.server_port}",))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
