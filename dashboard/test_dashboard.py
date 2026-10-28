"""Verify notebook-to-dashboard parity and the local HTTP API."""

import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from dashboard.server import Recommender, handler_for
from test_train import FUNCTIONS


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name) / "bundle"
        self.catalog = pd.DataFrame({
            "movieId": [10, 20, 30, 40, 50],
            "title": ["One (2000)", "Two (2001)", "Three (2002)",
                      "Four (2003)", "Five (2004)"],
            "genres": ["Drama", "Drama|Comedy", "Action", "Comedy", "Drama"],
        })
        self.ids = self.catalog.movieId.to_numpy()
        self.counts = np.array([30, 35, 50, 0, 1])
        self.model = FUNCTIONS["initialize_model"](2, 5, 3, 19)
        self.model["b_i"][:] = [2.0, 2.1, 1.8, 0.0, 2.2]
        self.metadata = {"dataset": "synthetic", "dataset_ratings": 116,
                         "training_ratings": 116, "full_refit": True,
                         "model_type": "ALS", "sweeps": 2,
                         "lam": .01, "gam": .001, "tau": .1}
        self.archive = FUNCTIONS["export_inference_bundle"](
            self.folder, self.model, self.catalog, self.ids, self.counts,
            self.metadata,
        )
        self.engine = Recommender(self.archive)

    def tearDown(self):
        self.temporary.cleanup()

    def test_predictions_match_notebook_and_exclude_rated_movies(self):
        FUNCTIONS.update(movie_ids=self.ids, n_movies=5, movies=self.catalog,
                         gam=.001, chosen_parameters={"lam": .01, "tau": .1})
        profile = {10: 5.0, 30: 1.0}
        expected = FUNCTIONS["recommend_for_profile"](
            profile, self.model, self.counts, minimum_ratings=1,
        )
        actual = self.engine.recommend({str(k): v for k, v in profile.items()}, 1)
        self.assertEqual([m["movieId"] for m in actual["movies"]],
                         expected.movieId.tolist())
        np.testing.assert_allclose([m["score"] for m in actual["movies"]],
                                   expected.ranking_score.to_numpy())
        np.testing.assert_allclose([m["raw_prediction"] for m in actual["movies"]],
                                   expected.predicted_rating.to_numpy())
        self.assertTrue(all(m["movieId"] not in profile for m in actual["movies"]))

    def test_inputs_filters_and_no_mutation(self):
        before = self.engine.v.copy()
        for ratings in ({}, {"999": 3}, {"10": float("nan")}, {"10": True},
                        {"10": "5"}, {"10": 5.1}):
            with self.assertRaises(ValueError):
                self.engine.recommend(ratings)
        movies = self.engine.recommend({"10": 4.5}, 0, "Comedy")["movies"]
        self.assertTrue(movies)
        self.assertTrue(all("Comedy" in m["genres"] for m in movies))
        np.testing.assert_array_equal(before, self.engine.v)
        self.assertEqual(self.engine.catalog("two")["movies"][0]["movieId"], 20)
        self.assertEqual(self.engine.catalog("absent")["total"], 0)
        self.assertEqual(self.engine.bundle_id, Recommender(self.folder).bundle_id)

    def test_catalog_alignment_rejected(self):
        self.catalog.iloc[::-1].to_csv(self.folder / "movies.csv", index=False)
        with self.assertRaisesRegex(ValueError, "order"):
            Recommender(self.folder)

    def test_combined_filters_sorting_and_unknown_year(self):
        result = self.engine.catalog("two", "Comedy", minimum_ratings=30,
                                     year_from=2001, year_to=2001)
        self.assertEqual([m["movieId"] for m in result["movies"]], [20])
        newest = self.engine.catalog(sort="newest")["movies"]
        self.assertEqual([m["year"] for m in newest], [2004, 2003, 2002, 2001, 2000])
        self.assertEqual(self.engine.catalog(sort="title")["movies"][0]["movieId"], 50)
        self.engine.years[4] = 0
        self.assertNotIn(50, [m["movieId"] for m in self.engine.catalog(year_to=2004)["movies"]])
        self.assertEqual(self.engine.catalog(sort="oldest")["movies"][-1]["movieId"], 50)
        for filters in ({"year_from": 2004, "year_to": 2000}, {"genre": "unknown"},
                        {"year_from": "2000.5"}, {"sort": "unknown"}):
            with self.assertRaises(ValueError):
                self.engine.catalog(**filters)

    def test_diversity_changes_order_without_changing_predictions(self):
        # Two almost-equally relevant dramas followed by an action movie.
        self.engine.genre_sets[:3] = [{"Drama"}, {"Drama"}, {"Action"}]
        scores = np.array([5., 4.9, 4.8, 0., 0.])
        ranked = np.array([0, 1, 2, 3, 4])
        self.assertEqual(list(self.engine.diversify(ranked, scores, 3, 0)), [0, 1, 2])
        self.assertEqual(self.engine.diversify(ranked, scores, 3, .4), [0, 2, 1])
        base = self.engine.recommend({"10": 5}, 0)
        diverse = self.engine.recommend({"10": 5}, 0, diversity=.4)
        expected = {m["movieId"]: m["raw_prediction"] for m in base["movies"]}
        self.assertEqual(expected, {m["movieId"]: m["raw_prediction"] for m in diverse["movies"]})
        for movie in diverse["movies"]:
            self.assertAlmostEqual(sum(movie["score_parts"].values()), movie["raw_prediction"])
        self.assertEqual(len(expected), len(diverse["movies"]))
        for invalid in (True, -.1, .6, float("nan"), "0.2"):
            with self.assertRaises(ValueError):
                self.engine.recommend({"10": 5}, diversity=invalid)

    def test_recommendation_filters_precede_limit_and_series(self):
        with patch("dashboard.server.COLLECTIONS", {"Example": (10, 20, 30)}):
            result = self.engine.recommend({"10": 5}, minimum_ratings=1, limit=1,
                                           query="two", year_from=2001, year_to=2001,
                                           genre="Comedy", diversity=.4)
            self.assertEqual([m["movieId"] for m in result["movies"]], [20])
            self.assertEqual(result["eligible_movies"], 1)
            self.assertEqual(result["series"][0]["movies"][0]["movieId"], 20)
            self.assertEqual(self.engine.recommend({"10": 5}, query="absent")["movies"], [])

    def test_series_uses_model_scores_and_respects_exclusions_and_filters(self):
        with patch("dashboard.server.COLLECTIONS", {"Example": (10, 30, 40, 50)}):
            full = self.engine.recommend({"10": 5}, minimum_ratings=0, limit=48)
            limited = self.engine.recommend({"10": 5}, minimum_ratings=0, limit=1)
            expected = [m for m in full["movies"] if m["movieId"] in (30, 40, 50)]
            actual = limited["series"][0]["movies"]
            self.assertEqual([m["movieId"] for m in actual], [m["movieId"] for m in expected])
            self.assertEqual([m["raw_prediction"] for m in actual], [m["raw_prediction"] for m in expected])
            self.assertEqual(limited["movies"], full["movies"][:1])
            self.assertEqual(self.engine.recommend({"10": 1}, 0)["series"], [])
            self.assertEqual(self.engine.recommend({"10": 5, "30": 5}, 20)["series"], [])
            self.assertEqual(self.engine.recommend({"10": 5}, 20, "Comedy")["series"], [])

    def test_http_endpoints_and_invalid_requests(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base) as response:
                self.assertIn(b"Movie Recommender", response.read())
            with urlopen(base + "/api/model") as response:
                self.assertEqual(json.load(response)["training_ratings"], 116)
            body = json.dumps({"ratings": {"10": 5}, "minimum_ratings": 1}).encode()
            with urlopen(Request(base + "/api/recommend", body,
                                 {"Content-Type": "application/json"})) as response:
                self.assertTrue(json.load(response)["movies"])
            with urlopen(base + "/api/movies?year_from=2001&year_to=2001&minimum_ratings=30&sort=newest") as response:
                self.assertEqual(json.load(response)["movies"][0]["movieId"], 20)
            body = json.dumps({"ratings": {"10": 5}, "q": "two", "year_from": 2001,
                               "year_to": 2001, "diversity": .4}).encode()
            with urlopen(Request(base + "/api/recommend", body)) as response:
                self.assertEqual([m["movieId"] for m in json.load(response)["movies"]], [20])
            for path in ("/api/movies?offset=-1", "/api/movies?limit=999",
                         "/api/movies?year_from=2020&year_to=2000", "/api/movies?genre=bad",
                         "/api/movies?sort=bad", "/api/movies?minimum_ratings=-1"):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + path)
                self.assertEqual(caught.exception.code, 400)
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + "/api/recommend", b'{"ratings":{"10":8}}'))
            self.assertEqual(caught.exception.code, 400)
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/../server.py")
            self.assertEqual(caught.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
