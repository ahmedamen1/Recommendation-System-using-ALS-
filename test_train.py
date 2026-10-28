"""Check ALS mathematics and edge cases without running data experiments."""

import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def load_functions():
    """Load imports/functions from the same source used by the notebook."""
    tree = ast.parse(Path("train.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef)
    )]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 "train.py", "exec"), namespace)
    namespace["numba"].set_num_threads(2)
    namespace["update_side_parallel"] = namespace["njit"](parallel=True)(
        namespace["update_side_kernel"]
    )
    return namespace


FUNCTIONS = load_functions()


class TestAls(unittest.TestCase):
    def setUp(self):
        self.data = csr_matrix(np.array([
            [5, 0, 3, 0], [4, 1, 0, 0], [0, 2, 4, 0], [0, 0, 0, 0],
        ], dtype=np.float32))
        self.items = self.data.T.tocsr()

    def test_reference_and_parallel_full_sweep(self):
        reference = FUNCTIONS["initialize_model"](4, 4, 3, 12)
        accelerated = {key: value.copy() for key, value in reference.items()}
        for data, own, other, own_bias, other_bias in (
            (self.data, "u", "v", "b_u", "b_i"),
            (self.items, "v", "u", "b_i", "b_u"),
        ):
            for model, function in (
                (reference, "update_side_reference"),
                (accelerated, "update_side"),
            ):
                FUNCTIONS[function](
                    data, model[own], model[other], model[own_bias],
                    model[other_bias], 0.01, 0.001, 0.1,
                    np.zeros_like(model[own]),
                )
        for key in reference:
            np.testing.assert_allclose(reference[key], accelerated[key],
                                       rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(accelerated["u"][3], 0)
        np.testing.assert_array_equal(accelerated["v"][3], 0)
        self.assertEqual(accelerated["b_u"][3], 0)

    def test_objective_and_feature_normal_equation(self):
        features = np.array([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=float)
        for feature_input in (None, features):
            model, history, sweep = FUNCTIONS["fit_als"](
                self.data, self.items, None, factors=2, sweeps=6,
                features=feature_input,
            )
            self.assertEqual(sweep, 6)
            self.assertTrue(np.all(np.diff(history["loss"]) <= 1e-9))
            sse = 0
            for row, column in zip(*self.data.nonzero()):
                prediction = (model["b_u"][row] + model["b_i"][column]
                              + model["u"][row] @ model["v"][column])
                sse += (self.data[row, column] - prediction) ** 2
            self.assertAlmostEqual(sse, FUNCTIONS["training_sse"](
                self.data, model
            ), places=9)
            prior = np.zeros_like(model["v"])
            feature_penalty = 0
            if feature_input is not None:
                prior = features @ model["features"]
                normal = 0.1 * features[:3].T @ features[:3] + 0.1 * np.eye(2)
                target = 0.1 * features[:3].T @ model["v"][:3]
                np.testing.assert_allclose(normal @ model["features"], target)
                np.testing.assert_allclose(model["v"][3], prior[3])
                feature_penalty = 0.1 * np.sum(model["features"] ** 2)
            objective = 0.5 * (
                0.01 * sse + feature_penalty
                + 0.001 * (np.sum(model["b_u"] ** 2)
                           + np.sum(model["b_i"] ** 2))
                + 0.1 * (np.sum(model["u"] ** 2)
                         + np.sum((model["v"][:3] - prior[:3]) ** 2))
            )
            self.assertAlmostEqual(objective, history["loss"].iloc[-1])

    def test_fixed_sweeps_and_test_does_not_change_training(self):
        rows = (np.array([0, 1]), np.array([1, 2]), np.array([2.0, 3.0]))
        model, history, sweep = FUNCTIONS["fit_als"](
            self.data, self.items, rows, factors=2, sweeps=5,
        )
        self.assertAlmostEqual(FUNCTIONS["rmse"](rows, model),
                               history["test_rmse"].iloc[-1])
        self.assertEqual(sweep, 5)
        # Test metrics cannot affect parameter updates or checkpoint choice.
        changed_rows = (rows[0], rows[1], np.array([100.0, -100.0]))
        changed_model, changed_history, changed_sweep = FUNCTIONS["fit_als"](
            self.data, self.items, changed_rows, factors=2, sweeps=5,
        )
        self.assertEqual(changed_sweep, 5)
        self.assertFalse(np.allclose(history["test_rmse"],
                                     changed_history["test_rmse"]))
        for key in model:
            np.testing.assert_array_equal(model[key], changed_model[key])
        empty = (np.array([], dtype=int), np.array([], dtype=int),
                 np.array([]))
        self.assertTrue(np.isnan(FUNCTIONS["rmse"](empty, model)))

    def test_recommendation_exclusions_and_invalid_id(self):
        FUNCTIONS["movie_ids"] = np.array([10, 20, 30, 40])
        FUNCTIONS["n_movies"] = 4
        FUNCTIONS["gam"] = 0.001
        FUNCTIONS["chosen_parameters"] = {"lam": 0.01, "tau": 0.1}
        FUNCTIONS["movies"] = FUNCTIONS["pd"].DataFrame({
            "movieId": [10, 20, 30, 40], "title": ["a", "b", "c", "d"],
            "genres": ["Drama"] * 4,
        })
        model = FUNCTIONS["initialize_model"](4, 4, 2, 12)
        result = FUNCTIONS["recommend_for_profile"](
            {10: 5.0}, model, np.array([30, 30, 1, 0]), minimum_ratings=20,
        )
        self.assertEqual(result["movieId"].tolist(), [20])
        with self.assertRaises(ValueError):
            FUNCTIONS["recommend_for_profile"](
                {999: 5.0}, model, np.ones(4),
            )


if __name__ == "__main__":
    unittest.main()
