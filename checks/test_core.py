"""Small semantic checks on synthetic data."""
import unittest
import numpy as np
from lgwm.evaluation import CLASSES, metrics, fit_harm_head, predict_harm


class CoreTests(unittest.TestCase):
    def test_identical_outcomes_have_chance_auc(self):
        classes = np.repeat(CLASSES, 3)
        values = np.tile([0.1, 0.4, 0.9], 5)
        self.assertEqual(metrics(classes, values)["RSWT"], 0.5)

    def test_harm_and_integrity_are_different_axes(self):
        classes = np.repeat(CLASSES, 2)
        values = np.repeat([1, 0, 0, 0, 1], 2)
        result = metrics(classes, values)
        self.assertEqual(result["Integrity_mAUC"], 1)
        self.assertEqual(result["Harm_AUC"], 0.5)

    def test_threshold_attains_recall_with_ties(self):
        classes = np.repeat(CLASSES, 10)
        values = np.tile([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], 5)
        self.assertGreaterEqual(metrics(classes, values)["recall_at_threshold"], 0.9)

    def test_portable_head_matches_sklearn(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(4)
        x = rng.normal(size=(100, 10)).astype(np.float32)
        classes = np.where(x[:, 0] > 0, CLASSES[0], CLASSES[4])
        head = fit_harm_head(x, classes)
        y = (classes == CLASSES[0]).astype(int)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(x), y)
        np.testing.assert_allclose(predict_harm(x, head), model.predict_proba(scaler.transform(x))[:, 1],
                                   rtol=1e-12, atol=1e-12)

    def test_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            metrics(CLASSES, [0, 0, float("nan"), 0, 0])


if __name__ == "__main__":
    unittest.main()
