import unittest
import numpy as np
from preprocess import Dataset, Subset


class TestSubset(unittest.TestCase):
    def setUp(self):
        self.n_samples = 1000
        self.n_dim = 15
        self.n_classes = 62
        X = np.random.rand(self.n_samples, self.n_dim)
        y = np.random.randint(0, self.n_classes, self.n_samples)
        self.subset = Subset(X, y, mapping=lambda x: x)

    def test_must_pass(self):
        self.assertTrue(True)

    def test_num_classes(self):
        self.assertEqual(self.n_classes, self.subset.num_classes)

    def test_filter(self):
        filtered_subset = self.subset.filtered()
        self.assertEqual(self.n_classes, filtered_subset.num_classes)

        labels = list(range(10))
        filtered_subset = self.subset.filtered(labels)
        self.assertEqual(len(labels), filtered_subset.num_classes)
        all_labels = set(lab for lab in filtered_subset.y)
        self.assertEqual(set(labels), all_labels)

        filtered_subset = self.subset.filtered(3)
        self.assertEqual(1, filtered_subset.num_classes)

    def test_equal_length(self):
        self.assertEqual(len(self.subset.X), len(self.subset.y))

    def test_X_y_shape(self):
        self.assertEqual(len(self.subset.X.shape), 2)
        self.assertEqual(len(self.subset.y.shape), 1)

    def test_not_none(self):
        self.assertIsNotNone(self.subset.X)
        self.assertIsNotNone(self.subset.y)

    def test_not_empty(self):
        self.assertGreater(len(self.subset.X), 0)
        self.assertGreater(len(self.subset.y), 0)

    def test_output_dict(self):
        d = self.subset.__dict__()
        self.assertIsInstance(d, dict)
        self.assertIn("X", d)
        self.assertIn("y", d)
        self.assertIn("mapping", d)
        self.assertIn("num_classes", d)
        self.assertEqual(len(d), 4)

    def test_from_dict(self):
        d = {
            "X": np.random.rand(self.n_samples, self.n_dim),
            "y": np.random.rand(self.n_samples),
        }
        subset = Subset.from_dict(d)
        d2 = subset.__dict__()
        self.assertTrue((d['X'] == d2['X']).all())
        self.assertTrue((d['y'] == d2['y']).all())

    def test_raise_error(self):
        with self.assertRaises(ValueError):
            Subset(
                np.random.rand(self.n_samples + 1, self.n_dim),
                np.random.rand(self.n_samples)
            )

    def test_sampled_ratio(self):
        self.assertEqual(len(self.subset.sampled(1.0)), self.n_samples)
        self.assertEqual(len(self.subset.sampled(0.5)), int(self.n_samples * 0.5))
        self.assertEqual(len(self.subset.sampled(0.333)), int(self.n_samples * 0.333))
        self.assertEqual(len(self.subset.sampled(1.8)), self.n_samples)
        self.assertEqual(len(self.subset.sampled(1808090909.9)), self.n_samples)
        another_ratio = np.random.rand(1)
        self.assertEqual(len(self.subset.sampled(another_ratio)), int(self.n_samples * another_ratio))

    def test_sampled_integer(self):
        self.assertEqual(len(self.subset.sampled(500)), 500)
        self.assertEqual(len(self.subset.sampled(300)), 300)
        self.assertEqual(len(self.subset.sampled(1000000)), 1000)
        self.assertEqual(len(self.subset.sampled(1000)), 1000)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(folder="../dataset")
        self.n_train = len(self.dataset.train)
        self.n_test = len(self.dataset.test)

    def test_sample_size(self):
        self.assertEqual(self.dataset.train.X.shape[0], self.dataset.train.y.shape[0])
        self.assertEqual(self.dataset.test.X.shape[0], self.dataset.test.y.shape[0])

    def test_dimension_size(self):
        self.assertEqual(self.dataset.train.X.shape[1], self.dataset.test.X.shape[1])
        self.assertEqual(len(self.dataset.train.y.shape), 1)
        self.assertEqual(len(self.dataset.test.y.shape), 1)

    def test_type(self):
        self.assertIsInstance(self.dataset.mapping, dict)
        self.assertIsInstance(self.dataset.train.X, np.ndarray)
        self.assertIsInstance(self.dataset.train.y, np.ndarray)
        self.assertIsInstance(self.dataset.test.X, np.ndarray)
        self.assertIsInstance(self.dataset.test.y, np.ndarray)

    def test_sample_train(self):
        self.dataset.sample_train(0.5)
        self.assertEqual(len(self.dataset.train), int(self.n_train * 0.5))
        self.assertEqual(len(self.dataset.train), self.dataset.train_size)

        self.dataset.sample_test(0.3)
        self.assertEqual(len(self.dataset.test), int(self.n_test * 0.3))
        self.assertEqual(len(self.dataset.test), self.dataset.test_size)

        self.dataset.sample_train(0.2)
        self.assertEqual(len(self.dataset.train), int(self.n_train * 0.2))
        self.assertEqual(len(self.dataset.train), self.dataset.train_size)

        self.dataset.sample_test(1.0)
        self.assertEqual(len(self.dataset.test), self.n_test)
        self.assertEqual(len(self.dataset.test), self.dataset.test_size)

        self.dataset.sample_test(3.0)
        self.assertEqual(len(self.dataset.test), self.n_test)
        self.assertEqual(len(self.dataset.test), self.dataset.test_size)

        self.dataset.sample_train(1000)
        self.assertEqual(len(self.dataset.train), 1000)
        self.assertEqual(len(self.dataset.train), self.dataset.train_size)

        self.dataset.sample_train(10000000000)
        self.assertEqual(len(self.dataset.train), self.n_train)
        self.assertEqual(len(self.dataset.train), self.dataset.train_size)

        self.dataset.sample_test(3534)
        self.assertEqual(len(self.dataset.test), 3534)
        self.assertEqual(len(self.dataset.test), self.dataset.test_size)
