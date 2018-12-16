import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from preprocess import Dataset
from .cross_val_utils import print_search_result


class SearchSession:
    def __init__(self, model, param_dist, dataset: Dataset, n_iter=200, cv=5):
        self.dataset = dataset
        self.searcher = RandomizedSearchCV(model, param_dist, n_iter=n_iter, scoring="f1", cv=cv, verbose=1,
                                           random_state=0, return_train_score=False)
        # default value for search result
        self._result = None
        # dafault values for test result of the best estimator
        self._acc, self._pre, self._rec, self._f1, self._supp = None, None, None, None, None
        # default value for whether the model is fitted
        self._fitted = False
        # default value for whether the model is tested
        self._tested = False

    def fit(self):
        self.searcher.fit(self.dataset.train.X, self.dataset.train.y)
        self._fitted = True
        self._result = pd.DataFrame(self.searcher.cv_results_)

    @property
    def fitted(self):
        return self._fitted

    @property
    def tested(self):
        return self._tested

    def report_best(self, best=True):
        if not self._fitted:
            print("searcher not fitted yet")
            self.fit()

        print(
            "The best estimator: ",
            self.searcher.best_index_,
            self.searcher.best_score_,
            self.searcher.best_params_,
            self.searcher.best_estimator_,
            sep="\n"
        )

    def report_result(self):
        if not self._fitted:
            print("searcher not fitted yet")
            self.fit()

        print_search_result(self._result, n=-1)

    def test(self):
        if not self._fitted:
            print("searcher not fitted yet")
            self.fit()

        self.searcher.best_estimator_.fit(self.dataset.train.X, self.dataset.train.y)  # refitting might be redundant

        y_true = self.dataset.test.y
        y_pred = self.searcher.best_estimator_.predict(self.dataset.test.X)
        self._acc = accuracy_score(y_true, y_pred)
        self._pre, self._rec, self._f1, self._supp = precision_recall_fscore_support(y_true, y_pred, average="macro")

    @property
    def acc(self):
        if not self._tested:
            print("searcher not tested yet")
            self.test()

        return self._acc

    @property
    def pre(self):
        if not self._tested:
            print("searcher not tested yet")
            self.test()

        return self._pre

    @property
    def rec(self):
        if not self._tested:
            print("searcher not tested yet")
            self.test()

        return self._rec

    @property
    def f1(self):
        if not self._tested:
            print("searcher not tested yet")
            self.test()

        return self._f1

    @property
    def supp(self):
        if not self._tested:
            print("searcher not tested yet")
            self.test()

        return self._supp