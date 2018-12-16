import heapq
from sklearn.model_selection._search import BaseSearchCV


def print_search_result(result, n):
    if isinstance(result, BaseSearchCV):
        result = result.cv_results_

    scores = result['mean_test_score']
    params = result['params']

    if n < 0:
        n = len(scores)

    print("Cross Validation result in descending order: (totalling {} trials)".format(n))
    for rank, candidate, in enumerate(heapq.nlargest(n, zip(scores, params), key=lambda tup: tup[0])):
        print("rank {}, score = {}\n hyperparams = {}".format(rank + 1, *candidate))
