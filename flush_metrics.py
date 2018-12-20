from global_utils import JsonMetricQueueWriter as Writer
import pickle
import pandas as pd


def flush_metrics(results: pd.DataFrame, time_interval=0):
    results.sort_values(by=["mean_test_score"], inplace=True)

    Writer("mean-f1-score", results.mean_test_score, time_interval).write()
    Writer("std-f1-score", results.std_test_score, time_interval).write()
    Writer("mean-fit-time", results.mean_fit_time, time_interval).write()
    Writer("std-fit-time", results.std_fit_time, time_interval).write()
    Writer("mean-score-time", results.mean_score_time, time_interval).write()
    Writer("std-score-time", results.std_score_time, time_interval).write()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/results/search-results.pkl")
    parser.add_argument("--interval", default=0.2, type=float)
    opt = parser.parse_args()

    with open(opt.path, "rb") as f:
        search_results = pickle.load(f)  # type: pd.DataFrame

    flush_metrics(search_results, opt.interval)

