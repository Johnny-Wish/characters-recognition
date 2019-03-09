import os
from preprocess import Dataset
from global_utils import dump, JsonMetricQueueWriter
from .search_session import SearchSession
from .sklearn_args import SklearnSessionParser, SklearnSessionArgs
from reflexive_import import ReflexiveImporter

if __name__ == '__main__':
    parser = SklearnSessionParser()
    args = SklearnSessionArgs(parser)

    dataset = Dataset(args.datafile, args.dataroot)
    dataset.sample(args.size)

    importer = ReflexiveImporter(
        module_name=args.model,
        var_list=["model", "parameter_distribution"],
        alias_list=["model", "param"]
    )
    session = SearchSession(importer["model"], importer["param"], dataset, args.n_iter, args.cv)
    session.report_args()

    # tune (search for) hyper-parameters
    session.fit()
    session.report_best()
    session.report_result()
    dump(session.search_results, os.path.join(args.output, "search-results.pkl"))

    # test the best estimator found
    session.test()
    for metric in session.test_result:
        print("testing {}: {}".format(metric, session.test_result[metric]))
    dump(session.test_result, os.path.join(args.output, "test-results.pkl"))

    session.search_results.sort_values(by=["mean_test_score"])
    JsonMetricQueueWriter("mean-f1-score", session.search_results.mean_test_score).write()
    JsonMetricQueueWriter("std-f1-score", session.search_results.std_test_score).write()
    JsonMetricQueueWriter("mean-fit-time", session.search_results.mean_fit_time).write()
    JsonMetricQueueWriter("std-fit-time", session.search_results.std_fit_time).write()
    JsonMetricQueueWriter("mean-score-time", session.search_results.mean_score_time).write()
    JsonMetricQueueWriter("std-score-time", session.search_results.std_score_time).write()
