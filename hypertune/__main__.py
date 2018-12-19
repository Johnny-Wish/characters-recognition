import os
import argparse
from preprocess import Dataset
from global_utils import dump, JsonMetricQueueWriter
from .search_session import SearchSession
from .reflexive_import import ReflexiveImporter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="dataset", help="folder to hold dataset")
    parser.add_argument("--datafile", default="emnist-byclass.mat", help="filename of dataset")
    parser.add_argument("--train_rate", default=0.01, type=float, help="ratio of training set to be used")
    parser.add_argument("--test_rate", default=0.03, type=float, help="ratio of testing set to be used")
    parser.add_argument("--module", required=True, help="module name for model and param_dist")
    parser.add_argument("--package", default="sklearn_models", help="package to hold the above module")
    parser.add_argument("--model", default="model", help="name of the model variable in the above `module`")
    parser.add_argument("--param", default="parameter_distribution",
                        help="name of the parameter_distribution variable in the above `module`")
    parser.add_argument("--n_iter", default=200, type=int, help="number of iterations to run random searching")
    parser.add_argument("--cv", default=5, type=int, help="number of folds for cross validation while searching")
    parser.add_argument("--outf", default="/output", help="folder to dump search and test results")
    opt = parser.parse_args()

    dataset = Dataset(opt.datafile, opt.dataroot).sample_train(opt.train_rate).sample_test(opt.test_rate)
    importer = ReflexiveImporter(opt.module, opt.package, opt.model, opt.param)
    session = SearchSession(importer.model, importer.param_dist, dataset, opt.n_iter, opt.cv)
    session.report_args()

    # tune (search for) hyper-parameters
    session.fit()
    session.report_best()
    session.report_result()
    dump(session.search_results, os.path.join(opt.outf, "search-results.pkl"))

    # test the best estimator found
    session.test()
    for metric in session.test_result:
        print("testing {}: {}".format(metric, session.test_result[metric]))
    dump(session.test_result, os.path.join(opt.outf, "test-results.pkl"))

    session.search_results.sort_values(by=["mean_test_score"])
    JsonMetricQueueWriter("mean-f1-score", session.search_results.mean_test_score).write()
    JsonMetricQueueWriter("std-f1-score", session.search_results.std_test_score).write()
    JsonMetricQueueWriter("mean-fit-time", session.search_results.mean_fit_time).write()
    JsonMetricQueueWriter("std-fit-time", session.search_results.std_fit_time).write()
    JsonMetricQueueWriter("mean-score-time", session.search_results.mean_score_time).write()
    JsonMetricQueueWriter("std-score-time", session.search_results.std_score_time).write()
