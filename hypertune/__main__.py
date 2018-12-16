import argparse
from preprocess import Dataset
from .search_session import SearchSession
from .reflexive_import import ReflexiveImporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="dataset", help="folder to hold dataset")
    parser.add_argument("--datafile", default="emnist-byclass.mat", help="filename of dataset")
    parser.add_argument("--module", required=True, help="module name for model and param_dist")
    parser.add_argument("--package", default="sklearn_models", help="package to hold the above module")
    parser.add_argument("--model", default="model", help="name of the model variable in the above `module`")
    parser.add_argument("--param", default="parameter_distribution",
                        help="name of the parameter_distribution variable in the above `module`")
    parser.add_argument("--n_iter", default=200, type=int, help="number of iterations to run random searching")
    parser.add_argument("--cv", default=5, type=int, help="number of folds for cross validation while searching")
    opt = parser.parse_args()

    dataset = Dataset(opt.datafile, opt.dataroot)
    importer = ReflexiveImporter(opt.module, opt.package, opt.model, opt.param)
    session = SearchSession(importer.model, importer.param_dist, dataset, opt.n_iter, opt.cv)

    # tune (search for) hyper-parameters
    session.fit()
    session.report_best()
    session.report_result()

    # test the best estimator found
    session.test()
    for metric in session.test_result:
        print("testing {}: {}".format(metric, session.test_result[metric]))