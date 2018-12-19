# Recognition of Handwritten English Characters

## Task Description

This project is a part of my bachelor graduation design. The task is to correctly identify English characters (uppercases, lowercases, and digits) based on 1-channel visual input. 

## My Objective

Everyone knows this is trivial task using state-of-the-art technology. Yet, my advisor picked this task and I have to finish it. Since there is very little challenge, I decide to take this project as a chance to practice version control.

In the meanwhile, I would like to set up this project in a docker container in the future, which is something I have not tried before.

## Instructions

### Environment Setup

1. Kindly Install python3.6 and the latest versions of scikit-learn and tensorflow
2. Clone the repository to your local machine.
3. Download the 700 MiB [dataset](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip) and unzip it.
4. Rename the unzipped folder to `dataset` and move it under the parent directory of this project.

Step 4 is not necessary if you insist on using a different path or folder name. However, relevant changes must be made to the args passed to the program. Otherwise, the default option is to look into a `dataset/` subfolder under the project root.

An alternative for steps 3 and 4 is to run the command `python dataset_download.py` (make sure you have a python version of 3.x)

### Hypertuning with Sklearn

1. Open terminal and cd the root folder ( `character-recgnition` by default)

2. Create a python module under `sklearn_models/` with your model to be hypertuned and parameters to be randomly searched. Name the model variable `model` and the param dict `parameter_distribution`, and make sure they are accessible by an external script. (See `sklearn_models/neural_net_adam` for example )

3. Run the command, specify other relevant arugments if necessary. (See `python -m hypertune -h` for help)

   ```bash
   python -m hypertune --module <YOUR_MODULE_NAME> --n_iter <NUMBER OF SEARCH ITERATIONS> --cv <NUMBER OF CV FOLDS> --outf <FOLDER FOR DUMPING RESULTS>
   ```

4. `n_iter` randomized search will be made, with the CV/test results being printed to screen and saved to the folder `outf`, where `n_iter` and `outf` are specified in the previous step.

## Unit Tests

All classes (except for essentially trivial ones) are equipped with respective unit test cases in `test/test_xxx.py`, contribution to test cases are encouraged for code robustness.

In particular, the `temp` package is used only for testing purpose of reflexive import (see `hypertune/reflexive_import.py` and `test/test_relfexive_import.py`). No changes should be made to this package.

## ETA

Although the deadline is May 2019, the project is such a trivial task that I plan on finishing it within a few days. However, due to the number of final examinations I have in future weeks, there might be some delay. 

### Update on Dec 19, 2018

Hypertuning interface finished and tested, using 1% of the training data and 3% of testing data, best F1 score at 72.5%. Check out the [logs](https://www.floydhub.com/wish1104/projects/character-recognition/7) and [dumped results](https://www.floydhub.com/wish1104/projects/character-recognition/7/output)

