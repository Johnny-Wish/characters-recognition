# Recognition of Handwritten English Characters

## Task Description

This project is a part of my bachelor graduation design. The task is to correctly identify English characters (uppercases, lowercases, and digits) based on 1-channel visual input. 

## My Objectives

Everyone knows this is trivial task for latest techniques in computer vision. Yet, my advisor picked this task and I have to finish it. Since there is very little challenge, I decide to take this project as a chance to practice version control.

In the meanwhile, I would like to set up this project in a docker container in the future, which is something I have not tried before.

## Instructions

### Environment Setup

1. Kindly install Python 3.6 and the latest versions of `scikit-learn` , `PyTorch`, `tensorboard`, and `tensorboardX`. Note that `tensorboardX` must be v1.6 or later for `PyTorch` >= 0.4 
2. `git clone` the repository to your local machine.
3. Download the 700 MiB dataset [here](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip) and unzip it.
4. Rename the unzipped folder to `dataset` and move it under the parent directory of this project.

Step 4 is not necessary if you insist on using a different path or folder name. However, relevant changes must be made to the args `--dataroot` and `--datafike` passed to command line. Otherwise, the default option is to look into a `dataset/` subfolder under the project root.

An alternative for steps 3 and 4 is to run the command `python dataset_download.py` (make sure you have a python version of 3.x)

### Hypertuning with Sklearn

1. Open terminal and cd the root folder ( `character-recognition` by default)

2. Create a python module under `sklearn_models/` with your model to be hypertuned and parameters to be randomly searched. Name the model variable `model` and the param dict `parameter_distribution`, and make sure they are accessible by an external script. (See `sklearn_models/neural_net_adam` for example )

3. Run the following command, specify other relevant arugments if necessary. (Run `python -m hypertune -h` for help)

   ```bash
   usage: __main__.py [-h] --model MODEL [--dataroot DATAROOT]
                      [--datafile DATAFILE] [--labels LABELS] [--balance]
                      [--size SIZE] [--output OUTPUT] [--verbose]
                      [--n_iter N_ITER] [--cv CV]
   
   optional arguments:
     -h, --help           show this help message and exit
     --model MODEL        python module (with package prefix) for dynamic model
                          import
     --dataroot DATAROOT  folder to hold dataset
     --datafile DATAFILE  filename of dataset
     --labels LABELS      specify certain labels to be used (all labels by
                          default)
     --balance            whether to down-sample to balance classes
     --size SIZE          float, size of dataset to be used (1.0 by default)
     --output OUTPUT      folder to store trained parameters
     --verbose            verbose tag
     --n_iter N_ITER      number of iterations for random searching
     --cv CV              number of folds for cross validation
   ```

4. `n_iter` randomized search will be made, with the CV/test results being printed to screen and saved to the folder `outf`, where `n_iter` and `outf` are specified in the previous step.

### PyTorch

#### Training

1. Open terminal and cd the root folder ( `character-recgnition` by default)

2. Use existent module files under `pytorch_models/` (`alexnet.py` and `lenet.py`) or create a new one.

3. Run the following command, specify other relevant args if necessary. (Run `python pytorch_models/train.py -h` for help):

   ```bash
   usage: train.py [-h] [--dataroot DATAROOT] [--datafile DATAFILE]
                   [--labels LABELS] [--balance] [--size SIZE] [--batch BATCH]
                   --model MODEL [--pretrained PRETRAINED] [--output OUTPUT]
                   [--verbose] [--report_period REPORT_PERIOD] [--cuda]
                   [--logdir LOGDIR]
                   [--param_summarize_period PARAM_SUMMARIZE_PERIOD]
                   [--max_steps MAX_STEPS] [--train_features] [--checkpoint]
                   [--n_epochs N_EPOCHS]
   
   optional arguments:
     -h, --help            show this help message and exit
     --dataroot DATAROOT   folder to hold dataset
     --datafile DATAFILE   filename of dataset
     --labels LABELS       specify certain labels to be used (all labels by
                           default)
     --balance             whether to down-sample to balance classes
     --size SIZE           float, size of dataset to be used (1.0 by default)
     --batch BATCH         size of mini-batch in this dataset
     --model MODEL         python module (with package prefix) for dynamic model
                           import
     --pretrained PRETRAINED
                           pretrained path to be passed to the model getter
     --output OUTPUT       folder to store trained parameters
     --verbose             verbose tag
     --report_period REPORT_PERIOD
                           how frequently to report session metrics, in number of
                           steps (mini-batches)
     --cuda                whether to use cuda (if available)
     --logdir LOGDIR       folder to store tensorboard summaries
     --param_summarize_period PARAM_SUMMARIZE_PERIOD
                           how frequently to summarize parameter distributions,
                           in number of steps (mini-batches)
     --max_steps MAX_STEPS
                           max number of steps to run before training is
                           terminated, disabled by default
     --train_features      to train the feature layers (disabled by default)
     --checkpoint          do checkpoint for the model (disabled by default)
     --n_epochs N_EPOCHS   number of epochs to run
   ```

#### Inference

1. Redo the above steps 1, 2, and 3. Just remember to change `pytorch_models/train.py` to `pytorch_models/infer.py`

   ```bash
   usage: infer.py [-h] [--dataroot DATAROOT] [--datafile DATAFILE]
                   [--labels LABELS] [--balance] [--size SIZE] [--batch BATCH]
                   --model MODEL [--pretrained PRETRAINED] [--output OUTPUT]
                   [--verbose] [--report_period REPORT_PERIOD] [--cuda]
                   [--logdir LOGDIR]
   
   optional arguments:
     -h, --help            show this help message and exit
     --dataroot DATAROOT   folder to hold dataset
     --datafile DATAFILE   filename of dataset
     --labels LABELS       specify certain labels to be used (all labels by
                           default)
     --balance             whether to down-sample to balance classes
     --size SIZE           float, size of dataset to be used (1.0 by default)
     --batch BATCH         size of mini-batch in this dataset
     --model MODEL         python module (with package prefix) for dynamic model
                           import
     --pretrained PRETRAINED
                           pretrained path to be passed to the model getter
     --output OUTPUT       folder to store trained parameters
     --verbose             verbose tag
     --report_period REPORT_PERIOD
                           how frequently to report session metrics, in number of
                           steps (mini-batches)
     --cuda                whether to use cuda (if available)
     --logdir LOGDIR       folder to store tensorboard summaries
   ```

   

## Unit Tests

All classes (except for essentially trivial ones) are equipped with respective unit test cases in `test/test_xxx.py`. Contributions to test cases are encouraged for code robustness.

In particular, the `temp` package is used only for testing purposes of reflexive import (see `hypertune/reflexive_import.py` and `test/test_relfexive_import.py`). No modification should be made to this package.

## ETA and Updates

Although the deadline is May 2019, the project is such a trivial task that I plan on finishing it within a few days. However, due to the number of final examinations I have in future weeks, there might be some delay. 

### Update on Dec 19, 2018

Hypertuning interface finished and tested, using 1% of the training data and 3% of testing data, best F1 score at 72.5%. Check out the [logs](https://www.floydhub.com/wish1104/projects/character-recognition/7), [dumped results](https://www.floydhub.com/wish1104/projects/character-recognition/7/output), and [metric graphs](https://www.floydhub.com/wish1104/projects/character-recognition/12).

### Update on Jan 10, 2019

I have been trying to implement the model in TensorFlow, the API of which is a mess. Check out the `feature/tf` branch for more details. Now, I am rethinking about my choice of framework and implementing models using PyTorch instead.

### Update on Jan 12, 2019

With a custom architected AlexNet model, the latest classification accuracy on 62-label dataset has improved to over 80%, F-score unknown.

### Update on Jan 13, 2019

A TensorboardX visualization tool is added for metric and parameter visualization. It seems the pre-trained convolution layers of alexnet is not updated at all. This is possibly due to gradient vanishment, but more likely, the fact that pretrained parameters are already good enough at capturing image features.

### Update on Jan 27, 2019

A simpler version of CNN, referred to as LeNet has been tested. Performance of this network (85%+ accuracy and F-score) is even better than pretrained AlexNet, and the training time is greatly reduced. See [this job](https://www.floydhub.com/wish1104/projects/character-recognition/72), [this job](https://www.floydhub.com/wish1104/projects/character-recognition/72) and [this job](https://www.floydhub.com/wish1104/projects/character-recognition/72) for training logs and outputs.

### Update on Feb 10, 2019

Some API changes has been made:

1. Abstracted a new `ForwardSession` class and a new `_SummarySession` class from `TrainingSession`.
2. Refactored the class`ReflexiveImporter` for general purposes.
3. Used reflexive API for importing models and corresponding data transformer.

Next step is to implement an `InferenceSession` extending `ForwardSession` and `_SummarySession`, and tools for visualizing model performances.

### Update on Mar 10, 2019

1. Implemented`InferenceSession`  (weeks ago)
2. Statically registered command line args passed to training, inferring, and hyper-tuning.
3. Implemented `DataPointVisualizer`,  `DataChunkVisualizer`, and `ConfusionMatrixVisualizer`.
4. Implemented functionalities of balancing, filtering by label, and random sampling for `Subset` and `Dataset`

