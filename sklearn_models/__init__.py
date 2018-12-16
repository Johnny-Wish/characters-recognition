"""
Created:   Dec 16, 2018
encoding:  UTF-8
Author:    Shuheng Liu
Contact:   wish1104@outlook.com
GitHub:    https://github.com/Johnny-Wish

(c) All Rights Reserved
License: https://choosealicense.com/licenses/gpl-3.0/

Package for storing sklearn models (classifiers)
Hypertune your custom model by creating a module under the package, make sure to include a sklearn estimator (
preferably named `model`) and a dict containing parameters and intervals (preferably named `parameter_distribution).
These two models must be immediately accessible from an external python script. Check `neural_net_adam.py` for example.
"""