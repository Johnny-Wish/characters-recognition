from sklearn.svm import SVC

model = SVC()
parameter_distribution = {
    "kernel": ["rbf", "poly", "linear"],
    "class_weight": ["None", "balanced"],
    "decision_function_shape": ["ovo", "ovr"],
    "random_state": [0],
}
