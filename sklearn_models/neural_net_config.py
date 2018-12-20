random_seed=0
_sizes1 = [(500 * (n + 1),) for n in range(4)]
_sizes2 = [(500 * (n + 1), 500 * (n + 1)) for n in range(4)]
_sizes3 = [(500 * (n + 1), 250 * (n + 1)) for n in range(4)]
sizes = _sizes1 + _sizes2 + _sizes3

if __name__ == '__main__':
    variables = vars().copy()
    print("MLP configurations: ")
    for name in variables:
        if name.startswith("__") or name.endswith("__"):
            continue
        print("\t{}: {}\n\t {}".format(name, type(variables[name]), variables[name]))
