import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class BaseVisualizer:
    def __init__(self):
        """
        an instance of BaseVisualizer for plotting, showing, and saving images
        """
        self.fig = plt.figure()
        self.num = self.fig.number
        self.default_save_path = "default"
        self.plotted = False

    def activate(self):
        """
        activate the figure attached to this instance
        :return: None
        """
        plt.figure(self.num)

    def resize(self, size0, size1=None):
        """
        set the width and height of the figure attached to this instance
        :param size0: width, or a tuple of (width, height) in inches
        :param size1: height, or None (if size0 is a tuple)
        :return: self
        """
        if size1 is None:
            size0, size1 = size0
        else:
            size0, size1 = size0, size1
        return self.set_width(size0).set_height(size1)

    def set_width(self, width):
        """
        set the width of the figure attached to this instance
        :param width: width in inches
        :return: self
        """
        if width is not None:
            self.fig.set_figwidth(width)
        return self

    def set_height(self, height):
        """
        set the height of the figure attached to this instance
        :param height: height in inches
        :return: self
        """
        if height is not None:
            self.fig.set_figheight(height)
        return self

    def plot(self, force_replot=False, **kwargs):
        """
        an interface to be called publicly when doing specific plotting.
        DO NOT override this method directly, override the protected `_plot()` instead
        :return: the instance itself
        """
        self.activate()  # activate self.fig before doing the actually plotting
        if not self.plotted:
            self._plot(**kwargs)
        elif force_replot:
            self._plot(**kwargs)
        else:
            print("Plotting aborted: figure {} has been previously plotted".format(self.num))
        self.plotted = True
        return self

    def _plot(self, **kwargs):
        """
        a method to be overridden for specific plotting purposes
        :return: None
        """
        raise NotImplementedError("{}.plot is not implemented".format(self.__class__.__name__))

    def show(self, warn=True):
        """
        shows the figure, must be run before another fig is shown
        :param warn: refer to matplotlib.pyplot.figure.Figure.show()
        :return: None
        """
        self.fig.show(warn=warn)

    def save(self, path=None, **kwargs):
        """
        save the current figure on hard drive
        :param path: path to save the model, if None, self.default_save_path will be used
        :param kwargs: additional keyword arguments used in Figure.savefig()
        :return: None
        """
        if path is None:
            path = self.default_save_path

        self.fig.savefig(path, **kwargs)


def parse_data_point(data_point):
    """
    parse and obtain the value and label of a data point of type list, tuple, or dict
    :param data_point: list or tuple of length 1, 2, or 3; or a dict with keys "X", "y", and "mapping"
    :return: data, label
    """
    if isinstance(data_point, (tuple, list)):
        if len(data_point) == 1:
            data, label, mapping = data_point[0], None, None
        elif len(data_point) == 2:
            data, label = data_point
            mapping = None
        elif len(data_point) == 3:
            data, label, mapping = data_point
        else:
            raise ValueError("data_point must have length 1, 2, or 3; got {}".format(len(data_point)))
    elif isinstance(data_point, dict):
        data = data_point.get("X", None)
        label = data_point.get("y", None)
        mapping = data_point.get("mapping", None)
    else:
        raise ValueError("data must be a list, tuple, or dict")

    if not isinstance(data, np.ndarray):
        raise ValueError("data is not an array, got {}".format(type(data)))

    if isinstance(mapping, dict):
        label = mapping[label]
    elif callable(mapping):
        label = mapping(label)
    elif mapping is not None:
        raise TypeError("Unrecognized label mapping {} of type {}", format(mapping, type(mapping)))

    return data, label


class DataPointVisualizer(BaseVisualizer):
    def __init__(self, data_point):
        super(DataPointVisualizer, self).__init__()

        self.data, self.label = parse_data_point(data_point)
        self.default_save_path = str("Data-point-of-label-{}".format(self.label))

    def _plot(self, width=None, height=None, **kwargs):
        self.set_width(width).set_height(height)
        annot = kwargs.pop("annot", True)
        fmt = kwargs.pop("fmt", "")
        cmap = kwargs.pop("cmap", "YlOrRd")
        annot_kws = kwargs.pop("annot_kws", dict(fontsize=6))
        sns.heatmap(
            self.data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            annot_kws=annot_kws,
            xticklabels=False,
            yticklabels=False,
            **kwargs
        )
        plt.title(self.default_save_path.replace("-", " "))
        plt.title(self.default_save_path.replace("-", " "))