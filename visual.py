import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

    def plot(self, replot=False, **kwargs):
        """
        an interface to be called publicly when doing specific plotting.
        DO NOT override this method directly, override the protected `_plot()` instead
        :return: the instance itself
        """
        self.activate()  # activate self.fig before doing the actually plotting
        if not self.plotted:
            self._plot(**kwargs)
        elif replot:
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
        shows the figure, runs self.plot() automatically
        :param warn: refer to matplotlib.pyplot.figure.Figure.show()
        :return: None
        """
        self.plot(replot=False)
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
        xticklabels = kwargs.pop("xticklabels", False)
        yticklabels = kwargs.pop("yticklabels", False)
        square = kwargs.pop("square", True)
        sns.heatmap(
            self.data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            annot_kws=annot_kws,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            square=square,
            **kwargs
        )
        plt.title(self.default_save_path.replace("-", " "))


def get_rectangular_layout(n_items, n_rows=None, n_cols=None):
    if n_items <= 0:
        raise ValueError("n_items must be positive, got {}".format(n_items))

    ceil = lambda x: int(np.ceil(x))

    if n_rows is None and n_cols is None:
        n_rows = ceil(np.sqrt(n_items))
        n_cols = ceil(n_items / n_rows)
    elif n_rows is None:
        if n_cols > 0:
            n_rows = ceil(n_items / n_cols)
        else:
            raise ValueError("n_cols must be positive, got {}".format(n_cols))
    elif n_cols is None:
        if n_rows > 0:
            n_cols = ceil(n_items / n_rows)
        else:
            raise ValueError("n_rows must be positive, got {}".format(n_rows))
    elif n_rows * n_cols < n_items:
        raise ValueError("n_rows * n_cols < n_items : {} * {} < {}".format(n_rows, n_cols, n_items))

    return n_rows, n_cols


class DataChunkVisualizer(BaseVisualizer):
    def __init__(self, data_chuck, n_rows=None, n_cols=None):
        super(DataChunkVisualizer, self).__init__()

        self.chunk = [parse_data_point(data_point) for data_point in data_chuck]
        if not self.chunk:
            raise ValueError("data chuck is empty")

        self.default_save_path = "Data-chunk-of-size-{}".format(self.chunk_size)
        self.n_rows, self.n_cols = get_rectangular_layout(self.chunk_size, n_rows, n_cols)

    @property
    def chunk_size(self):
        return len(self.chunk)

    def _plot(self, **kwargs):
        self.set_width(self.n_cols)
        self.set_height(self.n_rows + 0.25)
        self.fig.suptitle(self.default_save_path.replace("-", " "))

        for idx, data_point in enumerate(self.chunk):
            plt.subplot(self.n_rows, self.n_cols, idx + 1)
            data, label = data_point
            self._subplot(data, label, **kwargs)

    def _subplot(self, data, label, **kwargs):
        annot = kwargs.pop("annot", False)
        cmap = kwargs.pop("cmap", "YlOrRd")
        xticklabels = kwargs.pop("xtickslabels", False)
        yticklabels = kwargs.pop("ytickslabels", False)
        square = kwargs.pop("square", True)
        cbar = kwargs.pop("cbar", False)

        sns.heatmap(
            data,
            annot=annot,
            cmap=cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar=cbar,
            square=square,
            **kwargs
        )
        plt.title(label)


class ConfusionMatrixVisualizer(BaseVisualizer):
    def __init__(self, y_true=None, y_pred=None, labels=None, matrix=None, n_samples=None):
        super(ConfusionMatrixVisualizer, self).__init__()

        if y_true is None and y_pred is not None:
            raise ValueError("y_true and y_pred must be specified both or neither")
        elif y_true is not None and y_pred is None:
            raise ValueError("y_true and y_pred must be specified both or neither")
        elif y_true is None and y_pred is None and matrix is None:
            raise ValueError("matrix must be specified or y_true and y_pred must be specified")
        elif y_true is not None and y_pred is not None and matrix is not None:
            raise ValueError("matrix, and y_true/pred cannot be specified together")

        self.n_samples = n_samples
        if matrix is None:
            self.n_samples = len(y_true)
            self.matrix = confusion_matrix(y_true, y_pred, labels=labels)
        else:
            self.matrix = matrix
        self.n_classes = self.matrix.shape[0]

        self.default_save_path = "Confusion-Matrix-with-{}-samples-from-{}-classes".format(self.n_samples,
                                                                                           self.n_classes)

    def _plot(self, width=15, height=15, **kwargs):
        self.set_width(width).set_height(height)
        self.fig.suptitle(self.default_save_path.replace("-", " "), fontsize=30)

        annot = kwargs.pop("annot", False)
        fmt = kwargs.pop("fmt", ".2g")
        cmap = kwargs.pop("cmap", "Blues")
        annot_kws = kwargs.pop("annot_kws", dict(fontsize=6))
        xticklabels = kwargs.pop("xticklabels", False)
        yticklabels = kwargs.pop("yticklabels", False)
        square = kwargs.pop("square", True)

        sns.heatmap(
            self.matrix,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            square=square,
            **kwargs
        )

        plt.xlabel('Predicted label', fontsize=30)
        plt.ylabel('True label', fontsize=30)
        plt.tight_layout()
