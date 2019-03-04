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
