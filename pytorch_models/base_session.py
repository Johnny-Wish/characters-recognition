import torch
from global_utils import flush_json_metrics
from preprocess import Subset
from torch.utils.data import DataLoader
from pytorch_models.torch_utils import prepend_tag, get_metrics_dict, EmbedModule
from tensorboardX import SummaryWriter


class ForwardSession:
    def __init__(self, model, subset, batch, device, report_period=1):
        self.subset = subset  # type: Subset
        self.loader = DataLoader(subset, batch_size=batch, shuffle=True, num_workers=0)
        self.model = model.double().to(device)
        self.report_period = report_period

        self.device = device
        self._global_step = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def n_samples(self):
        return len(self.subset)

    def epoch(self, force_report=False):
        for samples_batch in self.loader:
            # report metrics only at the end of each report_period
            to_report = force_report or ((self._global_step + 1) % self.report_period == 0)
            self.step(samples_batch, report=to_report)

    def step(self, samples_batch, report=True, tag="unspecified-tag"):
        """
        Feed forward 1 step and compute metrics (excluding loss), while increasing global_step by 1
        :param samples_batch: a batch yielded when iterating through a torch.utils.data.DataLoader
        :param report: whether to report metrics or not
        :param tag: tag of the step, usually 'train' or 'infer', used when reporting metrics
        :return: a tuple of (features, labels, logits, tagged_metrics)
        """
        self._global_step += 1

        features, labels = self._split_features_labels(samples_batch)

        # feed forward
        logits = self.model(features)

        with torch.no_grad():
            predictions = logits.max(1)[1]
            metrics = get_metrics_dict(labels, predictions)

        tagged_metrics = prepend_tag(metrics, tag)

        if report:
            self._report_metrics(tagged_metrics)

        return features, labels, logits, tagged_metrics

    def _split_features_labels(self, samples_batch):
        return (
            samples_batch['X'].double().to(self.device),
            samples_batch['y'].long().to(self.device),
        )

    def _report_metrics(self, d: dict):
        flush_json_metrics(d, step=self._global_step)


class _SummarySession:
    """
    This class is extracted from `TrainingSession`, and should not be called directly
    """

    def __init__(self, param_summarize_period=25, summary_writer: SummaryWriter = None):
        self.param_summarize_period = param_summarize_period
        self.writer = summary_writer

    def _summarize_metrics(self, d: dict):
        self._global_step: int
        if self.writer is None:
            return
        for key in d:
            self.writer.add_scalar(key, d[key], self._global_step)

    def summarize_embedding(self, features, labels, step_id="current"):
        """
        write embedding summaries to local disk (if self.writer is not None)
        :param features: a torch.Tensor instance representing image batch
        :param labels: a torch.Tensor instance representing label batch
        :param step_id: int, None, or "current" (default)
            if None, do not specify global_step in add_embedding
            if positive int, specify global_step=step_id
            if "current", specify global_step = current step of training, (i.e., self._global_step)
        :return: None
        """
        self._global_step: int
        self.subset: Subset
        self.model: EmbedModule
        if self.writer is None:
            return
        if step_id == "current":
            step_id = self._global_step
        meta = [self.subset.mapping[int(label)] for label in labels]
        embedding = self.model.embed(features)
        self.writer.add_embedding(
            embedding,
            metadata=meta,
            label_img=features.float(),  # the tensorboardX backend assumes torch.float32 input
            global_step=step_id,
            tag="embedding",
        )

    def summarize_parameters(self):
        self._global_step: int
        self.model: EmbedModule
        if self.writer is None:
            return

        for tag, param in self.model.named_parameters():
            # summarize a parameter only if it requires gradient
            if param.requires_grad:
                self.writer.add_histogram(tag, param, global_step=self._global_step)

    def summarize_model(self, input):
        self.model: EmbedModule
        if self.writer is None:
            return
        # for PyTorch>0.4, tensorboardX must be v1.6 or later for the following line to work
        self.writer.add_graph(self.model, input_to_model=input, verbose=False)
