import torch
from global_utils import flush_json_metrics
from preprocess import Subset
from torch.utils.data import DataLoader
from pytorch_models.torch_utils import prepend_tag, get_metrics_dict
from tensorboardX import SummaryWriter


class ForwardSession:
    def __init__(self, model, train_set, batch, device, report_period=1):
        self.train_set = train_set  # type: Subset
        self.loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0)
        self.model = model.double().to(device)
        self.report_period = report_period

        self.device = device
        self._global_step = 0

    @property
    def global_step(self):
        return self._global_step

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

