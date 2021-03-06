import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import torch
from pytorch_models.base_session import ForwardSession, SummarizerMixin
from pytorch_models.torch_utils import get_metrics_dict, prepend_tag
from pytorch_models.pytorch_args import TorchInferParser, TorchInferArgs
from pytorch_models.build_session import BaseSessionBuilder
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix


class InferenceSession(ForwardSession, SummarizerMixin):
    def __init__(self, model, subset, batch, device, writer: SummaryWriter = None, summarize=True, report=True,
                 verbose=False):
        ForwardSession.__init__(self, model, subset, batch, device, report_period=report, verbose=verbose)
        SummarizerMixin.__init__(
            self,
            parameter_summary_period=False,
            embedding_summary_period=summarize,
            summary_writer=writer
        )
        self._name = "infer"
        self.reset()

    def reset(self):
        self._labels = self._logits = self._preds = self._metrics = self._confusion_matrix = None
        self.to_summarize_model = True
        return self

    def epoch(self):
        if self._metrics is not None:
            print("a previous epoch has been run; run self.reset() to force re-run")
            return

        self._labels = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._logits = torch.zeros(0, dtype=torch.float64, device=self.device)

        for samples_batch in self.loader:
            labels, logits = self.step(samples_batch)

            self._labels = torch.cat([self._labels, labels], 0)
            self._logits = torch.cat([self._logits, logits], 0)

        self._preds = self._logits.max(1)[1]
        self._confusion_matrix = confusion_matrix(self._labels, self._preds)
        metrics = get_metrics_dict(self._labels, self._preds)
        self._metrics = prepend_tag(metrics, "overall")

        self._report_metrics(self._metrics)
        self._summarize_metrics(self._metrics)

    def step(self, samples_batch):
        with torch.no_grad():  # register no gradients to speed up inference
            features, labels, logits, metrics = ForwardSession.step(self, samples_batch)

        self._summarize_metrics(metrics)

        if self.to_summarize_embedding:
            self.summarize_embedding(features, labels)

        if self.to_summarize_model:
            self.summarize_model(features)
            self.to_summarize_model = False  # the model only needs to be updated once

        return labels, logits

    @property
    def logits(self):
        if self._logits is None:
            self.epoch()
        return self._logits

    @property
    def labels(self):
        if self._labels is None:
            self.epoch()
        return self._labels

    @property
    def preds(self):
        if self._preds is None:
            self.epoch()
        return self._preds

    @property
    def metrics(self):
        if self._metrics is None:
            self.epoch()
        return self._metrics

    @property
    def confusion_matrix(self):
        if self._confusion_matrix is None:
            self.epoch()
        return self._confusion_matrix


class InferenceSessionBuilder(BaseSessionBuilder):
    def _set_session(self):
        super(InferenceSessionBuilder, self)._set_session()

        if self._session is not None:
            return

        self.args: TorchInferArgs
        self._session = InferenceSession(
            model=self._model,
            subset=self._dataset.test,
            batch=self.args.batch,
            device=self._device,
            writer=self._writer,
        )


if __name__ == '__main__':
    parser = TorchInferParser()
    args = TorchInferArgs(parser)
    session_builder = InferenceSessionBuilder(args)
    session = session_builder()
    session.epoch()
