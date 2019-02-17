import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import torch
from pytorch_models.base_session import ForwardSession, _SummarySession
from pytorch_models.torch_utils import get_metrics_dict, prepend_tag
from pytorch_models.pytorch_args import TorchInferParser, TorchInferArgs
from pytorch_models.build_session import BaseSessionBuilder
from tensorboardX import SummaryWriter


class InferenceSession(ForwardSession, _SummarySession):
    def __init__(self, model, subset, batch, device, writer: SummaryWriter = None):
        ForwardSession.__init__(self, model, subset, batch, device, report_period=1)
        _SummarySession.__init__(self, None, writer)
        self._labels, self._logits, self._preds, self._metrics = None, None, None, None

    def epoch(self, re_run=False, summarize_embedding=False, summarize_model=False):
        if self._metrics is not None and not re_run:
            print("a previous epoch has been run; use re_run=True to force re-run")
            return

        self._labels = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._logits = torch.zeros(0, dtype=torch.float64, device=self.device)

        for samples_batch in self.loader:
            labels, logits = self.step(
                samples_batch,
                summarize_embedding=summarize_embedding,
                summarize_model=summarize_model,
            )
            summarize_model = False  # the model only needs to be summarized once

            self._labels = torch.cat([self._labels, labels], 0)
            self._logits = torch.cat([self._logits, logits], 0)

        self._preds = self._logits.max(1)[1]
        metrics = get_metrics_dict(self._labels, self._preds)
        self._metrics = prepend_tag(metrics, "overall")

        self._report_metrics(self._metrics)
        self._summarize_metrics(self._metrics)

    def step(self, samples_batch, summarize_embedding=False, summarize_model=False):
        with torch.no_grad():  # register no gradients to speed up inference
            features, labels, logits, metrics = ForwardSession.step(
                self,
                samples_batch,
                report=True,
                tag="infer",
            )

        self._summarize_metrics(metrics)

        if summarize_embedding:
            self.summarize_embedding(features, labels)

        if summarize_model:
            self.summarize_model(features)

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
