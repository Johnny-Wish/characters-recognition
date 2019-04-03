import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import torch
import torch.nn.functional as F
from pytorch_models.torch_utils import prepend_tag, LossRegisterMixin, CheckpointerMixin, EmbedModule
from pytorch_models.base_session import ForwardSession, SummarizerMixin
from pytorch_models.pytorch_args import TorchTrainParser, TorchTrainArgs
from pytorch_models.build_session import BaseSessionBuilder
from torch.optim import Adam, Optimizer
from tensorboardX import SummaryWriter


class TrainingSession(LossRegisterMixin, CheckpointerMixin, ForwardSession, SummarizerMixin):
    def __init__(self, model: EmbedModule, subset, batch, device, max_steps=-1, optim=Adam, checkpoint_path=".",
                 report_period=1, parameter_summary_period=25, embedding_summary_period=False,
                 summary_writer: SummaryWriter = None):
        LossRegisterMixin.__init__(self)
        CheckpointerMixin.__init__(self, checkpoint_path=checkpoint_path)
        ForwardSession.__init__(self, model, subset, batch, device, report_period=report_period)
        SummarizerMixin.__init__(
            self,
            parameter_summary_period=parameter_summary_period,
            embedding_summary_period=embedding_summary_period,
            summary_writer=summary_writer
        )

        self.max_steps = max_steps

        if issubclass(optim, Optimizer):
            # only updates the parameters that require gradients
            self.optimizer = optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        elif isinstance(optim, Optimizer):
            self.optimizer = optim
        else:
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()))

    def epoch(self, force_report=False, ignore_max_steps=False, checkpoint=False):
        for samples_batch in self.loader:
            # report metrics only if the current period ends
            to_report = force_report or ((self._global_step + 1) % self.report_period == 0)
            if not self.step(
                    samples_batch,
                    report=to_report,
                    ignore_max_steps=ignore_max_steps,
                    checkpoint=checkpoint,
            ):
                break
            if self._global_step % self.param_summarize_period == 0:
                self.summarize_parameters()

    def step(self, samples_batch, report=True, ignore_max_steps=False, force_summarize_model=False, checkpoint=False):
        """
        Train the model for 1 step and collect metrics while increasing global_step by 1.
        In the meantime, summarize the model and/or embeddings w.r.t the input if necessary.
        :param samples_batch: a batch yielded when iterating through a torch.utils.data.DataLoader
        :param report: whether to report metrics or not
        :param ignore_max_steps: whether to ignore the internal max_step limit
        :param force_summarize_model: whether to summarize the model regardless of current step
        :param checkpoint: whether to do checkpoint for the model if a new lowest_loss is reached
        :return: bool, indicate whether the step was successful. False indicates that max_step is reached.
        """
        if (not ignore_max_steps) and self._global_step >= self.max_steps > 0:
            print("max_step = {} reached".format(self.max_steps))
            # notify caller that max step is reached
            return False

        # feed forward for one step
        features, labels, logits, tagged_metrics = ForwardSession.step(self, samples_batch, report=report, tag="train")
        # only summarize the model (graph) at the first step unless otherwise specified
        if force_summarize_model or self._global_step == 1:
            self.summarize_model(features)
        # compute the cross entropy loss
        loss = F.cross_entropy(logits, labels)
        # update the lowest loss and summarize the embedding if necessary
        loss_updated = self.update_lowest_loss(loss)
        if checkpoint and loss_updated:
            self.checkpoint()
            self.summarize_embedding(features, labels, step_id=None)
        # report loss if necessary
        loss_dict = {"loss": float(loss)}
        if report:
            self._report_metrics(loss_dict)
        # summarize all metrics including loss
        tagged_metrics.update(prepend_tag(loss_dict, "train"))
        self._summarize_metrics(tagged_metrics)
        # zero the gradient, backprop through the net, and do an optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # notify caller that this step was successful
        return True

    def checkpoint(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self._checkpoint_path, "{}.pth".format(self.model.__class__.__name__))
        )


class TrainingSessionBuilder(BaseSessionBuilder):
    def __init__(self, args: TorchTrainArgs):
        super(TrainingSessionBuilder, self).__init__(args)

        if isinstance(self.args, TorchTrainArgs):
            self.static_model_kwargs.update(
                train_features=self.args.train_features,
            )
        else:
            raise TypeError("require TorchTrainArgs, got {}".format(type(self.args)))

    def _set_session(self):
        super(TrainingSessionBuilder, self)._set_session()

        if self._session is not None:
            return

        self.args: TorchTrainArgs
        self._session = TrainingSession(
            model=self._model,
            subset=self._dataset.train,
            batch=self.args.batch,
            device=self._device,
            max_steps=self.args.max_steps,
            checkpoint_path=self.args.output,
            report_period=self.args.report_period,
            param_summarize_period=self.args.param_summarize_period,
            summary_writer=self._writer,
        )


if __name__ == '__main__':
    parser = TorchTrainParser()
    args = TorchTrainArgs(parser)
    session_builder = TrainingSessionBuilder(args)
    session = session_builder()

    for epoch in range(args.n_epochs):
        print("staring epoch {}".format(epoch + 1))
        session.epoch(checkpoint=args.checkpoint)
