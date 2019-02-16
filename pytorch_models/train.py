import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import torch
import torch.nn.functional as F
from pytorch_models.torch_utils import prepend_tag, LossRegister, Checkpointer, EmbedModule, get_dataset_and_model
from pytorch_models.base_session import ForwardSession, _SummarySession
from pytorch_models.pytorch_args import TorchTrainParser, TorchTrainArgs
from torch.optim import Adam, Optimizer
from tensorboardX import SummaryWriter


class TrainingSession(LossRegister, Checkpointer, ForwardSession, _SummarySession):
    def __init__(self, model: EmbedModule, subset, batch, device, max_steps=-1, optim=Adam, checkpoint_path=".",
                 report_period=1, param_summarize_period=25, summary_writer: SummaryWriter = None):
        LossRegister.__init__(self)
        Checkpointer.__init__(self, checkpoint_path=checkpoint_path)
        ForwardSession.__init__(self, model, subset, batch, device, report_period=report_period)
        _SummarySession.__init__(self, param_summarize_period=param_summarize_period, summary_writer=summary_writer)

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


if __name__ == '__main__':
    parser = TorchTrainParser()
    args = TorchTrainArgs(parser)

    device = torch.device("cuda" if args.cuda or torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    dataset, model = get_dataset_and_model(args)

    writer = SummaryWriter(log_dir=args.logdir)
    print("logging summaries at", writer.log_dir)

    session = TrainingSession(
        model=model,
        subset=dataset.train,
        batch=args.batch,
        device=device,
        max_steps=args.max_steps,
        checkpoint_path=args.output,
        report_period=args.report_period,
        param_summarize_period=args.param_summarize_period,
        summary_writer=writer,
    )
    print("training session instantiated")

    session.epoch(checkpoint=args.checkpoint)
