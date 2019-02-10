import os
import argparse
import torch
import torch.nn.functional as F
from pytorch_models.torch_utils import prepend_tag, LossRegister, Checkpointer, EmbedModule
from pytorch_models.base_session import ForwardSession, _SummarySession
from torch.optim import Adam, Optimizer
from preprocess import Dataset
from tensorboardX import SummaryWriter
from reflexive_import import ReflexiveImporter


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="../dataset")
    parser.add_argument("--batch", default=512, type=int)
    parser.add_argument("--model", default="lenet")
    parser.add_argument("--report_period", default=30, type=int)
    parser.add_argument("--param_summarize_period", default=25, type=int)
    parser.add_argument("--max_steps", default=1500, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", default="/output", type=str)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--train_features", action="store_true")
    parser.add_argument("--logdir", default="/output")
    parser.add_argument("--checkpoint", action="store_true")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if opt.cuda or torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    importer = ReflexiveImporter(
        module_name=opt.model,
        var_list=["get_model", "model_args", "model_kwargs", "transformer"],
        package_name="pytorch_models",
    )

    dataset = Dataset(folder=opt.folder, transformer=importer["transformer"])
    print("dataset loaded")

    get_model = importer["get_model"]  # type: callable
    args = importer["model_args"]  # type: tuple
    kwargs = importer["model_kwargs"]  # type: dict
    kwargs.update(dict(
        num_classes=dataset.num_classes,
        pretrained_path=opt.pretrained,
        train_features=opt.train_features,
    ))
    model = get_model(*args, **kwargs)
    print("using model", model)

    writer = SummaryWriter(log_dir=opt.logdir)
    print("logging summaries at", writer.log_dir)

    session = TrainingSession(
        model=model,
        subset=dataset.train,
        batch=opt.batch,
        device=device,
        max_steps=opt.max_steps,
        checkpoint_path=opt.output,
        report_period=opt.report_period,
        param_summarize_period=opt.param_summarize_period,
        summary_writer=writer,
    )
    print("training session instantiated")

    session.epoch(checkpoint=opt.checkpoint)
