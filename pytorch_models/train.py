import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import argparse
import torch
import torch.nn.functional as F
from pytorch_models.alexnet import get_alexnet
from pytorch_models.lenet import get_lenet
from pytorch_models.torch_utils import prepend_tag, LossRegister, Checkpointer, EmbedModule
from pytorch_models.base_session import ForwardSession
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torch.optim import Adam, Optimizer
from preprocess import Dataset, Reshape
from tensorboardX import SummaryWriter


class TrainingSession(LossRegister, Checkpointer, ForwardSession):
    def __init__(self, model: EmbedModule, train_set, batch, device, max_steps=-1, optim=Adam, checkpoint_path=".",
                 report_period=1, param_summarize_period=25, summary_writer: SummaryWriter = None):
        LossRegister.__init__(self)
        Checkpointer.__init__(self, checkpoint_path=checkpoint_path)
        ForwardSession.__init__(self, model, train_set, batch, device, report_period=report_period)

        self.param_summarize_period = param_summarize_period
        self.max_steps = max_steps

        if issubclass(optim, Optimizer):
            # only updates the parameters that require gradients
            self.optimizer = optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        elif isinstance(optim, Optimizer):
            self.optimizer = optim
        else:
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()))

        self.writer = summary_writer

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
        train the model for 1 step and collect metrics while increasing global_step by 1; summarize the model or
        embeddings if necessary
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

    def _summarize_metrics(self, d: dict):
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
        if self.writer is None:
            return
        if step_id == "current":
            step_id = self._global_step
        meta = [self.train_set.mapping[int(label)] for label in labels]
        embedding = self.model.embed(features)
        self.writer.add_embedding(
            embedding,
            metadata=meta,
            label_img=features.float(),  # the tensorboardX backend assumes torch.float32 input
            global_step=step_id,
            tag="embedding",
        )

    def summarize_parameters(self):
        if self.writer is None:
            return

        for tag, param in self.model.named_parameters():
            # summarize a parameter only if it requires gradient
            if param.requires_grad:
                self.writer.add_histogram(tag, param, global_step=self.global_step)

    def summarize_model(self, input):
        if self.writer is None:
            return
        # for PyTorch>0.4, tensorboardX must be v1.6 or later for the following line to work
        self.writer.add_graph(self.model, input_to_model=input, verbose=False)

    def checkpoint(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self._checkpoint_path, "{}.pth".format(self.model.__class__.__name__))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="../dataset")
    parser.add_argument("--batch", default=512, type=int)
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

    dataset = Dataset(
        folder=opt.folder,
        transformer=Compose([
            Reshape(28, 28, 1),
            ToPILImage(),
            Resize((227, 227)),
            ToTensor(),
        ])
    )
    print("dataset loaded")

    model = get_alexnet(
        num_channels=1,
        num_classes=dataset.num_classes,
        pretrained=True,
        pretrained_path=opt.pretrained if opt.pretrained else None,
        train_features=opt.train_features,
    )
    print("using model", model)

    writer = SummaryWriter(log_dir=opt.logdir)
    print("logging summaries at", writer.log_dir)

    session = TrainingSession(
        model=model,
        train_set=dataset.train,
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
