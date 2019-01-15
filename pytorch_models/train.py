import os
import sys

sub_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.split(sub_dir)[0]
sys.path += [sub_dir, root_dir]

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.alexnet import get_alexnet
from pytorch_models.torch_utils import ArrayTransform
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torch.optim import Adam
from preprocess import Dataset, Reshape
from global_utils import flush_json_metrics
from tensorboardX import SummaryWriter


class TrainingSession:
    def __init__(self, model: nn.Module, train_set, batch, device, max_steps, optim=Adam,
                 report_period=1, param_summarize_period=25, summary_writer: SummaryWriter = None):
        self.loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0)
        self.model = model.double().to(device)
        self.report_period = report_period
        self.param_summarize_period = param_summarize_period
        self.max_steps = max_steps
        if issubclass(optim, Optimizer):
            # only updates the parameters that require gradients
            self.optimizer = optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        elif isinstance(optim, Optimizer):
            self.optimizer = optim
        else:
            print("Abnormal optimizer specified: {}".format(optim))
            self.optimizer = optim

        self.device = device
        self.writer = summary_writer
        self._global_step = 0

    def epoch(self, ignore_max_steps=False):
        for samples_batch in self.loader:
            # report metrics only if the current period ends
            to_report = ((self._global_step + 1) % self.report_period == 0)
            if not self.step(samples_batch, report=to_report, ignore_max_steps=ignore_max_steps):
                break
            if self._global_step % self.param_summarize_period == 0:
                self.summarize_parameters()

    def step(self, samples_batch, report=True, ignore_max_steps=False, force_summarize_model=False):
        if (not ignore_max_steps) and self._global_step >= self.max_steps:
            print("max_step = {} reached".format(self.max_steps))
            return False
        self._global_step += 1

        # split the features and labels
        features = samples_batch['X'].double().to(device)
        labels = samples_batch['y'].long().to(device)

        # only summarize the model (graph) at the first step unless otherwise specified
        if force_summarize_model or self._global_step == 1:
            self.summarize_model(features)

        # feed forward and calculate cross-entropy loss
        logits = self.model(features)
        loss = F.cross_entropy(logits, labels)

        if report or self.writer:  # calculate the accuracy, and possibly other metrics in the future
            with torch.no_grad():
                acc = (logits.max(1)[1] == labels).float().mean()

            # torch.Tensor are not serializable, therefore casting metrics to built-in python classes
            metrics = {
                "train_loss": float(loss),
                "train_accuracy": float(acc),
            }

            self.summarize_metrics(metrics)  # flush summaries of metrics to disk

            if report:  # report the metrics
                self.report_metrics(metrics)

        # zero the gradient, backprop through the net, and do optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True

    @property
    def global_step(self):
        return self._global_step

    def summarize_metrics(self, d: dict):
        if self.writer is None:
            return
        for key in d:
            self.writer.add_scalar(key, d[key], self._global_step)

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

    def report_metrics(self, d: dict):
        flush_json_metrics(d, step=self.global_step)


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
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if opt.cuda or torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    dataset = Dataset(
        folder=opt.folder,
        transformer=Compose([
            Reshape(28, 28),
            ArrayTransform(Resize((227, 227))),
        ])
    )

    model = get_alexnet(
        num_channels=1,
        num_classes=dataset.num_classes,
        pretrained=True,
        pretrained_path=opt.pretrained if opt.pretrained else None,
        train_features=opt.train_features,
    )

    writer = SummaryWriter(log_dir=opt.logdir)

    session = TrainingSession(
        model=model,
        train_set=dataset.train,
        batch=opt.batch,
        device=device,
        max_steps=opt.max_steps,
        report_period=opt.report_period,
        param_summarize_period=opt.param_summarize_period,
        summary_writer=writer,
    )

    session.epoch()
    dump_path = os.path.join(opt.output, "{}-{}-step.pth".format(session.model.__class__.__name__, session.global_step))
    torch.save(session.model.state_dict(), dump_path)
