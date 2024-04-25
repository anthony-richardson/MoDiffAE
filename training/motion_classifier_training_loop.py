import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from tqdm import tqdm

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class MotionClassifierTrainLoop:
    def __init__(self, args, train_platform, model, train_data, validation_data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.weight_decay = args.weight_decay
        self.step = 0
        self.resume_step = 0
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_data) + 1
        print(f"Number of epochs: {self.num_epochs}")
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.train_data):

                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                             cond['y'].items()}

                self.run_step(cond)
                if self.step % self.log_interval == 0:

                    if self.validation_data is not None:
                        self.model.eval()
                        # Added this so that the memory is no issue
                        with torch.no_grad():
                            self.run_validation()
                        self.model.train()

                    for k, v in logger.get_current().name2val.items():
                        if k == 'train_loss':
                            print('step[{}]: train loss[{:0.5f}]'.format(self.step, v))
                        if k == 'validation_loss':
                            print('step[{}]: validation loss[{:0.5f}]'.format(self.step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_validation(self):
        for motion, cond in tqdm(self.validation_data, desc='Validation round'):
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                         cond['y'].items()}
            self.forward(cond, split='validation')

    def run_step(self, cond):
        loss = self.forward(cond, split='train')
        loss.backward()
        self.opt.step()
        self.log_step()

    def forward(self, cond, split):
        self.opt.zero_grad()

        og_motion = cond['y']['original_motion']
        target = cond['y']['labels'].squeeze()

        output = self.model(og_motion)

        loss = F.binary_cross_entropy_with_logits(output, target)

        action_output = output[:, :5]
        action_output = F.softmax(action_output, dim=-1)

        skill_level_output = output[:, 5]
        skill_level_output = torch.sigmoid(skill_level_output)

        action_target = target[:, :5]
        skill_level_target = target[:, 5]

        action_classifications = torch.argmax(action_output, dim=-1)
        action_labels_idxs = torch.argmax(action_target, dim=-1)
        action_correct_predictions = sum(action_classifications == action_labels_idxs).item()
        acc_technique = action_correct_predictions / len(og_motion)

        mae_skill_level = F.l1_loss(skill_level_target, skill_level_output)

        log_loss_dict(
            {
                'bce_w_logits': loss.item(),
                'acc_technique': acc_technique,
                'mae_skill_level': mae_skill_level
            },
            split
        )

        return loss

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"model{self.step:09d}.pt"

    def save(self):
        def save_checkpoint(state_dict):
            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.model.state_dict())

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.step:09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(losses, split):
    for key, values in losses.items():
        logger.logkv_mean(f'{split}_{key}', values)
