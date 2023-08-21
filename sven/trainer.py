import os
import abc
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from sven.model import save_model, parallelize_model, load_model
from sven.dataset import PrefixDataset, TextPromptDataset
from sven.utils import set_seed

class TrainerBase:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.dataset = None
        self.input_device = None

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def load_dataset(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def step(self, batch):
        raise NotImplementedError()

    def save(self, path, step, epoch, optimizer, scheduler):
        if not os.path.exists(path):
            os.makedirs(path)
        save_model(self.model, path, self.args)
        self.tokenizer.save_pretrained(path)
        step_file = os.path.join(path, 'step_file.txt')
        with open(step_file, 'w') as f:
            f.write(str(step)+'\n')
        epoch_file = os.path.join(path, 'epoch_file.txt')
        with open(epoch_file, 'w') as f:
            f.write(str(epoch)+'\n')
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))

    def add_to_loss_dict(self, acc_loss_dict, loss_dict):
        for key, val in loss_dict.items():
            if key not in acc_loss_dict:
                acc_loss_dict[key] = 0.0
            acc_loss_dict[key] += val

    def report_loss_dict(self, loss_dict, steps):
        ss = []
        for key, val in loss_dict.items():
            if key == 'kl_loss':
                r = 8
            else:
                r = 4
            ss.append(f'{key}: {round(val/steps, r)}')
        return ', '.join(ss)

    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f'Training args {self.args}')

        batch_size = 1
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                num_training_steps=total_steps)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)
        self.args.logger.info('  Fraction of trainable parameters = %s', str(round(num_trainable_params/num_params*100, 4)))

        global_step, acc_loss_dict = 0, OrderedDict()
        set_seed(self.args)
        self.model.train()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)
                if self.args.grad_acc_steps > 1:
                    loss = loss / self.args.grad_acc_steps
                    for key in loss_dict:
                        loss_dict[key] = loss_dict[key] / self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.add_to_loss_dict(acc_loss_dict, loss_dict)

                if (step+1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()  
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        reported_loss = self.report_loss_dict(acc_loss_dict, self.args.logging_steps)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s', idx+1, int(self.args.num_train_epochs), global_step, total_steps, reported_loss)
                        acc_loss_dict.clear()

            if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    reported_eval_loss = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx+1, reported_eval_loss)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir, global_step, idx+1, None, None)
                self.save(last_output_dir, global_step, idx+1, None, None)

        if (idx+1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                reported_eval_loss = self.do_eval()
            self.args.logger.info('final eval loss: %s', reported_eval_loss)
            output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.save(output_dir, global_step, idx+1, None, None)
            self.save(last_output_dir, global_step, self.args.num_train_epochs, None, None)

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = OrderedDict()
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            self.add_to_loss_dict(acc_loss_dict, loss_dict)
        return self.report_loss_dict(acc_loss_dict, len(val_dataloader))

def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)
    else:
        past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'cross_entropy':
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_type == 'nll':
        loss_fct = torch.nn.NLLLoss(reduction='none')
    elif loss_type == 'kl':
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
    else:
        assert False

    loss = loss_fct(inputs, targets)
    if loss_type == 'kl':
        loss = loss.sum(dim=1)
    loss = loss[weights != 0]
    return loss.mean()

class PrefixTrainer(TrainerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.pretrain_dir, True, self.args)
        for n, p in self.model.named_parameters():
            if n.startswith('prefix_params'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.model.train()

    def load_dataset(self):
        self.dataset = PrefixDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = PrefixDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)

        correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs, control_ids)
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio
        return_dict['lm_loss'] = lm_loss.item()

        if self.args.contrastive_loss_ratio != 0 or self.args.kl_loss_ratio != 0:
            incorrect_control_ids = -1 * (control_ids - 1)
            incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs, incorrect_control_ids)

            contrastive_loss = 0
            if self.args.contrastive_loss_ratio != 0:
                contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
                contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
                contrastive_log_probs = torch.log(contrastive_probs)
                contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64).to(self.input_device)
                contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
                contrastive_loss *= self.args.contrastive_loss_ratio / 100
                return_dict['contrastive_loss'] = contrastive_loss.item()

            kl_loss = 0
            if self.args.kl_loss_ratio != 0:
                correct_log_probs = F.log_softmax(correct_logits, dim=-1)
                self.model.eval()
                with torch.no_grad():
                    ref_logits, _ = get_logits_from_lm(self.model, inputs, None)
                self.model.train()
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                kl_loss += token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1-shift_weights)
                incorrect_log_probs = F.log_softmax(incorrect_logits, dim=-1)
                kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1-shift_weights)
                kl_loss = kl_loss * self.args.kl_loss_ratio / 1000
                return_dict['kl_loss'] = kl_loss.item()

        loss = lm_loss + contrastive_loss + kl_loss
        return_dict['loss'] = loss.item()
        return loss, return_dict

class TextPromptTrainer(TrainerBase):

    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.pretrain_dir, True, self.args)
        self.model.train()

    def load_dataset(self):
        self.dataset = TextPromptDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = TextPromptDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        inputs, labels= batch
        inputs = inputs.to(self.input_device)
        labels = labels.to(self.input_device)
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        return loss, {'loss': loss.item()}
