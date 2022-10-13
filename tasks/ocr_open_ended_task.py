import torch

from utils.logging_utils import setup_logger
from tasks.open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
import evaluation

import numpy as np
from tqdm import tqdm
import itertools

logger = setup_logger()

@META_TASK.register()
class OcrOpenEndedTask(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        loss = self.model(items).contiguous()
                    
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), items.ocr_tokens, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                loss = self.model(items)
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def train_scst(self):
        # design especially for self-critical sequential learning
        running_reward = .0
        running_reward_baseline = .0

        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with self-critical learning' % self.epoch, unit='it', total=len(self.train_dict_dataloader)) as pbar:
            for it, items in enumerate(self.train_dict_dataloader):
                items = items.to(self.device)
                outs, log_probs = self.model.beam_search(items, batch_size=items.batch_size, 
                                                            beam_size=self.training_beam_size, out_size=self.training_beam_size)
                
                self.optim.zero_grad()

                # Rewards
                bs = items.question_tokens.shape[0]
                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), items.ocr_tokens, join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.training_beam_size for a in answers_gt)))
                gens = {f"{idx}": [answer_gen, ] for idx, answer_gen in enumerate(answers_gen)}
                gts = {f"{idx}": answer_gt for idx, answer_gt in enumerate(answers_gt)}
                reward = self.train_cider.compute_score(gts, gens)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(self.device).view(bs, self.training_beam_size)
                reward_baseline = torch.mean(reward, dim=-1, keepdim=True)
                loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

                loss = loss.mean()
                loss.backward()
                self.optim.step()

                running_loss += loss.item()
                running_reward += reward.mean().item()
                running_reward_baseline += reward_baseline.mean().item()
                pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                                reward_baseline=running_reward_baseline / (it + 1))
                pbar.update()
