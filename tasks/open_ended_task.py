import torch
from torch.optim import Adam

from utils.logging_utils import setup_logger
from data_utils.vocab import Vocab
from data_utils.dataset import FeatureDataset, DictionaryDataset
from .base_task import BaseTask
from builders.task_builder import META_TASK
import evaluation
from evaluation import Cider

import os
import numpy as np
from tqdm import tqdm
import itertools
from shutil import copyfile

logger = setup_logger()

@META_TASK.register()
class OpenEndedTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.get_scores = config.TRAINING.GET_SCORES
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        self.train_cider = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset.answers)})

    def load_vocab(self, config):
        vocab = Vocab(config.DATASET)

        return vocab

    def load_feature_datasets(self, config):
        train_dataset = FeatureDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = FeatureDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = FeatureDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def load_dict_datasets(self, config):
        train_dataset = DictionaryDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = DictionaryDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = DictionaryDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        out = self.model(items).contiguous()
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_answer_tokens.view(-1))
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

                answers_gt = items.answer
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
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
                out = self.model(items).contiguous()
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_answer_tokens.view(-1))
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
                outs, log_probs = self.model.beam_search(items, batch_size=self.train_dict_dataloader.batch_size, 
                                                            beam_size=self.training_beam_size, out_size=self.training_beam_size)
                
                self.optim.zero_grad()

                # Rewards
                bs = items.question_tokens.shape[0]
                answers_gt = items.answer
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.training_beam_size for a in answers_gt)))
                gens = {f"{idx}": answer_gen for idx, answer_gen in enumerate(answers_gen)}
                gts = {f"{idx}": answer_gt for idx, answer_gt in enumerate(answers_gt)}
                reward = self.train_cider.compute_score(gts, gens)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(self.device).view(bs, self.config.training.training_beam_size)
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

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            use_rl = checkpoint["use_rl"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"]
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            use_rl = False
            best_val_score = .0
            patience = 0

        while True:
            if not use_rl:
                self.train()
            else:
                self.train_scst()

            val_loss = self.evaluate_loss(self.val_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.val_dict_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            if self.test_dict_dataloader is not None:
                scores = self.evaluate_metrics(self.test_dict_dataloader)
                logger.info("Evaluation scores %s", scores)

            # Prepare for next epoch
            best = False
            if val_score >= best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            switch_to_rl = False
            exit_train = False

            if patience == self.patience:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    self.optim = Adam(self.model.parameters(), lr=self.rl_learning_rate)
                    logger.info("Switching to RL")
                else:
                    logger.info('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

            self.save_checkpoint({
                'val_loss': val_loss,
                'val_cider': val_score,
                'patience': patience,
                'use_rl': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self, dataset, get_scores=True):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        logger.info(f"Loading checkpoint from {os.path.join(self.checkpoint_path, 'best_model.pth')} for predicting")
        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        with tqdm(desc='Getting predictions: ', unit='it', total=len(dataset)) as pbar:
            for it, items in enumerate(dataset):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, beam_size=self.evaluating_beam_size, out_size=1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.training_beam_size for a in answers_gt)))
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()
                
                if get_scores:
                    scores, _ = evaluation.compute_scores(gts, gens)
                else:
                    scores = None

                results.append({
                    "image_id": items.image_id,
                    "filename": items.filename,
                    "gens": gens,
                    "gts": gts,
                    "scores": scores
                })

                pbar.update()

        return results