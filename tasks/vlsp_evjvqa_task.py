import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.logging_utils import setup_logger
from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
import evaluation
from evaluation import Cider

import os
import numpy as np
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

logger = setup_logger()

@META_TASK.register()
class VlspEvjVqaTask(BaseTask):
    '''
        This task is designed especially for EVJVQA task at VLSP2022
    '''
    def __init__(self, config):
        super().__init__(config)

    def load_feature_datasets(self, config):
        train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.FEATURE_DATASET)
        dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.FEATURE_DATASET)
        public_test_dataset = build_dataset(config.JSON_PATH.PUBLIC_TEST, self.vocab, config.FEATURE_DATASET)
        private_test_dataset = build_dataset(config.JSON_PATH.PRIVATE_TEST, self.vocab, config.FEATURE_DATASET)

        return train_dataset, dev_dataset, public_test_dataset, private_test_dataset

    def load_dict_datasets(self, config):
        train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.DICT_DATASET)
        dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.DICT_DATASET)
        public_test_dataset = build_dataset(config.JSON_PATH.PUBLIC_TEST, self.vocab, config.DICT_DATASET)
        private_test_dataset = build_dataset(config.JSON_PATH.PRIVATE_TEST, self.vocab, config.DICT_DATASET)

        return train_dataset, dev_dataset, public_test_dataset, private_test_dataset

    def load_datasets(self, config):
        self.train_dataset, self.dev_dataset, self.public_test_dataset, self.private_test_dataset = self.load_feature_datasets(config)
        self.train_dict_dataset, self.dev_dict_dataset, self.public_test_dict_dataset, self.private_test_dict_dataset = self.load_dict_datasets(config)

    def create_feature_dataloaders(self, config):
        # creating iterable-dataset data loader
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.public_test_dataloader = DataLoader(
            dataset=self.public_test_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        ) if self.public_test_dict_dataset else None
        self.private_test_dataloader = DataLoader(
            dataset=self.private_test_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        ) if self.private_test_dict_dataset else None

    def create_dict_dataloaders(self, config):
        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.dev_dict_dataloader = DataLoader(
            dataset=self.dev_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.public_test_dict_dataloader = DataLoader(
            dataset=self.public_test_dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.public_test_dataset else None
        self.private_test_dict_dataloader = DataLoader(
            dataset=self.private_test_dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.private_test_dataset else None

    def create_dataloaders(self, config):
        self.create_feature_dataloaders(config)
        self.create_dict_dataloaders(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        self.train_cider = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset.answers)})

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

                answers_gt = items.answers
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
                outs, log_probs = self.model.beam_search(items, batch_size=items.batch_size, 
                                                            beam_size=self.training_beam_size, out_size=self.training_beam_size)
                
                self.optim.zero_grad()

                # Rewards
                bs = items.batch_size
                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=True)
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

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            # use_rl = checkpoint["use_rl"]
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # use_rl = False
            best_val_score = .0
            patience = 0

        while True:
            # if not use_rl:
            self.train()
            # else:
            #     self.train_scst()

            self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
            logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            # Prepare for next epoch
            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            # switch_to_rl = False
            exit_train = False

            if patience == self.patience:
                # if not use_rl:
                #     use_rl = True
                #     switch_to_rl = True
                #     patience = 0
                #     self.optim = Adam(self.model.parameters(), lr=self.rl_learning_rate)
                #     logger.info("Switching to RL")
                # else:
                logger.info('patience reached.')
                exit_train = True

            # if switch_to_rl and not best:
            #     self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience,
                # 'use_rl': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()

        if self.public_test_dict_dataset is not None:
            results = []
            overall_gens = {}
            overall_gts = {}
            with tqdm(desc='Getting predictions on public test: ', unit='it', total=len(self.public_test_dict_dataloader)) as pbar:
                for it, items in enumerate(self.public_test_dict_dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                    answers_gt = items.answers
                    answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                    gts = {}
                    gens = {}
                    for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gens['%d_%d' % (it, i)] = gen_i
                        gts['%d_%d' % (it, i)] = gts_i
                        overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                        overall_gts['%d_%d' % (it, i)] = gts_i
                    pbar.update()

                    results.append({
                        "id": items.question_id,
                        "image_id": items.image_id,
                        "filename": items.filename,
                        "gens": gens,
                        "gts": gts
                    })

                    pbar.update()

            scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
            logger.info("Evaluation score on public test: %s", scores)

            json.dump({
                "results": results,
                **scores
            }, open(os.path.join(self.checkpoint_path, "public_test_results.json"), "w+"), ensure_ascii=False)

        if self.private_test_dict_dataset is not None:
            results = []
            overall_gens = {}
            overall_gts = {}
            with tqdm(desc='Getting predictions on private test: ', unit='it', total=len(self.private_test_dict_dataloader)) as pbar:
                for it, items in enumerate(self.private_test_dict_dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                    answers_gt = items.answers
                    answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                    gts = {}
                    gens = {}
                    for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gens['%d_%d' % (it, i)] = gen_i
                        gts['%d_%d' % (it, i)] = gts_i
                        overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                        overall_gts['%d_%d' % (it, i)] = gts_i

                    pbar.update()

                    results.append({
                        "id": items.question_id,
                        "image_id": items.image_id,
                        "filename": items.filename,
                        "gens": gens,
                        "gts": gts
                    })

                    pbar.update()

            scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
            logger.info("Evaluation score on public test: %s", scores)

            json.dump({
                "results": results,
                **scores
            }, open(os.path.join(self.checkpoint_path, "private_test_results.json"), "w+"), ensure_ascii=False)