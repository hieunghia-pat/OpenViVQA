import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_utils.utils import collate_fn
from .open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
import evaluation
from evaluation import Cider

import os
import numpy as np
import itertools
from shutil import copyfile
import json
import datetime

@META_TASK.register()
class TrainingHuggingFaceModels(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)

    def load_feature_datasets(self, config):
        train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.FEATURE_DATASET)
        dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.FEATURE_DATASET)
        test_dataset = build_dataset(config.JSON_PATH.TEST, self.vocab, config.FEATURE_DATASET)

        return train_dataset, dev_dataset, test_dataset

    def load_dict_datasets(self, config):
        train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.DICT_DATASET)
        dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.DICT_DATASET)
        test_dataset = build_dataset(config.JSON_PATH.TEST, self.vocab, config.DICT_DATASET)

        return train_dataset, dev_dataset, test_dataset

    def load_datasets(self, config):
        self.train_dataset, self.dev_dataset, self.test_dataset = self.load_feature_datasets(config)
        self.train_dict_dataset, self.dev_dict_dataset, self.test_dict_dataset = self.load_dict_datasets(config)

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
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )

    def create_dict_dataloaders(self, config):
        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.dev_dict_dataloader = DataLoader(
            dataset=self.dev_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader = DataLoader(
            dataset=self.test_dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn
        )

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

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens = {}
        gts = {}
        # for estimating the eslapsed time
        durations = []
        for it, items in enumerate(dataloader):
            start_moment = datetime.datetime.now()
            items = items.to(self.device)
            outs = self.model.generate(items)

            answers_gt = items.answers
            answers_gen = self.vocab.decode_answer(outs.contiguous(), 
                                                    join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gens['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i

            # get the time of the ending moment
            end_moment = datetime.datetime.now()
            # estimating esplapsed time
            durations.append((end_moment - start_moment).total_seconds())
            avg_duration = np.array(durations).mean()
            remain_its = len(dataloader) - it
            total_time = str(datetime.timedelta(seconds=int(avg_duration * remain_its)))

            if it > 0 and it % self.config.TRAINING.ITER_TO_VERBOSE == 0:
                self.logger.info(f"Epoch {self.epoch+1} - Evaluating - Iter {it}/{len(dataloader)} - Estimating remaining: {total_time}")

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train(self):
        self.model.train()
        running_loss = .0
        # for estimating the eslapsed time
        durations = []
        for it, items in enumerate(self.train_dataloader):
            # get the time of the starting moment
            start_moment = datetime.datetime.now()
            # forward pass
            items = items.to(self.device)
            results = self.model(items)
            
            self.optim.zero_grad()
            loss = results.loss
            loss.backward()

            self.optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            self.scheduler.step()

            # get the time of the ending moment
            end_moment = datetime.datetime.now()
            # estimating esplapsed time
            durations.append((end_moment - start_moment).total_seconds())
            avg_duration = np.array(durations).mean()
            remain_its = len(self.train_dataloader) - it
            total_time = str(datetime.timedelta(seconds=int(avg_duration * remain_its)))

            if it > 0 and it % self.config.TRAINING.ITER_TO_VERBOSE == 0:
                self.logger.info(f"Epoch {self.epoch+1} - Training - Iter {it}/{len(self.train_dataloader)} - Loss: {running_loss / (it + 1)} - Estimating remaining: {total_time}")

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = .0
            patience = 0

        while True:
            self.train()

            # val scores
            self.logger.info(f"Epoch {self.epoch+1} - Validating")
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
            scores = {key: value for key, value in scores.items() if key in self.config.TRAINING.VERBOSE_SCORES}
            self.logger.info("Validation scores %s", scores)
            val_score = scores[self.score]

            # Prepare for next epoch
            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                self.logger.info('patience reached.')
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        overall_gens = {}
        overall_gts = {}
        self.logger.info(f"Epoch {self.epoch+1} - Evaluating")
        for it, items in enumerate(self.test_dict_dataloader):
            items = items.to(self.device)
            outs = self.model.generate(items)

            answers_gt = items.answers
            answers_gen = self.vocab.decode_answer(
                outs.contiguous().view(-1, self.vocab.max_answer_length),
                join_words=False)
            gts = {}
            gens = {}
            for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gens['%d_%d' % (it, i)] = gen_i
                gts['%d_%d' % (it, i)] = gts_i
                overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                overall_gts['%d_%d' % (it, i)] = gts_i

            results.append({
                "id": items.question_id,
                "image_id": items.image_id,
                "filename": items.filename,
                "gens": gens,
                "gts": gts
            })

        scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        scores = {key: value for key, value in scores.items() if key in self.config.TRAINING.VERBOSE_SCORES}
        self.logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)
