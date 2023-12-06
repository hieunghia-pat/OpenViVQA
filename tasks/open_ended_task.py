import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
import evaluation
from evaluation import Cider

import os
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

@META_TASK.register()
class OpenEndedTask(BaseTask):
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
        with tqdm(desc='Epoch %d - Evaluating' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    results = self.model(items)
                outs = results["scores"].argmax(dim=-1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous(), 
                                                        join_words=False)
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
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                results = self.model(items)
                out = results["scores"].contiguous()
                out = F.log_softmax(out, dim=-1)

                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

                if it % self.config.ITER_TO_VERBOSE == 0:
                    self.logger.info(f"Loss: {loss.item()}")


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
        with tqdm(desc='Predicting: ', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    result = self.model(items)
                outs = result["scores"].argmax(dim=-1)

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
        scores = {key: value for key, value in scores.items() if key in self.config.TRAINING.VERBOSE_SCORES}
        self.logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)
