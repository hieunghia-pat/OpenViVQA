import torch
from torch.utils.data import DataLoader

from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.dataset_builder import build_dataset
from builders.task_builder import META_TASK
from utils.logging_utils import setup_logger
import evaluation

import os
from shutil import copyfile
from tqdm import tqdm
import json

logger = setup_logger()

@META_TASK.register()
class ClassificationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.get_scores = config.TRAINING.GET_SCORES
        self.patience = config.TRAINING.PATIENCE

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config)
        self.dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config)
        self.test_dataset = build_dataset(config.JSON_PATH.TEST, self.vocab, config)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn
        )

    def evaluate_loss(self, dataloader: DataLoader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        out = self.model(items).contiguous()
                    
                    answer = items.answer_tokens
                    loss = self.loss_fn(out.view(-1, self.vocab.total_answers), answer.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader: DataLoader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model(items).contiguous()

                answers_gt = items.answer
                answers_gen = self.vocab.decode_answer(outs)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = [gts_i, ]
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                out = self.model(items).contiguous()
                answer = items.answer_tokens
                self.optim.zero_grad()
                loss = self.loss_fn(out.view(-1, self.vocab.total_answers), answer.view(-1))
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def lambda_lr(self, step):
        return self.learning_rate

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"]
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = .0
            patience = 0

        while True:
            self.train()

            self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dict_dataloader)
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

            exit_train = False
            if patience == self.patience:
                logger.info('patience reached.')
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

    def get_predictions(self, get_scores=True):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dataset)) as pbar:
            for it, items in enumerate(self.test_dataset):
                items = items.unsqueeze(dim=0)
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model(items)

                answers_gt = items.answer
                answers_gen = self.vocab.decode_answer(outs)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = [gts_i, ]
                pbar.update()
                
                if get_scores:
                    scores, _ = evaluation.compute_scores(gts, gens)
                else:
                    scores = None

                results.append({
                    "id": items.question_id,
                    "filename": items.filename,
                    "gens": gens,
                    "gts": gts,
                    "scores": scores
                })

                pbar.update()

        json.dump(results, open(os.path.join(self.checkpoint_path, "results.json"), "w+"), ensure_ascii=False)