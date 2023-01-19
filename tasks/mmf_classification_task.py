import torch
from torch import nn
from torch.nn import functional as F
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

class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        source = torch.ones_like(input)
        scattered_target = torch.zeros_like(input)
        scattered_target.scatter_(dim=-1, index=target.unsqueeze(-1), src=source)

        loss = F.binary_cross_entropy_with_logits(input, scattered_target, reduction="mean")

        return loss

@META_TASK.register()
class MmfClassificationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        # use multi-label loss
        self.loss_fn = BCEWithLogitsLoss()

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.TRAINING.SCORE
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.get_scores = config.TRAINING.GET_SCORES
        self.patience = config.TRAINING.PATIENCE

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.FEATURE_DATASET)
        self.dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.FEATURE_DATASET)
        self.test_dataset = build_dataset(config.JSON_PATH.TEST, self.vocab, config.FEATURE_DATASET)

    def create_dataloaders(self, config):
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

    def evaluate_loss(self, dataloader: DataLoader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        results = self.model(items)
                    out = results["scores"].contiguous()
                    
                    answer = items.answer
                    loss = self.loss_fn(out.view(-1, self.vocab.num_choices), answer.view(-1))
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
                    results = self.model(items)
                outs = results["scores"].contiguous()

                answers_gt = self.vocab.decode_answer(items.answer.squeeze(-1), items.ocr_tokens, join_word=True)
                answers_gen = self.vocab.decode_answer(outs.argmax(dim=-1), items.ocr_tokens, join_word=True)
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
                results = self.model(items)
                out = results["scores"].contiguous()
                answer = items.answer
                self.optim.zero_grad()
                loss = self.loss_fn(out, answer.view(-1))
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
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = .0
            patience = 0

        while True:
            self.train()

            # self.evaluate_loss(self.dev_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
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

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    result = self.model(items)
                outs = result["scores"].contiguous()

                question = self.vocab.decode_question(items.question_tokens, join_words=True)
                answers_gt = self.vocab.decode_answer(items.answer, items.ocr_texts, join_word=True)
                answers_gen = self.vocab.decode_answer(outs.argmax(dim=-1, keepdim=True), items.ocr_texts, join_word=True)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gens['%d_%d' % (it, i)] = gen_i
                    gts['%d_%d' % (it, i)] = gts_i
                    overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                    overall_gts['%d_%d' % (it, i)] = [gts_i, ]
                pbar.update()

                results.append({
                    "filename": items.filename,
                    "question": question,
                    "gens": gens,
                    "gts": gts
                })

                pbar.update()

        scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)