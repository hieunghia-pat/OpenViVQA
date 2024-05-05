import torch
from torch.utils.data import DataLoader

from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset
import evaluation
from evaluation import Cider

import os
import itertools
from shutil import copyfile
import json
from tqdm import tqdm

@META_TASK.register()
class OpenEndedTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def build_model(self, config):
        self.logger.info("Building model")
        self.model = build_model(config.model, self.vocab)
        self.config = config
        self.device = config.model.device

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.train, self.vocab, config)
        self.dev_dataset = build_dataset(config.dev, self.vocab, config)
        self.test_dataset = build_dataset(config.test, self.vocab, config)

    def create_dataloaders(self, config):
        # creating iterable-dataset data loader
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn
        )

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.training.warmup
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.beam_size = config.training.beam_size
        self.patience = config.training.patience
        self.train_cider = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset.answers)})

    def evaluate_metrics(self):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(self.test_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                items = items.to(self.device)
                predicted_ids, labels = self.model(**items)

                answers_gen = self.vocab.decode_answer(predicted_ids)
                for i, (gts_i, gen_i) in enumerate(zip(labels, answers_gen)):
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
            for ith, items in enumerate(self.train_dataloader, start=1):
                # forward pass
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
                    items = items.to(self.device)
                    returns = self.model(**items)

                # backward pass
                self.optim.zero_grad()
                loss = returns.loss
                loss.backward()
                self.optim.step()

                # update the training status
                this_loss = loss.item()
                running_loss += this_loss

                self.scheduler.step()

                pbar.set_postfix({
                    "Loss": running_loss / ith
                })
                pbar.update()

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
            # self.train()

            # val scores
            scores = self.evaluate_metrics()
            scores = {key: value for key, value in scores.items() if key in self.config.training.verbose_scores}
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
            outs = self.model.generate(outs)

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
        scores = {key: value for key, value in scores.items() if key in self.config.training.verbose_scores}
        self.logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)
