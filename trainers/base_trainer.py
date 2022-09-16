import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data_utils.vocab import Vocab, ClassificationVocab
from data_utils.dataset import FeatureDataset, FeatureClassificationDataset, DictionaryDataset
from data_utils.utils import collate_fn
from utils.logging_utils import setup_logger
from utils.instances import Instances

from builders.model_builder import build_model

import evaluation
from evaluation import Cider

import os
import numpy as np
import pickle
from tqdm import tqdm
import itertools
import random
from shutil import copyfile

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = setup_logger()

class BaseTrainer:
    def __init__(self, config):

        self.checkpoint_path = os.path.join(config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        if not os.path.isdir(self.checkpoint_path):
            logger.info("Creating checkpoint path")
            os.makedirs(self.checkpoint_path)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            logger.info("Creating vocab")
            if config.DATASET.TASK == "open-ended":
                self.vocab = Vocab(config.DATASET.VOCAB)
            else:
                self.vocab = ClassificationVocab(config.DATASET.VOCAB)
            logger.info("Saving vocab to {}" % os.path.join(self.checkpoint_path, "vocab.bin"))
            pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        else:
            logger.info("Loading vocab from {}" % os.path.join(self.checkpoint_path, "vocab.bin"))
            self.vocab = pickle.load(open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb"))

        logger.info("Loading datasets")
        self.train_dataset, self.dev_dataset, self.test_dataset = self.load_feature_datasets(config.DATASET)
        self.train_dict_dataset, self.dev_dict_dataset, self.test_dict_dataset = self.load_dict_datasets(config.DATASET)
        
        # creating iterable-dataset data loader
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_fn
        )
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                collate_fn=collate_fn
            )
        else:
            self.test_dataloader = None

        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.batch_size // config.training_beam_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_dict_dataloader = DataLoader(
            dataset=self.val_dict_dataset,
            batch_size=config.batch_size // config.training_beam_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        if self.test_dict_dataset is not None:
            self.test_dict_dataloader = DataLoader(
                dataset=self.test_dict_dataset,
                batch_size=config.batch_size // config.training_beam_size,
                shuffle=True,
                collate_fn=collate_fn
            )
        else:
            self.test_dict_dataloader = None

        logger.info("Building model")
        self.model = build_model(config.MODEL)
        self.config = config

        logger.info("Defining optimizer and objective function")
        self.optim = Adam(self.model.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)
        
        # training hyperparameters
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE
        self.rl_learning_rate = config.TRAINING.RL_LEARNING_RATE
        self.get_scores = config.TRAINING.GET_SCORES
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE

        self.train_cider = Cider(self.train_dataset.answers())

    def load_feature_datasets(self, config):
        if config.TASK == "open-ended":
            train_dataset = FeatureDataset(config.JSON_PATH.TRAIN, self.vocab, config)
            dev_dataset = FeatureDataset(config.JSON_PATH.DEV, self.vocab, config)
            test_dataset = FeatureDataset(config.JSON_PATH.TEST, self.vocab, config)
        else:
            train_dataset = FeatureClassificationDataset(config.JSON_PATH.TRAIN, self.vocab, config)
            dev_dataset = FeatureClassificationDataset(config.JSON_PATH.DEV, self.vocab, config)
            test_dataset = FeatureClassificationDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def load_dict_datasets(self, config):
        train_dataset = DictionaryDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = DictionaryDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = DictionaryDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset

    def evaluate_loss(self, dataloader: DataLoader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
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

    def evaluate_metrics(self, dataloader: DataLoader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, beam_size=self.evaluating_beam_size, out_size=1)

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.training_beam_size for a in answers_gt)))
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

    def train_xe(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
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
        running_reward = .0
        running_reward_baseline = .0

        vocab: Vocab = self.train_dataset.vocab

        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with self-critical learning' % self.epoch, unit='it', total=len(self.train_dict_dataloader)) as pbar:
            for it, items in enumerate(self.train_dict_dataloader):
                outs, log_probs = self.model.beam_search(items, beam_size=self.training_beam_size, 
                                                        out_size=self.training_beam_size)
                
                self.optim.zero_grad()

                # Rewards
                bs = items.question_tokens.shape[0]
                answers_gt = items.answers
                answers_gen = vocab.decode_answer(outs.contiguous().view(-1, vocab.max_answer_length), join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.training_beam_size for a in answers_gt)))
                reward = self.train_cider.compute_score(answers_gt, answers_gen)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(device).view(bs, self.config.training.training_beam_size)
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

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from {}" % fname)

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        logger.info(f"resuming from epoch {checkpoint['epoch']} - validation loss {checkpoint['val_loss']} - \
                        best cider on val {checkpoint['best_val_score']} - best cider on test {checkpoint['best_test_score']}")

        return Instances(
            use_rl = checkpoint['use_rl'],
            best_val_score=checkpoint['best_val_score'],
            best_test_score=checkpoint['best_test_score'],
            patience=checkpoint['patience'],
            epoch=checkpoint["epoch"],
            optimizer=checkpoint["optimizer"],
            scheduler=checkpoint["scheduler"]
        )

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def train(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            use_rl = checkpoint["use_rl"]
            best_val_score = checkpoint["best_val_score"]
            best_test_score = checkpoint["best_test_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"]
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            use_rl = False
            best_val_score = .0
            best_test_score = .0
            patience = 0

        while True:
            if not use_rl:
                self.train_xe()
            else:
                self.train_scst()

            val_loss = self.evaluate_loss(self.val_dataloader)

            # val scores
            scores = self.evaluate_metrics(self.val_dict_dataloader)
            print("Validation scores", scores)
            val_score = scores[self.score]

            if self.test_dict_dataloader is not None:
                scores = self.evaluate_metrics(self.test_dict_dataloader)
                print("Evaluation scores", scores)

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
                    print("Switching to RL")
                else:
                    print('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

            self.save_checkpoint({
                'val_loss': val_loss,
                'val_cider': val_score,
                'patience': patience,
                'best_val_score': best_val_score,
                'best_test_score': best_test_score,
                'use_rl': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                        os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self, dataset: DictionaryDataset, get_scores=True):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        logger.info(f"Loading checkpoint from {os.path.join(self.checkpoint_path, 'best_model.pth')} for predicting")
        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        results = []
        with tqdm(desc='Getting predictions: ', unit='it', total=len(dataset)) as pbar:
            for it, items in enumerate(dataset):
                items = items.to(device)
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
