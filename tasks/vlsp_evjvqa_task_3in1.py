from ast import Return
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.logging_utils import setup_logger
from utils.instances import Instances
from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
import evaluation
from data_utils import vocab
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
#ENGLISH
    def load_feature_datasets_en(self, config):
#ENGLISH train_dev dataset
        train_dataset_en = build_dataset(config.JSON_PATH.TRAIN_EN, self.vocab_en, config.FEATURE_DATASET)
        dev_dataset_en = build_dataset(config.JSON_PATH.DEV_EN, self.vocab_en, config.FEATURE_DATASET)

#ENGLISH test dataset
        if config.JSON_PATH.TEST_EN is not None:
            test_dataset_en = build_dataset(config.JSON_PATH.TEST_EN, self.vocab_en, config.FEATURE_DATASET)
        else:
            test_dataset_en = None

# public test
        if config.JSON_PATH.PUBLIC_TEST is not None:
            public_test_dataset = build_dataset(config.JSON_PATH.PUBLIC_TEST, self.vocab, config.FEATURE_DATASET)
        else:
            public_test_dataset = None
#private dataset
        if config.JSON_PATH.PRIVATE_TEST is not None:
            private_test_dataset = build_dataset(config.JSON_PATH.PRIVATE_TEST, self.vocab, config.FEATURE_DATASET)
        else:
            private_test_dataset = None
        return train_dataset_en,dev_dataset_en,test_dataset_en,public_test_dataset,private_test_dataset


#VIETNAMESE
    def load_feature_datasets_vi(self, config):
#VIETNAMESE train_dev dataset        
        train_dataset_vi = build_dataset(config.JSON_PATH.TRAIN_VI, self.vocab_vi, config.FEATURE_DATASET)
        dev_dataset_vi = build_dataset(config.JSON_PATH.DEV_VI, self.vocab_vi, config.FEATURE_DATASET)
#VIETNAMESE test dataset
        if config.JSON_PATH.TEST_VI is not None:
            test_dataset_vi = build_dataset(config.JSON_PATH.TEST_VI, self.vocab_vi, config.FEATURE_DATASET)
        else:
            test_dataset_vi = None
        return train_dataset_vi,dev_dataset_vi,test_dataset_vi
    
#JAPANESE   
    def load_feature_datasets_ja(self, config):
#JAPANESE train_dev dataset        
        train_dataset_ja = build_dataset(config.JSON_PATH.TRAIN_JA, self.vocab_ja, config.FEATURE_DATASET)
        dev_dataset_ja = build_dataset(config.JSON_PATH.DEV_JA, self.vocab_ja, config.FEATURE_DATASET)
#JAPANESE test dataset
        if config.JSON_PATH.TEST_JA is not None:
            test_dataset_ja = build_dataset(config.JSON_PATH.TEST_JA, self.vocab_ja, config.FEATURE_DATASET)
        else:
            test_dataset_ja = None

        return  train_dataset_ja, dev_dataset_ja, test_dataset_ja



#ENGLISH
    def load_dict_datasets_en(self, config):
#ENGLISH train_dev dataset
        train_dataset_en = build_dataset(config.JSON_PATH.TRAIN_EN, self.vocab_en, config.DICT_DATASET)
        dev_dataset_en = build_dataset(config.JSON_PATH.DEV_EN, self.vocab_en, config.DICT_DATASET)
#ENGLISH test dataset
        if config.JSON_PATH.TEST_EN is not None:
            test_dataset_en = build_dataset(config.JSON_PATH.TEST_EN, self.vocab_en, config.DICT_DATASET)
        else:
            test_dataset_en = None
#public test
        if config.JSON_PATH.PUBLIC_TEST is not None:
            public_test_dataset = build_dataset(config.JSON_PATH.PUBLIC_TEST, self.vocab, config.DICT_DATASET)
        else:
            public_test_dataset = None
#private dataset
        if config.JSON_PATH.PRIVATE_TEST is not None:
            private_test_dataset = build_dataset(config.JSON_PATH.PRIVATE_TEST, self.vocab, config.DICT_DATASET)
        else:
            private_test_dataset = None
        return train_dataset_en,dev_dataset_en,test_dataset_en,public_test_dataset,private_test_dataset
    
#VIETNAMESE
    def load_dict_datasets_vi(self, config):
#VIETNAMESE train_dev dataset        
        train_dataset_vi = build_dataset(config.JSON_PATH.TRAIN_VI, self.vocab_vi, config.DICT_DATASET)
        dev_dataset_vi = build_dataset(config.JSON_PATH.DEV_VI, self.vocab_vi, config.DICT_DATASET)
#VIETNAMESE test dataset
        if config.JSON_PATH.TEST_VI is not None:
            test_dataset_vi = build_dataset(config.JSON_PATH.TEST_VI, self.vocab_vi, config.DICT_DATASET)
        else:
            test_dataset_vi = None
        return train_dataset_vi,dev_dataset_vi,test_dataset_vi
#JANPANESE
    def load_dict_datasets_ja(self, config):
#JAPANESE train_dev dataset     
        train_dataset_ja = build_dataset(config.JSON_PATH.TRAIN_JA, self.vocab_ja, config.DICT_DATASET)
        dev_dataset_ja = build_dataset(config.JSON_PATH.DEV_JA, self.vocab_ja, config.DICT_DATASET)
#JAPANESE test dataset
        if config.JSON_PATH.TEST_JA is not None:
            test_dataset_ja = build_dataset(config.JSON_PATH.TEST_JA, self.vocab_ja, config.DICT_DATASET)
        else:
            test_dataset_ja = None
        return train_dataset_ja, dev_dataset_ja,test_dataset_ja
     
#ENGLISH
    def load_datasets_en(self, config):
        self.train_dataset_en, self.dev_dataset_en,self.test_dataset_en,self.public_test_dataset,self.private_test_dataset= self.load_feature_datasets_en(config)
        self.train_dict_dataset_en, self.dev_dict_dataset_en,self.test_dict_dataset_en,self.public_test_dict_dataset,self.private_test_dict_dataset= self.load_dict_datasets_en(config)
#VIETNAMESE
    def load_datasets_vi(self, config):
        self.train_dataset_vi, self.dev_dataset_vi,self.test_dataset_vi= self.load_feature_datasets_vi(config)
        self.train_dict_dataset_vi, self.dev_dict_dataset_vi,self.test_dict_dataset_vi= self.load_dict_datasets_vi(config)
#JANPANESE
    def load_datasets_ja(self, config):
            self.train_dataset_ja, self.dev_dataset_ja,self.test_dataset_ja= self.load_feature_datasets_ja(config)
            self.train_dict_dataset_ja, self.dev_dict_dataset_ja,self.test_dict_dataset_ja= self.load_dict_datasets_ja(config)
#ENGLISH
    def create_feature_dataloaders_en(self, config):
        # creating iterable-dataset data loader
        self.train_dataloader_en = DataLoader(
            dataset=self.train_dataset_en,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader_en = DataLoader(
            dataset=self.dev_dataset_en,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader_en = DataLoader(
            dataset=self.test_dataset_en,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        ) if self.test_dict_dataset_en else None
#private and public test
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

#VIETNAMESE
    def create_feature_dataloaders_vi(self, config):
        self.train_dataloader_vi = DataLoader(
            dataset=self.train_dataset_vi,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader_vi = DataLoader(
            dataset=self.dev_dataset_vi,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader_vi = DataLoader(
            dataset=self.test_dataset_vi,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        ) if self.test_dict_dataset_vi else None
#JAPANESE
    def create_feature_dataloaders_ja(self, config):
        self.train_dataloader_ja = DataLoader(
            dataset=self.train_dataset_ja,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader_ja = DataLoader(
            dataset=self.dev_dataset_ja,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader_ja = DataLoader(
            dataset=self.test_dataset_ja,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        ) if self.test_dict_dataset_ja else None

#ENGLISH
    def create_dict_dataloaders_en(self, config):
        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader_en = DataLoader(
            dataset=self.train_dict_dataset_en,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.dev_dict_dataloader_en = DataLoader(
            dataset=self.dev_dict_dataset_en,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader_en = DataLoader(
            dataset=self.test_dict_dataset_en,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.test_dataset_en else None
#private and public test
        self.public_test_dict_dataloader = DataLoader(
            dataset=self.public_test_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.public_test_dataset else None
        self.private_test_dict_dataloader = DataLoader(
            dataset=self.private_test_dict_dataset,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.private_test_dataset else None

#VIETNAMESE
    def create_dict_dataloaders_vi(self, config):
        self.train_dict_dataloader_vi = DataLoader(
            dataset=self.train_dict_dataset_vi,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.dev_dict_dataloader_vi = DataLoader(
            dataset=self.dev_dict_dataset_vi,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader_vi = DataLoader(
            dataset=self.test_dict_dataset_vi,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.test_dataset_vi else None
#JAPANESE
    def create_dict_dataloaders_ja(self, config):
        self.train_dict_dataloader_ja = DataLoader(
            dataset=self.train_dict_dataset_ja,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.dev_dict_dataloader_ja = DataLoader(
            dataset=self.dev_dict_dataset_ja,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.test_dict_dataloader_ja = DataLoader(
            dataset=self.test_dict_dataset_ja,
            batch_size=config.DATASET.DICT_DATASET.BATCH_SIZE // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        ) if self.test_dataset_ja else None
#ENGLISH
    def create_dataloaders_en(self, config):
        self.create_feature_dataloaders_en(config)
        self.create_dict_dataloaders_en(config)
#VIETNAMESE
    def create_dataloaders_vi(self, config):
        self.create_feature_dataloaders_vi(config)
        self.create_dict_dataloaders_vi(config)
#JAPANESE
    def create_dataloaders_ja(self, config):
        self.create_feature_dataloaders_ja(config)
        self.create_dict_dataloaders_ja(config)
#ENGLISH
    def configuring_hyperparameters_en(self, config):
        self.epoch_en = 0
        self.warmup_en = config.TRAINING.WARMUP
        self.score_en = config.TRAINING.SCORE
        self.learning_rate_en = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate_en = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size_en = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size_en = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience_en = config.TRAINING.PATIENCE
        self.train_cider_en = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset_en.answers)})
#VIETNAMESE
    def configuring_hyperparameters_vi(self, config):
        self.epoch_vi = 0
        self.warmup_vi = config.TRAINING.WARMUP
        self.score_vi = config.TRAINING.SCORE
        self.learning_rate_vi = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate_vi = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size_vi = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size_vi = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience_vi = config.TRAINING.PATIENCE
        self.train_cider_vi = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset_vi.answers)})
#JAPANESE
    def configuring_hyperparameters_ja(self, config):
        self.epoch_ja = 0
        self.warmup_ja = config.TRAINING.WARMUP
        self.score_ja = config.TRAINING.SCORE
        self.learning_rate_ja = config.TRAINING.LEARNING_RATE
        self.rl_learning_rate_ja = config.TRAINING.RL_LEARNING_RATE
        self.training_beam_size_ja = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size_ja = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience_ja = config.TRAINING.PATIENCE
        self.train_cider_ja = Cider({f"{idx}": answer for idx, answer in enumerate(self.train_dataset_ja.answers)})
#ENGLISH
    def evaluate_loss_en(self, dataloader):
        self.model_en.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch_en, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device_en)
                    with torch.no_grad():
                        out = self.model_en(items).contiguous()
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn_en(out.view(-1, len(self.vocab_en)), shifted_right_answer_tokens.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss
#VIETNAMESE
    def evaluate_loss_vi(self, dataloader):
        self.model_vi.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch_vi, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device_vi)
                    with torch.no_grad():
                        out = self.model_vi(items).contiguous()
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn_vi(out.view(-1, len(self.vocab_vi)), shifted_right_answer_tokens.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss


#JAPANESE
    def evaluate_loss_ja(self, dataloader):
        self.model_ja.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch_ja, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device_ja)
                    with torch.no_grad():
                        out = self.model_ja(items).contiguous()
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    loss = self.loss_fn_ja(out.view(-1, len(self.vocab_ja)), shifted_right_answer_tokens.view(-1))
                    this_loss = loss.item()
                    running_loss += this_loss

                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

#ENGLISH
    def evaluate_metrics_en(self, dataloader):
        self.model_en.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch_en, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device_en)
                with torch.no_grad():
                    outs, _ = self.model_en.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size_en, out_size=1)
  
                answers_gt = items.answer
                answers_gen = self.vocab_en.decode_answer(outs.contiguous().view(-1, self.vocab_en.max_answer_length), join_words=False)
                #print('\n',len(answers_gt),"gt: ",answers_gt)
                #print(len(answers_gen),"gen: ",answers_gen)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i

                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores
#VIETNAMESE
    def evaluate_metrics_vi(self, dataloader):
        self.model_vi.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch_vi, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device_vi)
                with torch.no_grad():
                    outs, _ = self.model_vi.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size_vi, out_size=1)
  
                answers_gt = items.answer
                answers_gen = self.vocab_vi.decode_answer(outs.contiguous().view(-1, self.vocab_vi.max_answer_length), join_words=False)
                #print('\n',len(answers_gt),"gt: ",answers_gt)
                #print(len(answers_gen),"gen: ",answers_gen)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i

                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores
#JAPANESE
    def evaluate_metrics_ja(self, dataloader):
        self.model_ja.eval()
        gens = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch_ja, unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device_ja)
                with torch.no_grad():
                    outs, _ = self.model_ja.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size_ja, out_size=1)
  
                answers_gt = items.answer
                answers_gen = self.vocab_ja.decode_answer(outs.contiguous().view(-1, self.vocab_ja.max_answer_length), join_words=False)
                #print('\n',len(answers_gt),"gt: ",answers_gt)
                #print(len(answers_gen),"gen: ",answers_gen)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i

                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores
#ENGLISH
    def train_en(self):
        self.model_en.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch_en, unit='it', total=len(self.train_dataloader_en)) as pbar:
            for it, items in enumerate(self.train_dataloader_en):
                items = items.to(self.device_en)
                out = self.model_en(items).contiguous()
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim_en.zero_grad()
                loss = self.loss_fn_en(out.view(-1, len(self.vocab_en)), shifted_right_answer_tokens.view(-1))
                loss.backward()

                self.optim_en.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler_en.step()
    #VIETNAMESE
    def train_vi(self):
        self.model_vi.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch_vi, unit='it', total=len(self.train_dataloader_vi)) as pbar:
            for it, items in enumerate(self.train_dataloader_vi):
                items = items.to(self.device_vi)
                out = self.model_vi(items).contiguous()
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim_vi.zero_grad()
                loss = self.loss_fn_vi(out.view(-1, len(self.vocab_vi)), shifted_right_answer_tokens.view(-1))
                loss.backward()

                self.optim_vi.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler_vi.step()
#JANPANESE
    def train_ja(self):
        self.model_ja.train()
        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch_ja, unit='it', total=len(self.train_dataloader_ja)) as pbar:
            for it, items in enumerate(self.train_dataloader_ja):
                items = items.to(self.device_ja)
                out = self.model_ja(items).contiguous()
                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim_ja.zero_grad()
                loss = self.loss_fn_ja(out.view(-1, len(self.vocab_ja)), shifted_right_answer_tokens.view(-1))
                loss.backward()

                self.optim_ja.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler_ja.step()
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
                answers_gt = items.answer
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
#ENGLISH
    def start_en(self):
        if os.path.isfile(os.path.join(self.checkpoint_path_en, "last_model.pth")):
            checkpoint = self.load_checkpoint_en(os.path.join(self.checkpoint_path_en, "last_model.pth"))
            use_rl = checkpoint["use_rl_en"]
            best_val_score = checkpoint["best_val_score_en"]
            patience = checkpoint["patience_en"]
            self.epoch_en = checkpoint["epoch_en"] + 1
            self.optim_en.load_state_dict(checkpoint['optimizer_en'])
            self.scheduler_en.load_state_dict(checkpoint['scheduler_en'])
        else:
            use_rl = False
            best_val_score = .0
            patience = 0
        while True:
            # if not use_rl:
            #     self.train()
            # else:
            #     self.train_scst()
            if not use_rl:
              self.train_en() #khi dừng train muốn predict thì cmt từ dòng này tới 279
            self.evaluate_loss_en(self.dev_dataloader_en)

            # val scores
            scores = self.evaluate_metrics_en(self.dev_dict_dataloader_en)
            logger.info("Validation scores en %s", scores)
            val_score = scores[self.score_en]

            # # Prepare for next epoch
            best = False # đừng cmt dòng này
            if val_score >= best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            switch_to_rl = False
            exit_train = False

            if patience == self.patience_en:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    self.optim_en = Adam(self.model_en.parameters(), lr=self.rl_learning_rate)
                    logger.info("Switching to RL")
                else:
                    logger.info('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint_en(os.path.join(self.checkpoint_path_en, "best_model.pth"))

            self.save_checkpoint_en({
                'best_val_score_en': best_val_score,
                'patience_en': patience,
                'use_rl_en': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path_en, "last_model.pth"), 
                        os.path.join(self.checkpoint_path_en, "best_model.pth"))
                copyfile(os.path.join(self.checkpoint_path_en, "last_model.pth"), 
                        os.path.join('/content/drive/MyDrive/', "best_model.pth"))

            if exit_train or self.epoch_en >=15: #sửa chữ exit_train thành 1(if 1:)
                break
            
            self.epoch_en += 1
#VIETNAMESE
    def start_vi(self):
        if os.path.isfile(os.path.join(self.checkpoint_path_vi, "last_model.pth")):
            checkpoint = self.load_checkpoint_vi(os.path.join(self.checkpoint_path_vi, "last_model.pth"))
            use_rl = checkpoint["use_rl_vi"]
            best_val_score = checkpoint["best_val_score_vi"]
            patience = checkpoint["patience_vi"]
            self.epoch_vi = checkpoint["epoch_vi"] + 1
            self.optim_vi.load_state_dict(checkpoint['optimizer_vi'])
            self.scheduler_vi.load_state_dict(checkpoint['scheduler_vi'])
        else:
            use_rl = False
            best_val_score = .0
            patience = 0
        while True:
            # if not use_rl:
            #     self.train()
            # else:
            #     self.train_scst()
            #ENGLISH
            if not use_rl:
              self.train_vi() #khi dừng train muốn predict thì cmt từ dòng này tới 279
            self.evaluate_loss_vi(self.dev_dataloader_vi)

            # val scores
            scores = self.evaluate_metrics_vi(self.dev_dict_dataloader_vi)
            logger.info("Validation scores vi %s", scores)
            val_score = scores[self.score_vi]

            # # Prepare for next epoch
            best = False # đừng cmt dòng này
            if val_score >= best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            switch_to_rl = False
            exit_train = False

            if patience == self.patience_vi:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    self.optim_vi = Adam(self.model_vi.parameters(), lr=self.rl_learning_rate)
                    logger.info("Switching to RL")
                else:
                    logger.info('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint_vi(os.path.join(self.checkpoint_path_vi, "best_model.pth"))

            self.save_checkpoint_vi({
                'best_val_score_vi': best_val_score,
                'patience_vi': patience,
                'use_rl_vi': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path_vi, "last_model.pth"), 
                        os.path.join(self.checkpoint_path_vi, "best_model.pth"))
                copyfile(os.path.join(self.checkpoint_path_vi, "last_model.pth"), 
                        os.path.join('/content/drive/MyDrive/', "best_model.pth"))

            if exit_train or self.epoch_vi>=15: #sửa chữ exit_train thành 1(if 1:)
                break
            
            self.epoch_vi += 1
#JAPANESE
    def start_ja(self):
        if os.path.isfile(os.path.join(self.checkpoint_path_ja, "last_model.pth")):
            checkpoint = self.load_checkpoint_ja(os.path.join(self.checkpoint_path_ja, "last_model.pth"))
            use_rl = checkpoint["use_rl_ja"]
            best_val_score = checkpoint["best_val_score_ja"]
            patience = checkpoint["patience_ja"]
            self.epoch_ja = checkpoint["epoch_ja"] + 1
            self.optim_ja.load_state_dict(checkpoint['optimizer_ja'])
            self.scheduler_ja.load_state_dict(checkpoint['scheduler_ja'])
        else:
            use_rl = False
            best_val_score = .0
            patience = 0
        while True:
            # if not use_rl:
            #     self.train()
            # else:
            #     self.train_scst()
            if not use_rl:
              self.train_ja() #khi dừng train muốn predict thì cmt từ dòng này tới 279
            self.evaluate_loss_ja(self.dev_dataloader_ja)

            # val scores
            scores = self.evaluate_metrics_ja(self.dev_dict_dataloader_ja)
            logger.info("Validation scores ja%s", scores)
            val_score = scores[self.score_ja]

            # # Prepare for next epoch
            best = False # đừng cmt dòng này
            if val_score >= best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            switch_to_rl = False
            exit_train = False

            if patience == self.patience_ja:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    self.optim_ja = Adam(self.model_ja.parameters(), lr=self.rl_learning_rate)
                    logger.info("Switching to RL")
                else:
                    logger.info('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint_ja(os.path.join(self.checkpoint_path_ja, "best_model.pth"))

            self.save_checkpoint_ja({
                'best_val_score_ja': best_val_score,
                'patience_ja': patience,
                'use_rl_ja': use_rl
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path_ja, "last_model.pth"), 
                        os.path.join(self.checkpoint_path_ja, "best_model.pth"))
                copyfile(os.path.join(self.checkpoint_path_ja, "last_model.pth"), 
                        os.path.join('/content/drive/MyDrive/', "best_model.pth"))

            if exit_train or self.epoch_ja >=15: #sửa chữ exit_train thành 1(if 1:)
                break
            
            self.epoch_ja += 1
#ENGLISH
    def get_predictions_en(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path_en, 'last_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint_en(os.path.join(self.checkpoint_path_en, "last_model.pth"))

        self.model_en.eval()

        if self.test_dict_dataset_en is not None:
            results = []
            overall_gens = {}
            overall_gts = {}
            with tqdm(desc='Getting predictions on public test: ', unit='it', total=len(self.test_dict_dataset_en)) as pbar:
                for it, items in enumerate(self.test_dict_dataset_en):
                    items = Instances.cat([items])
                    items = items.to(self.device_en)
                    with torch.no_grad():
                        outs, _ = self.model_en.beam_search(items, batch_size=items.batch_size, beam_size=3, out_size=1)

                    answers_gt = items.answer
                    answers_gen = self.vocab_en.decode_answer(outs.contiguous().view(-1, self.vocab_en.max_answer_length), join_words=False)
                    gts = {}
                    gens = {}
                    for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gens['%d_%d' % (it, i)] = gen_i
                        gts['%d_%d' % (it, i)] = gts_i
                        overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                        overall_gts['%d_%d' % (it, i)] = [gts_i, ]
                    pbar.update()

                    results.append({
                        "id": items.question_id,
                        # "image_id": items.image_id,
                        # "filename": items.filename,
                        "gens": gens,
                        # "gts": gts
                    })

                    pbar.update()

            # scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
            # logger.info("Evaluation score on public test: %s", scores)

            json.dump({
                "results": results
            }, open(os.path.join(self.checkpoint_path_en, "public_test_results_en.json"), "w+"), ensure_ascii=False)
#VIETNAMESE
    def get_predictions_vi(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path_vi, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint_vi(os.path.join(self.checkpoint_path_vi, "best_model.pth"))

        self.model_vi.eval()

        if self.test_dict_dataset_vi is not None:
            results = []
            overall_gens = {}
            overall_gts = {}
            with tqdm(desc='Getting predictions on public test: ', unit='it', total=len(self.test_dict_dataset_vi)) as pbar:
                for it, items in enumerate(self.test_dict_dataset_vi):
                    items = Instances.cat([items])
                    items = items.to(self.device_vi)
                    with torch.no_grad():
                        outs, _ = self.model_vi.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size_vi, out_size=1)

                    answers_gt = items.answer
                    answers_gen = self.vocab_vi.decode_answer(outs.contiguous().view(-1, self.vocab_vi.max_answer_length), join_words=False)
                    gts = {}
                    gens = {}
                    for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gens['%d_%d' % (it, i)] = gen_i
                        gts['%d_%d' % (it, i)] = gts_i
                        overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                        overall_gts['%d_%d' % (it, i)] = [gts_i, ]
                    pbar.update()

                    results.append({
                        "id": items.question_id,
                        # "image_id": items.image_id,
                        # "filename": items.filename,
                        "gens": gens,
                        # "gts": gts
                    })

                    pbar.update()

            # scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
            # logger.info("Evaluation score on public test: %s", scores)

            json.dump({
                "results": results
            }, open(os.path.join(self.checkpoint_path_vi, "public_test_results_vi.json"), "w+"), ensure_ascii=False)
#JANPANESE
    def get_predictions_ja(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path_ja, 'best_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint_ja(os.path.join(self.checkpoint_path_ja, "best_model.pth"))

        self.model_ja.eval()

        if self.test_dict_dataset_en is not None:
            results = []
            overall_gens = {}
            overall_gts = {}
            with tqdm(desc='Getting predictions on public test: ', unit='it', total=len(self.test_dict_dataset_ja)) as pbar:
                for it, items in enumerate(self.test_dict_dataset_ja):
                    items = Instances.cat([items])
                    items = items.to(self.device_ja)
                    with torch.no_grad():
                        outs, _ = self.model_ja.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size_ja, out_size=1)

                    answers_gt = items.answer
                    answers_gen = self.vocab_ja.decode_answer(outs.contiguous().view(-1, self.vocab_ja.max_answer_length), join_words=False)
                    gts = {}
                    gens = {}
                    for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gens['%d_%d' % (it, i)] = gen_i
                        gts['%d_%d' % (it, i)] = gts_i
                        overall_gens['%d_%d' % (it, i)] = [gen_i, ]
                        overall_gts['%d_%d' % (it, i)] = [gts_i, ]
                    pbar.update()

                    results.append({
                        "id": items.question_id,
                        # "image_id": items.image_id,
                        # "filename": items.filename,
                        "gens": gens,
                        # "gts": gts
                    })

                    pbar.update()

            # scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
            # logger.info("Evaluation score on public test: %s", scores)

            json.dump({
                "results": results
            }, open(os.path.join(self.checkpoint_path_ja, "public_test_results_ja.json"), "w+"), ensure_ascii=False)

    
        # if self.private_test_dict_dataset is not None:
        #     results = []
        #     overall_gens = {}
        #     overall_gts = {}
        #     with tqdm(desc='Getting predictions on private test: ', unit='it', total=len(self.private_test_dict_dataset)) as pbar:
        #         for it, items in enumerate(self.private_test_dict_dataset):
        #             items = items.unsqueeze(dim=0)
        #             items = items.to(self.device)
        #             with torch.no_grad():
        #                 outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

        #             answers_gt = items.answer
        #             answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
        #             gts = {}
        #             gens = {}
        #             for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
        #                 gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
        #                 gens['%d_%d' % (it, i)] = gen_i
        #                 gts['%d_%d' % (it, i)] = gts_i
        #                 overall_gens['%d_%d' % (it, i)] = [gen_i, ]
        #                 overall_gts['%d_%d' % (it, i)] = gts_i

        #             pbar.update()

        #             results.append({
        #                 "id": items.question_id,
        #                 "image_id": items.image_id,
        #                 "filename": items.filename,
        #                 "gens": gens,
        #                 "gts": gts
        #             })

        #             pbar.update()

        #     scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        #     logger.info("Evaluation score on public test: %s", scores)

        #     json.dump({
        #         "results": results,
        #         **scores
        #     }, open(os.path.join(self.self.checkpoint_path, "private_test_results.json"), "w+"), ensure_ascii=False)