import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.logging_utils import setup_logger
from data_utils.utils import collate_fn
from .assemble_base_task import AssembleBaseTask
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
class AssembleTask(AssembleBaseTask):
    '''
        This task is designed especially for EVJVQA task at VLSP2022
    '''
    def __init__(self, config):
        super().__init__(config)
        self.results_path = config.PRETRAINED_MODELS.RESULTS_PATH
    def load_feature_datasets(self, config):        
        e_dataset = build_dataset(config.JSON_PATH.E_DATASET, self.vocab[0], config.FEATURE_DATASET)
        v_dataset = build_dataset(config.JSON_PATH.V_DATASET, self.vocab[1], config.FEATURE_DATASET)
        j_dataset = build_dataset(config.JSON_PATH.J_DATASET, self.vocab[2], config.FEATURE_DATASET)
        
        return [e_dataset, v_dataset, j_dataset]

    def load_datasets(self, config):
        self.dataset = self.load_feature_datasets(config)

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

    def start(self):
        pass

    def get_predictions(self):
        for idx, ckpt_path in enumerate(self.checkpoint_path):
            if not os.path.isfile(os.path.join(ckpt_path, 'best_model.pth')):
                logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
                raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

            self.load_checkpoint(os.path.join(ckpt_path, "best_model.pth"), idx)
            self.model[idx].eval()
        
        results = []
        overall_gens = {}
        overall_gts = {}
        for idx, data in enumerate(self.dataset):
            if data is not None:
                with tqdm(desc='Getting predictions on public test: ', unit='it', total=len(data)) as pbar:
                    for it, items in enumerate(data):
                        items = items.to(self.device)
                        with torch.no_grad():
                            outs, _ = self.model[idx].beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                        answers_gt = items.answers
                        answers_gen = self.vocab[idx].decode_answer(outs.contiguous().view(-1, self.vocab[idx].max_answer_length), join_words=False)
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
        }, open(os.path.join(self.results_path, "public_test_results.json"), "w+"), ensure_ascii=False)

        