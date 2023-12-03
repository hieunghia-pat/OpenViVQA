import torch
from torch import nn
from torch.nn import functional as F

from tasks.open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
import evaluation

import os
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

@META_TASK.register()
class TrainingMMF(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)

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
                                                        items.ocr_tokens, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

        scores, _ = evaluation.compute_scores(gts, gens)

        return scores

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
                answers_gen, in_fixed_vocab = self.vocab.decode_answer_with_determination(
                  outs.contiguous().view(-1, self.vocab.max_answer_length),
                  items.ocr_tokens,
                  join_words=False)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i, in_fixed_vocab_i) in enumerate(zip(answers_gt, answers_gen, in_fixed_vocab)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = (gen_i, in_fixed_vocab_i)
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
