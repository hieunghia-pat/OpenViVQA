import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import NLLLoss

from utils.logging_utils import setup_logger
from tasks.open_ended_task import OpenEndedTask
from builders.task_builder import META_TASK
import evaluation
import evaluate
from pycocoevalcap.cider.cider import Cider

import os
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

from torch.autograd import Variable

logger = setup_logger()


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


@META_TASK.register()
class TrainingStacMR(OpenEndedTask):
    def __init__(self, config):
        super().__init__(config)

        self.crit = LanguageModelCriterion()
        self.criterion = ContrastiveLoss(margin=config.MODEL.LOSS_FN.MARGIN,
                                         measure=config.MODEL.LOSS_FN.MEASURE,
                                         max_violation=config.MODEL.LOSS_FN.MAX_VIOLATION)

        self.crit.to(self.device)
        self.criterion.to(self.device)
        #self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    with torch.no_grad():
                        results = self.model(items, model='inference')

                    out = results["predicted_token"].contiguous()
                    seq_prob = out['scores']
                    img_emb = results['img_emb']
                    cap_emb = results['cap_emb']
                    # out = F.log_softmax(out, dim=-1)
                    
                    shifted_right_answer_tokens = items.shifted_right_answer_tokens
                    # loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                    
                    answer_mask = self.model.generate_answer_tokens(items.answer,
                                                                    items.answer_tokens,
                                                                    mask=True)
                    
                    caption_loss = self.crit(seq_prob, 
                                             shifted_right_answer_tokens, 
                                             answer_mask)
                
                    retrieval_loss = self.criterion(img_emb, cap_emb)
                    
                    loss = 2.0 * retrieval_loss + caption_loss
                    
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
                    results = self.model(items, mode='inference')
                outs = results["predicted_token"]

                answers_gt = items.answers
                answers_gen = self.vocab.decode_answer(outs.contiguous(),
                                                       items.ocr_tokens,
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
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                results = self.model(items, mode='train')
                seq_probs = results["scores"].contiguous()
                img_emb = results['img_emb']
                cap_emb = results['cap_emb']
                #out = F.log_softmax(out, dim=-1)

                shifted_right_answer_tokens = items.shifted_right_answer_tokens
                self.optim.zero_grad()
                # loss = self.loss_fn(out.view(-1, out.shape[-1]), shifted_right_answer_tokens.view(-1))
                answer_mask = self.model.generate_answer_tokens(items.answer,
                                                                items.answer_tokens,
                                                                mask=True)
                caption_loss = self.crit(seq_probs, 
                                         shifted_right_answer_tokens, 
                                         answer_mask)
                
                retrieval_loss = self.criterion(img_emb, cap_emb)
                
                loss = 2.0 * retrieval_loss + caption_loss
                
                loss.backward()

                self.optim.step()
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"), weights_only=True)
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
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'last_model.pth')):
            logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"), weights_only=True)

        self.model.eval()
        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions: ', unit='it', total=len(self.test_dict_dataloader)) as pbar:
            for it, items in enumerate(self.test_dict_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    result = self.model(items, mode='inference')
                outs = result["predicted_token"]

                answers_gt = items.answers
                answers_gen, in_fixed_vocab = self.vocab.decode_answer_with_determination(outs.contiguous().view(-1, self.vocab.max_answer_length),
                                                        items.ocr_tokens, join_words=False)
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
        logger.info("Evaluation scores on test: %s", scores)

        json.dump({
            "results": results,
            **scores,
        }, open(os.path.join(self.checkpoint_path, "test_results.json"), "w+"), ensure_ascii=False)
