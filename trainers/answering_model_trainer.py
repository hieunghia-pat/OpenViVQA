from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data_utils.vocab import Vocab
from data_utils.utils import *
from models.modules.transformer import FusionTransformer
from data_utils.dataset import *
import evaluation
from evaluation import Cider, PTBTokenizer
from data_utils.feature import Feature

import multiprocessing
from tqdm import tqdm
import itertools
from typing import Tuple, Union
import random
from shutil import copyfile
from yacs.config import CfgNode

device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self,  model: FusionTransformer, 
                        train_datasets: Tuple[FeatureDataset, DictionaryDataset],
                        val_datasets: Tuple[FeatureDataset, DictionaryDataset],
                        test_datasets: Tuple[Union[FeatureDataset, None], Union[DictionaryDataset, None]],
                        vocab: Vocab,
                        config: CfgNode,
                        collate_fn=collate_fn):
        self.model = model
        self.vocab = vocab
        self.config = config

        self.optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)
        
        self.epoch = 0

        self.train_dataset, self.train_dict_dataset = train_datasets
        self.val_dataset, self.val_dict_dataset = val_datasets

        # creating iterable-dataset data loader
        self.train_dataloader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.workers,
            collate_fn=collate_fn
        )
        self.val_dataloader = data.DataLoader(
            dataset=self.val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.workers,
            collate_fn=collate_fn
        )

        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = data.DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.dataset.batch_size // config.training.training_beam_size,
            shuffle=True,
            num_workers=config.dataset.workers,
            collate_fn=collate_fn
        )
        self.val_dict_dataloader = data.DataLoader(
            dataset=self.val_dict_dataset,
            batch_size=config.dataset.batch_size // config.training.training_beam_size,
            shuffle=True,
            num_workers=config.dataset.workers,
            collate_fn=collate_fn
        )
        
        self.test_dataset, self.test_dict_dataset = test_datasets

        if self.test_dataset is not None:
            self.test_dataloader = data.DataLoader(
                dataset=self.test_dataset,
                batch_size=config.dataset.batch_size,
                shuffle=True,
                num_workers=config.dataset.workers,
                collate_fn=collate_fn
            )
        else:
            self.test_dataloader = None

        if self.test_dict_dataset is not None:
            self.test_dict_dataloader = data.DataLoader(
                dataset=self.test_dict_dataset,
                batch_size=config.dataset.batch_size // config.training.training_beam_size,
                shuffle=True,
                num_workers=config.dataset.workers,
                collate_fn=collate_fn
            )
        else:
            self.test_dict_dataloader = None

        self.train_cider = Cider(PTBTokenizer.tokenize(self.train_dataset.answers))

    def evaluate_loss(self, dataloader: data.DataLoader):
        # Calculating validation loss
        self.model.eval()
        running_loss = .0
        with tqdm(desc='Epoch %d - Validation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, sample in enumerate(dataloader):
                region_features = sample["region_features"]
                if region_features is not None:
                    region_features = region_features.to(device)

                grid_features = sample["grid_features"]
                if grid_features is not None:
                    grid_features = grid_features.to(device)

                assert region_features is not None or grid_features is not None, "both region-based features and grid-based features are None"

                region_boxes = sample["region_boxes"]
                grid_boxes = sample["grid_boxes"]

                visual = Feature({
                        "region_features": region_features,
                        "grid_features": grid_features,
                        "features": region_features if region_features is not None else grid_features,
                        "boxes": region_boxes if region_boxes is not None else grid_boxes
                    })

                question_tokens = sample["question_tokens"].to(device)
                answer_tokens = sample["answer_tokens"].to(device)
                shifted_right_answer_tokens = sample["shifted_right_answer_tokens"].to(device)

                linguistic = Feature({
                    "question_tokens": question_tokens,
                    "answer_tokens": answer_tokens
                })
                
                with torch.no_grad():
                    out = self.model(visual, linguistic).contiguous()
                
                loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_answer_tokens.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

        val_loss = running_loss / len(dataloader)

        return val_loss

    def evaluate_metrics(self, dataloader: data.DataLoader):
        self.model.eval()
        gen = {}
        gts = {}
        with tqdm(desc='Epoch %d - Evaluation' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for it, sample in enumerate(dataloader):
                region_features = sample["region_features"]
                if region_features is not None:
                    region_features = region_features.to(device)

                grid_features = sample["grid_features"]
                if grid_features is not None:
                    grid_features = grid_features.to(device)

                assert region_features is not None or grid_features is not None, "both region-based features and grid-based features are None"

                region_boxes = sample["region_boxes"]
                grid_boxes = sample["grid_boxes"]

                visual = Feature({
                        "region_features": region_features,
                        "grid_features": grid_features,
                        "region_boxes": region_boxes,
                        "grid_boxes": grid_boxes,
                        "features": region_features if region_features is not None else grid_features,
                        "boxes": region_boxes if region_boxes is not None else grid_boxes
                    })

                question_tokens = sample["question_tokens"].to(device)

                linguistic = Feature({
                    "question_tokens": question_tokens
                })

                with torch.no_grad():
                    outs, _ = self.model.beam_search(visual, linguistic, 
                                                    max_len=self.vocab.max_answer_length, eos_idx=self.vocab.eos_idx, 
                                                    beam_size=self.config.training.evaluating_beam_size, out_size=1)

                answers_gt = [" ".join(answer) for answer in sample["answers"]]
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = [gts_i, ]
                pbar.update()

        gts = evaluation.PTBTokenizer.tokenize(gts)
        gen = evaluation.PTBTokenizer.tokenize(gen)
        scores, _ = evaluation.compute_scores(gts, gen)

        return scores

    def train_xe(self):
        # Training with cross-entropy loss
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with cross-entropy loss' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, sample in enumerate(self.train_dataloader):
                region_features = sample["region_features"]
                if region_features is not None:
                    region_features = region_features.to(device)

                grid_features = sample["grid_features"]
                if grid_features is not None:
                    grid_features = grid_features.to(device)

                assert region_features is not None or grid_features is not None, "both region-based features and grid-based features are None"

                region_boxes = sample["region_boxes"]
                grid_boxes = sample["grid_boxes"]

                visual = Feature({
                        "region_features": region_features,
                        "grid_features": grid_features,
                        "features": region_features if region_features is not None else grid_features,
                        "boxes": region_boxes if region_boxes is not None else grid_boxes
                    })

                question_tokens = sample["question_tokens"].to(device)
                answer_tokens = sample["answer_tokens"].to(device)
                shifted_right_answer_tokens = sample["shifted_right_answer_tokens"].to(device)

                linguistic = Feature({
                    "question_tokens": question_tokens,
                    "answer_tokens": answer_tokens
                })
                
                out = self.model(visual, linguistic).contiguous()

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
        # Training with self-critical learning
        tokenizer_pool = multiprocessing.Pool()
        running_reward = .0
        running_reward_baseline = .0

        vocab: Vocab = self.train_dataset.vocab

        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training with self-critical learning' % self.epoch, unit='it', total=len(self.train_dict_dataloader)) as pbar:
            for it, sample in enumerate(self.train_dict_dataloader):
                region_features = sample["region_features"]
                if region_features is not None:
                    region_features = region_features.to(device)

                grid_features = sample["grid_features"]
                if grid_features is not None:
                    grid_features = grid_features.to(device)

                assert region_features is not None or grid_features is not None, "both region-based features and grid-based features are None"

                region_boxes = sample["region_boxes"]
                grid_boxes = sample["grid_boxes"]

                visual = Feature({
                        "region_features": region_features,
                        "grid_features": grid_features,
                        "region_boxes": region_boxes,
                        "grid_boxes": grid_boxes,
                        "features": region_features if region_features is not None else grid_features,
                        "boxes": region_boxes if region_boxes is not None else grid_boxes
                    })

                question_tokens = sample["question_tokens"].to(device)

                linguistic = Feature({
                    "question_tokens": question_tokens
                })

                outs, log_probs = self.model.beam_search(visual, linguistic,
                                                        max_len=vocab.max_answer_length, eos_idx=vocab.eos_idx,
                                                        beam_size=self.config.training.training_beam_size, 
                                                        out_size=self.config.training.training_beam_size)
                
                self.optim.zero_grad()

                # Rewards
                bs = question_tokens.shape[0]
                answers_gt = [[" ".join(answer)] for answer in sample["answers"]]
                answers_gen = vocab.decode_answer(outs.contiguous().view(-1, vocab.max_answer_length), join_words=True)
                answers_gt = list(itertools.chain(*([a, ] * self.config.training.training_beam_size for a in answers_gt)))
                answers_gen, answers_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [answers_gen, answers_gt])
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
        warm_up = self.config.training.warmup
        step += 1
        return (self.model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        checkpoint = torch.load(fname)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        print(f"resuming from epoch {checkpoint['epoch']} - validation loss {checkpoint['val_loss']} - best cider on val {checkpoint['best_val_cider']} - best cider on test {checkpoint['best_test_cider']}")

        return {
            "use_rl": checkpoint['use_rl'],
            "best_val_cider": checkpoint['best_val_cider'],
            "best_test_cider": checkpoint['best_test_cider'],
            "patience": checkpoint['patience'],
            "epoch": checkpoint["epoch"],
            "optimizer": checkpoint["optimizer"],
            "scheduler": checkpoint["scheduler"]
        }

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

        torch.save(dict_for_saving, os.path.join(self.config.training.checkpoint_path, 
                                                    self.config.model.name, 
                                                    "last_model.pth"))

    def train(self, checkpoint_filename: str = None):
        
        if checkpoint_filename is not None and os.path.isfile(checkpoint_filename):
            checkpoint = self.load_checkpoint(checkpoint_filename)
            use_rl = checkpoint["use_rl"]
            best_val_cider = checkpoint["best_val_cider"]
            best_test_cider = checkpoint["best_test_cider"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"]
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            use_rl = False
            best_val_cider = .0
            best_test_cider = .0
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
            val_cider = scores['CIDEr']

            if self.test_dict_dataloader is not None:
                scores = self.evaluate_metrics(self.test_dict_dataloader)
                print("Evaluation scores", scores)

            # Prepare for next epoch
            best = False
            if val_cider >= best_val_cider:
                best_val_cider = val_cider
                patience = 0
                best = True
            else:
                patience += 1

            switch_to_rl = False
            exit_train = False

            if patience == 5:
                if not use_rl:
                    use_rl = True
                    switch_to_rl = True
                    patience = 0
                    self.optim = Adam(self.model.parameters(), lr=5e-6)
                    print("Switching to RL")
                else:
                    print('patience reached.')
                    exit_train = True

            if switch_to_rl and not best:
                self.load_checkpoint(os.path.join(self.config.training.checkpoint_path, self.config.model.name, "best_model.pth"))

            self.save_checkpoint({
                'val_loss': val_loss,
                'val_cider': val_cider,
                'patience': patience,
                'best_val_cider': best_val_cider,
                'best_test_cider': best_test_cider,
                'use_rl': use_rl,
            })

            if best:
                copyfile(   
                            os.path.join(self.config.training.checkpoint_path, self.config.model.name, "last_model.pth"), 
                            os.path.join(self.config.training.checkpoint_path, self.config.model.name, "best_model.pth")
                        )

            if exit_train:
                break

            self.epoch += 1
            
            print("+"*10)

    def get_predictions(self, dataset: DictionaryDataset, checkpoint_filename: str=None, get_scores=True):
        if checkpoint_filename is not None and os.path.isfile(checkpoint_filename):
            self.load_checkpoint(checkpoint_filename)
        else:
            raise Exception("Checkpoint file name is not found")
            
        self.model.eval()
        results = []
        with tqdm(desc='Getting predictions: ', unit='it', total=len(dataset)) as pbar:
            for it, sample in enumerate(dataset):
                image_id = sample["image_id"]
                filename = sample["filename"]

                region_features = sample["region_features"]
                if region_features is not None:
                    region_features = torch.tensor(region_features).unsqueeze(0).to(device)

                grid_features = sample["grid_features"]
                if grid_features is not None:
                    grid_features = torch.tensor(grid_features).unsqueeze(0).to(device)

                assert region_features is not None or grid_features is not None, "both region-based features and grid-based features are None"

                region_boxes = sample["region_boxes"]
                if region_boxes is not None:
                    region_boxes = torch.tensor(region_boxes).unsqueeze(0).to(device)

                grid_boxes = sample["grid_boxes"]
                if grid_boxes is not None:
                    grid_boxes = torch.tensor(grid_boxes).unsqueeze(0).to(device)

                visual = Feature({
                        "region_features": region_features,
                        "grid_features": grid_features,
                        "region_boxes": region_boxes,
                        "grid_boxes": grid_boxes,
                        "features": region_features if region_features is not None else grid_features,
                        "boxes": region_boxes if region_boxes is not None else grid_boxes
                    })

                question_tokens = sample["question_tokens"].unsqueeze(0).to(device)

                linguistic = Feature({
                    "question_tokens": question_tokens
                })

                with torch.no_grad():
                    outs, _ = self.model.beam_search(visual, linguistic, 
                                                    max_len=self.vocab.max_answer_length, eos_idx=self.vocab.eos_idx, 
                                                    beam_size=self.config.training.evaluating_beam_size, out_size=1)

                answers_gt = [" ".join(answer) for answer in sample["answer"]]
                answers_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), join_words=False)
                gts = {}
                gens = {}
                for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gens['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = [gts_i, ]
                
                gts = evaluation.PTBTokenizer.tokenize(gts)
                gens = evaluation.PTBTokenizer.tokenize(gens)
                if get_scores:
                    scores, _ = evaluation.compute_scores(gts, gens)
                else:
                    scores = None

                results.append({
                    "image_id": image_id,
                    "filename": filename,
                    "gens": gens,
                    "gts": gts,
                    "scores": scores
                })
                
                pbar.update()

        return results