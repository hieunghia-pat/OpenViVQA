from .modules.stacmr_modules import (
    ObjectEncoder,
    OCREncoder,
    EncoderImagePrecompAttn,
    EncoderText,
    EncoderRNN,
    DecoderRNN)
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class VSRN(nn.Module):
    def __init__(self, config, vocab=None):
        super().__init__()
        # vocab_size, word_dim, embed_size, num_layers, use_abs=False
        self.vocab = vocab
        self.config = config
        self.d_model = self.config.D_MODEL
        self.obj_enc = ObjectEncoder(obj_in_dim=self.config.OBJECT_EMBEDDING.D_FEATURE,
                                     hidden_size=self.config.D_MODEL)

        self.ocr_enc = OCREncoder(ocr_in_dim=self.config.OCR_EMBEDDING.D_FEATURE,
                                  hidden_size=self.config.D_MODEL)

        self.img_enc = EncoderImagePrecompAttn(embed_size=self.config.D_MODEL,
                                               use_abs=self.config.ENCODER.USE_ABS,
                                               no_imgnorm=self.config.ENCODER.NO_IMGNORM)

        self.txt_enc = EncoderText(vocab_size=len(self.vocab),
                                   word_dim=self.config.TEXT_EMBEDDING.D_EMBEDDING,
                                   embed_size=self.config.D_MODEL,
                                   num_layers=self.config.RNN.NUM_LAYERS,
                                   use_abs=self.config.ENCODER.USE_ABS)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        #   captioning elements

        self.encoder = EncoderRNN(
            dim_vid=self.config.D_MODEL,
            dim_hidden=self.config.D_MODEL,
            bidirectional=self.config.RNN.BIDIRECTIONAL,
            input_dropout_p=self.config.INPUT_DROPOUT_P,
            rnn_cell=self.config.RNN.RNN_TYPE,
            rnn_dropout_p=self.config.RNN.RNN_DROPOUT_P)

        self.decoder = DecoderRNN(
            vocab_size=len(self.vocab),
            max_len=self.config.CLASSIFIER.MAX_LEN,
            dim_hidden=self.config.D_MODEL,
            dim_word=self.config.TEXT_EMBEDDING.D_EMBEDDING,
            input_dropout_p=self.config.INPUT_DROPOUT_P,
            rnn_cell=self.config.RNN.RNN_TYPE,
            rnn_dropout_p=self.config.RNN.RNN_DROPOUT_P,
            bidirectional=self.config.RNN.BIDIRECTIONAL)

        self.caption_model = S2VTAttModel(self.encoder, self.decoder)

        if torch.cuda.is_available():
            self.caption_model.cuda()

    def forward(self,
                item,
                mode='train'):

        obj_boxes = item.grid_boxes.squeeze().to(self.config.DEVICE)
        obj_features = item.grid_features.to(self.config.DEVICE)
        ocr_boxes = item.ocr_boxes.squeeze().to(self.config.DEVICE)
        ocr_token_embeddings = item.ocr_fasttext_features.to(self.config.DEVICE)
        ocr_rec_features = item.ocr_rec_features.to(self.config.DEVICE)
        ocr_det_features = item.ocr_det_features.to(self.config.DEVICE)
        
        answer_tokens = self.generate_answer_tokens(item.answer, item.ocr_tokens)
        shifted_right_answer_tokens = torch.zeros_like(answer_tokens).fill_(self.vocab.padding_idx)
        shifted_right_answer_tokens[:-1] = answer_tokens[1:]
        
        answer_tokens = torch.where(answer_tokens == self.vocab.eos_idx, self.vocab.padding_idx, answer_tokens) # remove eos_token in answer
        answer_mask = torch.where(answer_tokens > 0, 1, 0)
        
        caption_tokens = answer_tokens.squeeze().to(self.config.DEVICE)
        caption_masks = answer_mask.squeeze().to(self.config.DEVICE)
        
        B, _ = caption_masks.shape
        temp = np.zeros(B)
        for i in range(len(temp)):
            temp[i]=_

        objects = self.obj_enc(obj_boxes, obj_features)
        ocrs = self.ocr_enc(ocr_boxes,
                            ocr_token_embeddings,
                            ocr_rec_features,
                            ocr_det_features)

        cap_emb = self.txt_enc(caption_tokens, list(temp))

        img_emb, GCN_img_emd = self.img_enc(objects, ocrs)

        if mode == 'train':
            seq_probs, predicted_token = self.caption_model(vid_feats=GCN_img_emd,
                                                            target_variable=caption_tokens,
                                                            mode=mode)
            out = {
                'img_emb': img_emb,
                'cap_emb': cap_emb,
                'scores': seq_probs,
                'GCN_img_emd': GCN_img_emd
            }

        if mode == 'inference':
            seq_probs, predicted_token = self.caption_model(vid_feats=GCN_img_emd,
                                                            target_variable=None,
                                                            mode=mode)
            out = {'img_emb': img_emb,
                   'cap_emb': cap_emb,
                   'scores': seq_probs,
                   'GCN_img_emd': GCN_img_emd,
                   'predicted_token': predicted_token
                   }

        return out
    
    def generate_answer_tokens(self, answers, ocr_tokens, mask=False):
        answer_tokens = []
        for i in range(len(answers)):
            answer_tokens.append(self.vocab.encode_answer(answers[i], ocr_tokens[i]))
        
        answer_tokens = torch.stack(answer_tokens).to(self.config.DEVICE)
        
        if mask == True:
            result = torch.where(answer_tokens > 0, 1, 0)
        else:
            result = answer_tokens
        return result


class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, vid_feats, target_variable=None,
                mode='train'):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """

        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        seq_prob, seq_preds = self.decoder(encoder_outputs,
                                           encoder_hidden,
                                           target_variable,
                                           mode)
        return seq_prob, seq_preds