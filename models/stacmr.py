from .modules.stacmr_modules import ObjectEncoder, OCREncoder, EncoderImagePrecompAttn, EncoderText, EncoderRNN, DecoderRNN
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
        self.obj_enc = ObjectEncoder(obj_in_dim=self.config.OBJ_IN_DIM,
                                     hidden_size=self.config.EMBED_SIZE)

        self.ocr_enc = OCREncoder(ocr_in_dim=self.config.OCR_IN_DIM,
                                  hidden_size=self.config.EMBED_SIZE)

        self.img_enc = EncoderImagePrecompAttn(embed_size=self.config.EMBED_SIZE,
                                               use_abs=self.config.USE_ABS,
                                               no_imgnorm=self.config.NO_IMGNORM)

        self.txt_enc = EncoderText(vocab_size=self.config.VOCAB_SIZE,
                                   word_dim=self.config.WORD_DIM,
                                   embed_size=self.config.EMBED_SIZE,
                                   num_layers=self.config.NUM_LAYERS,
                                   use_abs=self.config.USE_ABS)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        #####   captioning elements

        self.encoder = EncoderRNN(
            dim_vid=self.config.EMBED_SIZE,
            dim_hidden=self.config.EMBED_SIZE,
            bidirectional=self.config.BIDIRECTIONAL,
            input_dropout_p=self.config.INPUT_DROPOUT_P,
            rnn_cell=self.config.RNN_TYPE,
            rnn_dropout_p=self.config.RNN_DROPOUT_P)

        self.decoder = DecoderRNN(
            vocab_size=self.config.VOCAB_SIZE,
            max_len=self.config.MAX_LEN,
            dim_hidden=self.config.EMBED_SIZE,
            dim_word=self.config.WORD_DIM,
            input_dropout_p=self.config.INPUT_DROPOUT_P,
            rnn_cell=self.config.RNN_TYPE,
            rnn_dropout_p=self.config.RNN_DROPOUT_P,
            bidirectional=self.config.BIDIRECTIONAL)

        self.caption_model = S2VTAttModel(self.encoder, self.decoder)

        if torch.cuda.is_available():
            self.caption_model.cuda()

    def forward(self,
                item,
                mode='train'):

        obj_boxes = item.grid_boxes.to(self.config.DEVICE)
        obj_features = item.grid_features.to(self.config.DEVICE)
        ocr_boxes = item.ocr_boxes.to(self.config.DEVICE)
        ocr_token_embeddings = item.ocr_token_embeddings.to(self.config.DEVICE)
        ocr_rec_features = item.ocr_rec_features.to(self.config.DEVICE)
        ocr_det_features = item.ocr_det_features.to(self.config.DEVICE)
        caption_tokens = item.answer_tokens.squeeze().to(self.config.DEVICE)
        caption_masks = item.answer_mask.squeeze().to(self.config.DEVICE)
        
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
                'scores': seq_probs, # seg_probs
                'GCN_img_emd': GCN_img_emd
            }

        if mode == 'inference':
            seq_probs, predicted_token = self.caption_model(vid_feats=GCN_img_emd,
                                                            target_variable=None,
                                                            mode=mode)
            out = {'img_emb': img_emb,
                   'cap_emb': cap_emb,
                   'scores': seq_probs, # seg_probs
                   'GCN_img_emd': GCN_img_emd,
                   'predicted_token': predicted_token
                   }

        return out


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