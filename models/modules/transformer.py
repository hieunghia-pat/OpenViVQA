import torch
from torch import nn
from models.answering_model import AnsweringModel
from models.modules.embeddings import SinusoidPositionalEmbedding
from configs.constants import *
from data_utils.vocab import Vocab
from yacs.config import CfgNode
from data_utils.feature import Feature

class StandardTransformer(AnsweringModel):
    def __init__(self, bos_idx, encoder, decoder, use_img_pos=False):
        super(StandardTransformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.use_img_pos = use_img_pos
        if self.use_img_pos:
            self.sinusoid_pos_embedding = SinusoidPositionalEmbedding(encoder.d_model, normalize=True)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input, tokens, boxes=None, grid_size=None):
        pos_emb = self.sinusoid_pos_embedding(input) if self.use_img_pos else None
        enc_output, mask_enc = self.encoder(input, boxes, grid_size, positional_emb=pos_emb)
        dec_output = self.decoder(tokens, enc_output, mask_encoder=mask_enc, positional_emb=pos_emb)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, boxes=None, grid_size=None, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            pos_emb = self.sinusoid_pos_embedding(visual) if self.use_img_pos else None
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual, boxes, grid_size, positional_emb=pos_emb)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, mask_encoder=self.mask_enc, positional_emb=pos_emb)

class FusionTransformer(AnsweringModel):
    def __init__(self, vocab: Vocab, config: CfgNode):
        super(FusionTransformer, self).__init__()

        self.vocab = vocab
        self.config = config

        self.d_model = self.config.model.d_model

        self.construct_model()

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_encoder(self):
        model_config = self.config.model
        encoder_config = self.config.model.fusion.encoder

        encoder = encoders[encoder_config.module]
        encoder_args = dict(encoder_config.args)
        encoder_self_attention = attentions[encoder_config.self_attention.module]
        encoder_self_attention_args = dict(encoder_config.self_attention.args)

        return encoder(N=model_config.nlayers, d_model=self.d_model, d_k=model_config.d_k, 
                        d_v=model_config.d_v, d_ff=model_config.d_ff, dropout=model_config.dropout, 
                        attention_module=encoder_self_attention, 
                        attention_module_kwargs=encoder_self_attention_args, 
                        **encoder_args)

    def get_guided_encoder(self):
        model_config = self.config.model
        guided_encoder_config = self.config.model.fusion.guided_encoder

        guided_encoder = encoders[guided_encoder_config.module]
        guided_encoder_args = dict(guided_encoder_config.args)
        guided_encoder_self_attention = attentions[guided_encoder_config.self_attention.module]
        guided_encoder_self_attention_args = dict(guided_encoder_config.self_attention.args)
        guided_encoder_guided_attention = attentions[guided_encoder_config.guided_attention.module]
        guided_encoder_guided_attention_args = dict(guided_encoder_config.guided_attention.args)

        return guided_encoder(N=model_config.nlayers, d_model=self.d_model, d_k=model_config.d_k, 
                                d_v=model_config.d_v, d_ff=model_config.d_ff, dropout=model_config.dropout, 
                                self_attention_module=guided_encoder_self_attention, 
                                self_attention_module_kwargs=guided_encoder_self_attention_args,
                                guided_attention_module=guided_encoder_guided_attention, 
                                guided_attention_module_kwargs=guided_encoder_guided_attention_args, 
                                **guided_encoder_args)

    def get_decoder(self):
        model_config = self.config.model
        decoder_config = self.config.model.decoder

        decoder = decoders[decoder_config.module]
        decoder_args = dict(decoder_config.args)
        decoder_self_attention = attentions[decoder_config.self_attention.module]
        decoder_self_attention_args = dict(decoder_config.self_attention.args)
        decoder_enc_attention = attentions[decoder_config.enc_attention.module]
        decoder_enc_attention_args = dict(decoder_config.enc_attention.args)

        return decoder(vocab_size=len(self.vocab), max_len=self.vocab.max_answer_length, N_dec=model_config.nlayers,
                        padding_idx=self.vocab.padding_idx, d_model=self.d_model,
                        d_k=model_config.d_k, d_v=model_config.d_v,
                        d_ff=model_config.d_ff, dropout=model_config.dropout,
                        self_att_module=decoder_self_attention, enc_att_module=decoder_enc_attention,
                        self_att_module_kwargs=decoder_self_attention_args,
                        enc_att_module_kwargs=decoder_enc_attention_args, **decoder_args)

    def get_visual_embedding(self):
        model_config = self.config.model

        embedding = visual_embeding[model_config.visual_embedding]
        return embedding(
            model_config.d_model,
            model_config.dropout
        )

    def get_language_embedding(self):
        model_config = self.config.model

        embedding = language_embedding[model_config.language_embedding]
        return embedding(
            self.vocab,
            model_config.embedding_dim,
            model_config.d_model,
            model_config.dropout
        )

    def construct_model(self):
        # embedding modules
        self.visual_embedding = self.get_visual_embedding()
        self.language_embedding = self.get_language_embedding()

        # fusion modules
        self.encoder = self.get_encoder()
        self.guided_encoder = self.get_guided_encoder()
        self.visual_fc = nn.Linear(self.d_model, self.d_model)
        self.question_fc = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

        # generative modules
        self.decoder = self.get_decoder()

    def forward(self, visuals, linguistics):
        visual_features, visual_masks = self.visual_embedding(visuals.features)
        question_features, question_masks = self.language_embedding(linguistics.question_tokens)

        visual_inputs = Feature({
            "features": visual_features,
            "boxes": visuals.boxes,
            "masks": visual_masks
        })

        linguistic_inputs = Feature({
            "features": question_features,
            "masks": question_masks
        })

        question_features = self.encoder(linguistic_inputs)
        guided_features = self.guided_encoder(visual_inputs, linguistic_inputs)

        # Fuse guided features and question features
        question_features = self.question_fc(question_features)
        guided_features = self.visual_fc(guided_features)
        fused_features = self.layer_norm(guided_features + question_features)

        answer_tokens = linguistics.answer_tokens
        fused_inputs = Feature({
            "features": fused_features,
            "masks": visual_masks
        })
        dec_output = self.decoder(answer_tokens, fused_inputs)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visuals, linguistics, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                visual_features, visual_masks = self.visual_embedding(visuals.features)
                question_features, question_masks = self.language_embedding(linguistics.question_tokens)

                visual_inputs = Feature({
                    "features": visual_features,
                    "boxes": visuals.boxes,
                    "masks": visual_masks
                })

                linguistic_inputs = Feature({
                    "features": question_features,
                    "masks": question_masks
                })

                question_features = self.encoder(linguistic_inputs)
                guided_features = self.guided_encoder(visual_inputs, linguistic_inputs)

                # Fuse guided features and question features
                question_features = self.question_fc(question_features)
                guided_features = self.visual_fc(guided_features)
                fused_features = self.layer_norm(guided_features + question_features)

                fused_inputs = Feature({
                    "features": fused_features,
                    "masks": visual_masks
                })

                self.enc_out = fused_features,
                self.mask_enc = visual_masks
                
                bs = visual_features.shape[0]
                it = torch.zeros((bs, 1)).long().fill_(self.bos_idx)
            else:
                it = prev_output

        dec_output = self.decoder(it, fused_inputs)
        return dec_output