from models.modules.attentions import *
from models.modules.language_models import *
from models.modules.encoders import *
from models.modules.decoders import *
from models.modules.embeddings import *

attentions = {
    "scaled_dot_product_attention": ScaledDotProductAttention,
    "augmented_geometry_scaled_dot_product_attention": AugmentedGeometryScaledDotProductAttention,
    "augmented_memory_scaled_dot_product_attention": AugmentedMemoryScaledDotProductAttention,
    "apdative_scaled_dot_product_attention": AdaptiveScaledDotProductAttention,
    None: None
}

encoders = {
    "encoder": Encoder,
    "guided_encoder": GuidedEncoder
}

decoders = {
    "decoder": Decoder,
    "meshed_decoder": MeshedDecoder,
    "apdaptive_decoder": AdaptiveDecoder
}

pretrained_language_model_names = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "bartpho_syllable": "vinai/bartpho-syllable",
    "bartpho_word": "vinai/bartpho-word",
    "gpt_2": "NlpHUST/gpt-neo-vi-small"
}

pretrained_language_model = {
    "bert_model": BERTModel,
    "pho_bert_model": PhoBERTModel,
    "bart_pho_model": BARTPhoModel,
    "gpt_2": GPT2Model
}

word_embedding = {
    "fasttex": "fasttext.vi.300d",
    "phow2v_syllable_100": "phow2v.syllable.100d",
    "phow2v_syllable_300": "phow2v.syllable.300d",
    "phow2v_word_100": "phow2v.word.100d",
    "phow2v_word_300": "phow2v.word.300d"
}

visual_embeding = {
    "standard_visual_embedding": VisualEmbedding
}

language_embedding = {
    "lstm_language_embedding": LSTMTextEmbedding
}