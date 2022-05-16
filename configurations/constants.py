import enum

class Constants(enum.Enum):

    # attentions
    ScaledDotProductAttention = "scaled_dot_product_attention"
    AugmentedGeometryScaledDotProductAttention = "augmented_geometry_scaled_dot_product_attention"
    AugmentedMemoryScaledDotProductAttention = "augmented_geometry_scaled_dot_product_attention"
    AdaptiveScaledDotProductAttention = "adaptive_scaled_dot_product_attention"

    # pretrained language model names
    phobert_base = "vinai/phobert-base"
    phobert_large = "vinai/phobert-large"
    bartpho_base = "vinai/bartpho-syllable"
    batpho_large = "vinai/bartpho-word"
    gpt_2 = "NlpHUST/gpt-neo-vi-small"

    # pretrained language models
    PhoBERTModel = "pho_bert_model"
    BARTPhoModel = "bart_pho_model"
    ViGPTModel = "gpt_2"

    # tokenizers
    vncorenlp = "vncorenlp"
    pyvi = "pyvi"
    spacy = "spacy"

    # word embeddings
    fasttext = "fasttext.vi.300d"
    phow2v_syllable_100 = "phow2v.syllable.100d"
    phow2v_syllable_300 = "phow2v.syllable.300d"
    phow2v_word_100 = "phow2v.word.100d"
    phow2v_word_300 = "phow2v.word.300d"