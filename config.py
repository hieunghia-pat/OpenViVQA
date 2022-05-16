from yacs.config import CfgNode
from configurations.constants import Constants

config = CfgNode()

# configuration for paths
config.path = CfgNode()
config.path.train_json_path = "features/annotations/OpenViVQA/openvivqa_train.json"
config.path.dev_json_path = "features/annotations/OpenViVQA/openvivqa_dev.json"
config.path.test_json_path = "features/annotations/OpenViVQA/openvivqa_test.json"
config.path.image_features_path = "features/region_features/OpenViVQA/faster_rcnn"
config.path.images_path = ""

# configuration for training
config.training = CfgNode()
config.training.checkpoint_path = "saved_models"
config.training.start_from = None
config.training.learning_rate = 1.
config.training.warmup = 10000
config.training.get_scores = False
config.training.training_beam_size = 5
config.training.evaluating_beam_size = 5

# model configuration
config.model = CfgNode()
config.model.model_name = "modified_mcan_using_region"
config.model.nhead = 8
config.model.nlayers = 3
config.model.d_model = 512
config.model.d_k = 64
config.model.d_v = 64
config.model.d_ff = 2048
config.model.d_feature = 2048
config.model.dropout = .1
config.model.pretrained_language_model_name = Constants.phobert_base.name  # vinai/phobert-base
                                                    # vinai/phobert-large
                                                    # vinai/bartpho-syllable
                                                    # vinai/bartpho-word
                                                    # NlpHUST/gpt-neo-vi-small
config.model.pretrained_language_model = None   # PhoBERTModel
                                                # BARTPhoModel
                                                # ViGPTModel

config.model.language_model_hidden_size = 768

config.model.transformer = CfgNode()
config.model.transformer.transformer_args = CfgNode()

config.model.transformer.encoder = CfgNode()
config.model.transformer.encoder.encoder_self_attention = Constants.ScaledDotProductAttention.name
config.model.transformer.encoder.encoder_self_attention_args = CfgNode()
config.model.transformer.encoder.encoder_args = CfgNode()

config.model.transformer.decoder = CfgNode()
config.model.transformer.decoder.decoder_args = CfgNode()
config.model.transformer.decoder.decoder_self_attention = Constants.ScaledDotProductAttention.name
config.model.transformer.decoder.decoder_enc_attention = Constants.ScaledDotProductAttention.name
config.model.transformer.decoder.decoder_self_attention_args = CfgNode()
config.model.transformer.decoder.decoder_enc_attention_args = CfgNode()

# dataset configuration
config.dataset = CfgNode()
config.dataset.batch_size = 32
config.dataset.workers = 2
config.dataset.tokenizer = None         # vncorenlp
                                        # pyvi
                                        # spacy
config.dataset.word_embedding = None    # "fasttext.vi.300d"
                                        # "phow2v.syllable.100d"
                                        # "phow2v.syllable.300d"
                                        # "phow2v.word.100d"
                                        # "phow2v.word.300d"
config.dataset.min_freq = 1

def get_default_config():
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return config.clone()