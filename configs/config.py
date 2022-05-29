from yacs.config import CfgNode

config = CfgNode()

## configuration for paths
config.path = CfgNode()
config.path.train_json_path = "features/annotations/OpenViVQA/openvivqa_train.json"
config.path.dev_json_path = "features/annotations/OpenViVQA/openvivqa_dev.json"
config.path.test_json_path = "features/annotations/OpenViVQA/openvivqa_test.json"
config.path.image_features_path = "features/region_features/OpenViVQA/faster_rcnn"
config.path.images_path = None

## configuration for training
config.training = CfgNode()
config.training.checkpoint_path = "saved_models"
config.training.learning_rate = 1.
config.training.warmup = 10000
config.training.get_scores = False
config.training.training_beam_size = 5
config.training.evaluating_beam_size = 5

## model configuration
config.model = CfgNode()
config.model.name = "modified_mcan_using_region"
config.model.nhead = 8
config.model.nlayers = 3
config.model.d_model = 512
config.model.d_k = 64
config.model.d_v = 64
config.model.d_ff = 2048
config.model.d_feature = 2048
config.model.dropout = .5
config.model.embedding_dim = 300

## pretrained language model components

config.model.pretrained_language_model_name = None  # phobert_base
                                                    # phobert_large
                                                    # bartpho_syllable
                                                    # bartpho_word      
config.model.pretrained_language_model = None # PhoBERTModel
                                              # BARTPhoModel
                                              # GPT2Model

config.model.language_model_hidden_size = 768

## embedding modules
config.model.visual_embedding = "standard_visual_embedding"
config.model.language_embedding = "lstm_language_embedding"

## fusion modules
config.model.fusion = CfgNode()
config.model.fusion.args = CfgNode()
config.model.fusion.module = "standard_fusion_module"

### fusion encoder module
config.model.fusion.encoder = CfgNode()
config.model.fusion.encoder.args = CfgNode()
config.model.fusion.encoder.args.total_memory = None
config.model.fusion.encoder.args.use_aoa = False
config.model.fusion.encoder.module = "encoder"  # encoder
                                                # guided_encoder
config.model.fusion.encoder.self_attention = CfgNode()
config.model.fusion.encoder.self_attention.module = None  # scaled_dot_product_attention
                                                            # augmented_geometry_scaled_dot_product_attention
                                                            # augmented_memory_scaled_dot_product_attention
                                                            # apdative_scaled_dot_product_attention
config.model.fusion.encoder.self_attention.args = CfgNode()

### fusion guided-encoder module
config.model.fusion.guided_encoder = CfgNode()
config.model.fusion.guided_encoder.args = CfgNode()
config.model.fusion.guided_encoder.args.use_aoa = False
config.model.fusion.guided_encoder.module = "guided_encoder"  # encoder
                                                              # guided_encoder
config.model.fusion.guided_encoder.self_attention = CfgNode()
config.model.fusion.guided_encoder.self_attention.module = None # scaled_dot_product_attention
                                                                # augmented_geometry_scaled_dot_product_attention
                                                                # augmented_memory_scaled_dot_product_attention
                                                                # apdative_scaled_dot_product_attention
config.model.fusion.guided_encoder.self_attention.args = CfgNode()

config.model.fusion.guided_encoder.guided_attention = CfgNode()
config.model.fusion.guided_encoder.guided_attention.module = None  # scaled_dot_product_attention
                                                                  # augmented_geometry_scaled_dot_product_attention
                                                                  # augmented_memory_scaled_dot_product_attention
                                                                  # apdative_scaled_dot_product_attention
config.model.fusion.guided_encoder.guided_attention.args = CfgNode()

### decoder module
config.model.decoder = CfgNode()
config.model.decoder.args = CfgNode()
config.model.decoder.args.use_aoa = False
config.model.decoder.module = "decoder" # decoder
                                        # meshed_decoder
                                        # adaptive_decoder
config.model.decoder.self_attention = CfgNode()
config.model.decoder.self_attention.module = None # scaled_dot_product_attention
                                                        # augmented_geometry_scaled_dot_product_attention
                                                        # augmented_memory_scaled_dot_product_attention
                                                        # apdative_scaled_dot_product_attention
config.model.decoder.self_attention.args = CfgNode()

config.model.decoder.enc_attention = CfgNode()
config.model.decoder.enc_attention.module = None  # scaled_dot_product_attention
                                                        # augmented_geometry_scaled_dot_product_attention
                                                        # augmented_memory_scaled_dot_product_attention
                                                        # apdative_scaled_dot_product_attention
config.model.decoder.enc_attention.args = CfgNode()

# dataset configuration
config.dataset = CfgNode()
config.dataset.batch_size = 32
config.dataset.workers = 2

config.dataset.tokenizer = None # vncorenlp
                                # pyvi
                                # spacy

config.dataset.word_embedding = None  # fasttext.vi.300d
                                      # phow2v.syllable.100d
                                      # phow2v.syllable.300d
                                      # phow2v.word.100d
                                      # phow2v.word.300d

config.dataset.min_freq = 1

def get_default_config():
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return config.clone()