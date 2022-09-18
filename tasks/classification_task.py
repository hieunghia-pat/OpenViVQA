from data_utils.vocab import ClassificationVocab
from data_utils.dataset import FeatureClassificationDataset
from .base_task import BaseTask
from builders.task_builder import META_TASK

@META_TASK.register()
class ClassificationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def load_vocab(self, config):
        vocab = ClassificationVocab(config.DATASET.VOCAB)

        return vocab

    def load_feature_datasets(self, config):
        train_dataset = FeatureClassificationDataset(config.JSON_PATH.TRAIN, self.vocab, config)
        dev_dataset = FeatureClassificationDataset(config.JSON_PATH.DEV, self.vocab, config)
        test_dataset = FeatureClassificationDataset(config.JSON_PATH.TEST, self.vocab, config)

        return train_dataset, dev_dataset, test_dataset