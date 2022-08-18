import torch
from torchvision import transforms
import re

def get_tokenizer(tokenizer: str):
    if tokenizer is None:
        return lambda s: s 
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("Please install PyVi package. "
                  "See the docs at https://github.com/trungtv/pyvi for more information.")
    elif tokenizer == "spacy":
        try:
            from spacy.lang.vi import Vietnamese
            return Vietnamese()
        except ImportError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
    elif tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            # annotator = VnCoreNLP(r"data_utils/vncorenlp/VnCoreNLP-1.1.1.jar", port=9000, annotators="wseg", max_heap_size='-Xmx500m')
            annotator = VnCoreNLP(address="http://127.0.0.1", port=9000, max_heap_size='-Xmx500m')

            def tokenize(s: str):
                words = annotator.tokenize(s)[0]
                return " ".join(words)

            return tokenize
        except ImportError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise
        except AttributeError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise

def preprocess_sentence(sentence, tokenizer: str):
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r".\. *\. *\. *", " ... ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    sentence = get_tokenizer(tokenizer)(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens

def get_transform(target_size):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def reporthook(t):
    """
    https://github.com/tqdm/tqdm.
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def unk_init(token, dim):
    '''
        For default:
            + <pad> is 0
            + <sos> is 1
            + <eos> is 2
            + <unk> is 3
    '''

    if token in ["<pad>", "<p>"]:
        return torch.zeros(dim)
    if token in ["<sos>", "<bos>", "<s>"]:
        return torch.ones(dim)
    if token in ["<eos>", "</s>"]:
        return torch.ones(dim) * 2
    return torch.ones(dim) * 3

def default_value():
    return None

def classification_collate_fn(samples):
    image_ids = []
    region_features = []
    grid_features = []
    region_boxes = []
    grid_boxes = []
    questions = []
    answers = []
    max_seq_len = 0
    for sample in samples:
        image_id = sample["image_id"]
        region_feature = sample["region_features"]
        grid_feature = sample["grid_features"]
        region_box = sample["region_boxes"]
        grid_box = sample["grid_boxes"]
        question = sample["question"]
        answer = sample["answer"]

        if region_feature is not None and max_seq_len < region_feature.shape[0]:
            max_seq_len = region_feature.shape[0]

        if grid_feature is not None and max_seq_len < grid_feature.shape[0]:
            max_seq_len = grid_feature.shape[0]

        assert max_seq_len != 0, "both region-based features and grid-based features are not presented"

        if image_id is not None:
            image_ids.append(image_id)
        if region_box is not None:
            region_boxes.append(torch.tensor(region_box))
        if grid_box is not None:
            grid_boxes.append(torch.tensor(grid_box))
        if question is not None:
            questions.append(question)
        if answer is not None:
            answers.append(answer)

        if region_feature is not None:
            region_features.append(torch.tensor(region_feature))
        
        if grid_feature is not None:
            grid_features.append(torch.tensor(grid_feature))

    if len(region_features) > 0:
        zero_region_feature = torch.zeros_like(region_features[-1][-1]).unsqueeze(0) # padding tensor for region features (1, dim)
    if len(grid_features) > 0:
        zero_grid_feature = torch.zeros_like(grid_features[-1][-1]).unsqueeze(0) # padding tensor for grid features (1, dim)
    if len(region_boxes) > 0 or len(grid_boxes) > 0:
        zero_box = torch.zeros_like(region_boxes[-1][-1]).unsqueeze(0) # (1, 4)
    else:
        zero_box = None
    for batch_ith in range(len(samples)):
        if len(region_features) > 0:
            for _ in range(region_features[batch_ith].shape[0], max_seq_len):
                region_features[batch_ith] = torch.cat([region_features[batch_ith], zero_region_feature], dim=0)
                if zero_box is not None and len(region_boxes) > 0:
                    region_boxes[batch_ith] = torch.cat([region_boxes[batch_ith], zero_box], dim=0)
        if len(grid_features) > 0:
            for _ in range(grid_features[batch_ith].shape[0], max_seq_len):
                grid_features[batch_ith] = torch.cat([grid_features[batch_ith], zero_grid_feature], dim=0)
                if zero_box is not None and len(grid_boxes) > 0:
                    grid_boxes[batch_ith] = torch.cat([grid_boxes[batch_ith], zero_box], dim=0)

    if len(region_features) > 0:
        region_features = torch.cat([feature.unsqueeze_(0) for feature in region_features], dim=0)
    else:
        region_features = None

    if len(grid_features) > 0:
        grid_features = torch.cat([feature.unsqueeze_(0) for feature in grid_features], dim=0)
    else:
        grid_features = None

    if len(image_ids) == 0:
        image_ids = None
    
    if len(region_boxes) > 0:
        region_boxes = torch.cat([box.unsqueeze_(0) for box in region_boxes], dim=0)
    else:
        region_boxes = None

    if len(grid_boxes) > 0:
        grid_boxes = torch.cat([box.unsqueeze_(0) for box in grid_boxes], dim=0)
    else:
        grid_boxes = None

    if len(questions) > 0:
        questions = torch.cat([question.unsqueeze_(0) for question in questions], dim=0)
    else:
        questions = None

    if len(answers) > 0:
        answers = torch.cat([answer.unsqueeze_(0) for answer in answers], dim=0)
    else:
        answers = None

    return {
        "image_ids": image_ids,
        "region_features": region_features, 
        "grid_features": grid_features,
        "region_boxes": region_boxes,
        "grid_boxes": grid_boxes,
        "questions": questions,
        "answers": answers
    }

def generatation_collate_fn(samples):
    image_ids = []
    filenames = []
    region_features = []
    grid_features = []
    region_boxes = []
    grid_boxes = []
    questions = []
    question_tokens = []
    answers = []
    answer_tokens = []
    shifted_right_answer_tokens = []
    max_seq_len = 0
    for sample in samples:
        image_id = sample["image_id"]
        filename = sample["filename"]
        region_feature = sample["region_features"]
        grid_feature = sample["grid_features"]
        region_box = sample["region_boxes"]
        grid_box = sample["grid_boxes"]
        question = sample["question"]
        question_token = sample["question_tokens"]
        answer = sample["answer"]
        answer_token = sample["answer_tokens"]
        shifted_right_answer_token = sample["shifted_right_answer_tokens"]

        if region_feature is not None and max_seq_len < region_feature.shape[0]:
            max_seq_len = region_feature.shape[0]

        if grid_feature is not None and max_seq_len < grid_feature.shape[0]:
            max_seq_len = grid_feature.shape[0]

        assert max_seq_len != 0, "both region-based features and grid-based features are not presented"

        if image_id is not None:
            image_ids.append(image_id)
        if filename is not None:
            filenames.append(filename)
        if region_box is not None:
            region_boxes.append(torch.tensor(region_box))
        if grid_box is not None:
            grid_boxes.append(torch.tensor(grid_box))
        if question is not None:
            questions.append(question)
        if answer is not None:
            answers.append(answer)
        if question_token is not None:
            question_tokens.append(question_token)
        if answer_token is not None:
            answer_tokens.append(answer_token)
        if shifted_right_answer_token is not None:
            shifted_right_answer_tokens.append(shifted_right_answer_token)

        if region_feature is not None:
            region_features.append(torch.tensor(region_feature))
        
        if grid_feature is not None:
            grid_features.append(torch.tensor(grid_feature))

    if len(region_features) > 0:
        zero_region_feature = torch.zeros_like(region_features[-1][-1]).unsqueeze(0) # padding tensor for region features (1, dim)
    if len(grid_features) > 0:
        zero_grid_feature = torch.zeros_like(grid_features[-1][-1]).unsqueeze(0) # padding tensor for grid features (1, dim)
    if len(region_boxes) > 0 or len(grid_boxes) > 0:
        zero_box = torch.zeros_like(region_boxes[-1][-1]).unsqueeze(0) # (1, 4)
    else:
        zero_box = None
    for batch_ith in range(len(samples)):
        if len(region_features) > 0:
            for ith in range(region_features[batch_ith].shape[0], max_seq_len):
                region_features[batch_ith] = torch.cat([region_features[batch_ith], zero_region_feature], dim=0)
                if zero_box is not None and len(region_boxes) > 0:
                    region_boxes[batch_ith] = torch.cat([region_boxes[batch_ith], zero_box], dim=0)
        if len(grid_features) > 0:
            for ith in range(grid_features[batch_ith].shape[0], max_seq_len):
                grid_features[batch_ith] = torch.cat([grid_features[batch_ith], zero_grid_feature], dim=0)
                if zero_box is not None and len(grid_boxes) > 0:
                    grid_boxes[batch_ith] = torch.cat([grid_boxes[batch_ith], zero_box], dim=0)

    if len(region_features) > 0:
        region_features = torch.cat([feature.unsqueeze_(0) for feature in region_features], dim=0)
    else:
        region_features = None

    if len(grid_features) > 0:
        grid_features = torch.cat([feature.unsqueeze_(0) for feature in grid_features], dim=0)
    else:
        grid_features = None

    if len(image_ids) == 0:
        image_ids = None
    
    if len(filenames) == 0:
        filenames = None
    
    if len(region_boxes) > 0:
        region_boxes = torch.cat([box.unsqueeze_(0) for box in region_boxes], dim=0)
    else:
        region_boxes = None

    if len(grid_boxes) > 0:
        grid_boxes = torch.cat([box.unsqueeze_(0) for box in grid_boxes], dim=0)
    else:
        grid_boxes = None
    
    if len(questions) == 0:
        questions = None

    if len(answers) == 0:
        answers = None

    if len(question_tokens) > 0:
        question_tokens = torch.cat([token.unsqueeze_(0) for token in question_tokens], dim=0)
    else:
        question_tokens = None

    if len(answer_tokens) > 0:
        answer_tokens = torch.cat([token.unsqueeze_(0) for token in answer_tokens], dim=0)
    else:
        answer_tokens = None
    
    if len(shifted_right_answer_tokens) > 0:
        shifted_right_answer_tokens = torch.cat([token.unsqueeze_(0) for token in shifted_right_answer_tokens], dim=0)
    else:
        shifted_right_answer_tokens = None

    return {
        "image_ids": image_ids,
        "filenames": filenames,
        "region_features": region_features, 
        "grid_features": grid_features,
        "region_boxes": region_boxes,
        "grid_boxes": grid_boxes,
        "questions": questions,
        "answers": answers,
        "question_tokens": question_tokens, 
        "answer_tokens": answer_tokens, 
        "shifted_right_answer_tokens": shifted_right_answer_tokens
    }