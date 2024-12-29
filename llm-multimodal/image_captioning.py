import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from nltk.translate import bleu_score, meteor_score
from rouge import Rouge
from pycocoevalcap.cider import Cider
import argparse 

def load_model(model_name):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).quantize(4)
    return processor, model

def generate_caption(image_url, model_name):
    processor, model = load_model(model_name)
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def evaluate_caption(generated_caption, reference_caption):
    scores = {
        "BLEU": bleu_score.sentence_bleu([reference_caption.split()], generated_caption.split()),
        "ROUGE": Rouge().get_scores(generated_caption, reference_caption),
        "METEOR": meteor_score.meteor_score([reference_caption], generated_caption),
        "CIDEr": Cider().compute_score({0: [generated_caption]}, {0: [reference_caption]})[0]
    }
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning with BLIP")
    parser.add_argument('--image_url', type=str, required=True, help='URL of the image to caption')
    parser.add_argument('--reference_caption', type=str, required=True, help='Reference caption for evaluation')
    parser.add_argument('--model_name', type=str, default="Salesforce/blip-image-captioning-base", help='Model name to use')  
    
    args = parser.parse_args()  
    
    generated_caption = generate_caption(args.image_url, args.model_name)  
    print("Generated Caption:", generated_caption)
    
    scores = evaluate_caption(generated_caption, args.reference_caption) 
    print("Evaluation Scores:", scores)