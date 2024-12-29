# LLM Multimodal

This package implements zero-shot image captioning using Hugging Face models (BLIP, Llava, Qwen-VL, Paligemma).

## Installation

Install the required libraries:
bash
pip install -r requirements.txt

## Usage

1. Replace `URL_TO_YOUR_IMAGE` with the URL of the image you want to caption.
2. Replace `YOUR_VIETNAMESE_CAPTION` with your reference Vietnamese caption.
3. Change `model_name` to the desired model (BLIP, Llava, Qwen-VL, or Paligemma).
4. Run the script:
bash
python image_captioning.py
```

### Explanation
- The `image_captioning.py` file contains functions to load the model, generate captions, and evaluate them using BLEU, ROUGE, METEOR, and CIDEr metrics.
- The model is quantized to 4 bits for efficiency.
- The script can be run directly, and it will print the generated caption and evaluation scores.

Make sure to replace the placeholders in the code with your actual image URL and Vietnamese caption for evaluation. This structure keeps the implementation concise and focused on the essential functionality.