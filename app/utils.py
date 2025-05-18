from transformers import MarianMTModel, MarianTokenizer

def translate_to_english(text, src_lang='fr'):
    tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-en')
    model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{src_lang}-en')
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

