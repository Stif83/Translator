from transformers import MarianMTModel, MarianTokenizer

MODEL_FR_EN = "../models/opus-mt-fr-en"
MODEL_EN_FR = "../models/opus-mt-en-fr"

tokenizer_fr_en = MarianTokenizer.from_pretrained(MODEL_FR_EN)
model_fr_en = MarianMTModel.from_pretrained(MODEL_FR_EN)

tokenizer_en_fr = MarianTokenizer.from_pretrained(MODEL_EN_FR)
model_en_fr = MarianMTModel.from_pretrained(MODEL_EN_FR)

def translate_fr_to_en(text: str) -> str:
    """Traduit du français vers l'anglais"""
    inputs = tokenizer_fr_en(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_fr_en.generate(**inputs)
    return tokenizer_fr_en.decode(outputs[0], skip_special_tokens=True)

def translate_en_to_fr(text: str) -> str:
    """Traduit de l'anglais vers le français"""
    inputs = tokenizer_en_fr(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_en_fr.generate(**inputs)
    return tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)


print("FR → EN :", translate_fr_to_en("Bonjour, comment allez-vous ?"))
print("EN → FR :", translate_en_to_fr("Hello, how are you?"))
