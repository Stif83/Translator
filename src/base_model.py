from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


MODEL_FR_EN = "../models/opus-mt-fr-en"
MODEL_EN_FR = "../models/opus-mt-en-fr"

logger.info("Chargement des modèles de traduction...")
try:
    tokenizer_fr_en = MarianTokenizer.from_pretrained(MODEL_FR_EN)
    model_fr_en = MarianMTModel.from_pretrained(MODEL_FR_EN)

    tokenizer_en_fr = MarianTokenizer.from_pretrained(MODEL_EN_FR)
    model_en_fr = MarianMTModel.from_pretrained(MODEL_EN_FR)
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles: {e}")
    raise


def translate_fr_to_en(text: str) -> str:
    """Traduit du français vers l'anglais"""
    try:
        inputs = tokenizer_fr_en(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model_fr_en.generate(**inputs, max_length=512)
        return tokenizer_fr_en.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Erreur traduction FR->EN: {e}")
        raise

def translate_en_to_fr(text: str) -> str:
    """Traduit de l'anglais vers le français"""
    try:
        inputs = tokenizer_en_fr(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model_en_fr.generate(**inputs, max_length=512)
        return tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Erreur traduction EN->FR: {e}")
        raise


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()

        if not data or 'text' not in data or 'direction' not in data:
            return jsonify({
                'error': 'Paramètres manquants. Requis: text, direction'
            }), 400

        text = data['text'].strip()
        direction = data['direction']

        if not text:
            return jsonify({'error': 'Texte vide'}), 400

        if direction == 'fr-en':
            translated_text = translate_fr_to_en(text)
        elif direction == 'en-fr':
            translated_text = translate_en_to_fr(text)
        else:
            return jsonify({
                'error': 'Direction invalide. Utilisez "fr-en" ou "en-fr"'
            }), 400

        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'direction': direction
        })

    except Exception as e:
        logger.error(f"Erreur dans l'API: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'API de traduction fonctionnelle'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)