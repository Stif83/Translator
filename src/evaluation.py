import time

from translator.src.TranslationEngine import TranslationEngine
from translator.src.save_and_load import load_model_hierarchical


def main_evaluation_and_usage():
    """Exemple d'utilisation complète"""
    print("=== ÉVALUATION ET UTILISATION DES MODÈLES ===")

    # 1. Charger les modèles
    print("\n1️⃣ Chargement des modèles...")
    model_fr_en = load_model_hierarchical('fr_to_en', 'IBMModel2')
    model_en_fr = load_model_hierarchical('en_to_fr', 'IBMModel2')

    if not model_fr_en or not model_en_fr:
        print("❌ Impossible de charger les modèles")
        return

    # 2. Créer le traducteur
    translator = TranslationEngine(model_fr_en, model_en_fr)

    print("\n2️⃣ Analyse de qualité des modèles:")
    try:
        fr_en_quality = translator.analyze_translation_quality(model_fr_en, 'FR→EN', sample_size=50)
        en_fr_quality = translator.analyze_translation_quality(model_en_fr, 'EN→FR', sample_size=50)

        print(f"✅ Qualité FR→EN: {fr_en_quality['quality_ratio'] * 100:.1f}%")
        print(f"✅ Qualité EN→FR: {en_fr_quality['quality_ratio'] * 100:.1f}%")
    except Exception as e:
        print(f"⚠️ Analyse de qualité échouée: {e}")

    # Tests de traduction avec SEULEMENT les méthodes qui marchent
    print("\n3️⃣ Tests de traduction (méthodes fiables):")
    test_phrases = [
        ("bonjour", "fr_to_en"),
        ("hello", "en_to_fr"),
        ("je mange une pomme", "fr_to_en"),
        ("the cat is sleeping", "en_to_fr"),
        ("merci", "fr_to_en"),
        ("thank you", "en_to_fr")
    ]

    # SEULEMENT les méthodes qui fonctionnent
    methods = ['word_by_word', 'probabilistic', 'conservative']

    for phrase, direction in test_phrases:
        print(f"\n--- '{phrase}' ({direction}) ---")
        for method in methods:
            try:
                start_time = time.time()
                translation = translator.translate_sentence(phrase, direction, method)
                end_time = time.time()
                print(f"{method:12}: {translation} ({(end_time - start_time) * 1000:.0f}ms)")
            except Exception as e:
                print(f"{method:12}: ERREUR - {e}")

    # Test interactif simplifié
    print(f"\n4️⃣ Test interactif simplifié:")
    print("Tapez une phrase pour la traduire (ou 'quit' pour quitter):")

    direction = 'fr_to_en'
    method = 'conservative'

    while True:
        try:
            user_input = input(f"\n[{direction}] [{method}] > ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'switch':
                direction = 'en_to_fr' if direction == 'fr_to_en' else 'fr_to_en'
                print(f"Direction changée vers: {direction}")
                continue
            elif user_input.lower().startswith('method:'):
                new_method = user_input.split(':', 1)[1].strip()
                if new_method in methods:
                    method = new_method
                    print(f"Méthode changée vers: {method}")
                else:
                    print(f"Méthodes disponibles: {', '.join(methods)}")
                continue

            if user_input:
                translation = translator.translate_sentence(user_input, direction, method)
                print(f"→ {translation}")

        except KeyboardInterrupt:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")


