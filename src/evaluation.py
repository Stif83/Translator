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

    # 3. Test rapide
    print("\n2️⃣ Test rapide...")
    test_phrases = [
        ("bonjour", "fr_to_en"),
        ("hello", "en_to_fr"),
        ("je mange une pomme", "fr_to_en"),
        ("the cat is sleeping", "en_to_fr")
    ]

    for phrase, direction in test_phrases:
        try:
            translation = translator.translate_sentence(phrase, direction)
            print(f"{phrase} → {translation}")
        except Exception as e:
            print(f"Erreur pour '{phrase}': {e}")

    # 4. Évaluation (si vous avez des données de test)
    print("\n3️⃣ Évaluation...")
    print("Pour l'évaluation complète, vous aurez besoin d'un jeu de données de test.")
    print("Exemple de format: [(source1, target1), (source2, target2), ...]")
