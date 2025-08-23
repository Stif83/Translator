
from nltk import IBMModel2, IBMModel3

from src.preprocessing import preprocess_data
from src.alignment import aligned_sentences_fr_to_en, aligned_sentences_en_to_fr
from src.training import train_models_with_monitoring
from translator.src.save_and_load import save_models_hierarchical, list_saved_models


def main():
    print("=== TRADUCTEUR NLTK ===")

    print("1️⃣ Preprocessing des données...")
    data_fr, data_en = preprocess_data()
    print(f"   Données chargées : {len(data_fr)} segments")

    print("2️⃣ Création des alignements...")
    aligned_fr_en = aligned_sentences_fr_to_en(data_fr, data_en)
    aligned_en_fr = aligned_sentences_en_to_fr(data_fr, data_en)
    print(f"   Alignements créés : {len(aligned_fr_en)} segments")

    print("3️⃣ Entraînement des modèles...")
    model_fr_en, model_en_fr = train_models_with_monitoring(
        aligned_fr_en,
        aligned_en_fr,
        iterations=10,
        subset_size=None,
        model=IBMModel3
    )

    print("✅ Entraînement terminé !")
    return model_fr_en, model_en_fr


if __name__ == "__main__":
    model_fr_en, model_en_fr = main()
    save_models_hierarchical(model_fr_en, model_en_fr)

    list_saved_models()

    print("\n🔄 Exemples d'utilisation :")
    print("# Charger un modèle spécifique :")
    print("model = load_model_hierarchical('fr_to_en', 'IBMModel1')")
    print("\n# Charger tous les modèles :")
    print("all_models = load_all_models_hierarchical()")
    print("fr_en_model = all_models['fr_to_en']['IBMModel1']")