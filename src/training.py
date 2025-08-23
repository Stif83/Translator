from nltk.translate import IBMModel1, IBMModel2, IBMModel3
import time


def train_models_with_monitoring(aligned_fr_en, aligned_en_fr, iterations=5, subset_size=5000, model=IBMModel1):
    print(f"Début d'entraînement sur {subset_size} segments avec {iterations} itérations...")

    # FR -> EN
    start_time = time.time()
    print("🔄 Entraînement FR->EN en cours...")
    model_fr_en = model(aligned_fr_en[:subset_size], iterations)
    fr_en_time = time.time() - start_time
    print(f"✅ FR->EN terminé en {fr_en_time:.1f}s")

    # EN -> FR
    start_time = time.time()
    print("🔄 Entraînement EN->FR en cours...")
    model_en_fr = model(aligned_en_fr[:subset_size], iterations)
    en_fr_time = time.time() - start_time
    print(f"✅ EN->FR terminé en {en_fr_time:.1f}s")

    print(f"📊 Temps total : {fr_en_time + en_fr_time:.1f}s")

    return model_fr_en, model_en_fr