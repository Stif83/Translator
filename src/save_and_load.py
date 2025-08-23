import dill
import os


def get_model_type(model):
    """Détermine le type de modèle IBM"""
    model_name = model.__class__.__name__
    if hasattr(model, 'model_type'):
        return f"IBMModel{model.model_type}"
    elif 'IBM' in model_name:
        return model_name
    else:
        return "IBMModel1"  # Par défaut


def save_models_hierarchical(model_fr_en, model_en_fr, base_dir='models'):
    """Sauvegarde les modèles avec structure hiérarchique"""

    # Déterminer les types de modèles
    model_type_fr_en = get_model_type(model_fr_en)
    model_type_en_fr = get_model_type(model_en_fr)

    # Définir les chemins
    models_config = [
        {
            'model': model_fr_en,
            'direction': 'fr_to_en',
            'model_type': model_type_fr_en,
            'filename': 'fr_to_en.pkl'
        },
        {
            'model': model_en_fr,
            'direction': 'en_to_fr',
            'model_type': model_type_en_fr,
            'filename': 'en_to_fr.pkl'
        }
    ]

    print("💾 Sauvegarde hiérarchique des modèles...")

    for config in models_config:
        try:
            # Créer le chemin complet : models/direction/TypeModel/
            model_dir = os.path.join(
                base_dir,
                config['direction'],
                config['model_type']
            )

            # Créer les dossiers si nécessaire
            os.makedirs(model_dir, exist_ok=True)

            # Chemin complet du fichier
            filepath = os.path.join(model_dir, config['filename'])

            # Sauvegarder le modèle
            with open(filepath, 'wb') as f:
                dill.dump(config['model'], f)

            print(f"✅ {config['direction']} → {filepath}")

        except Exception as e:
            print(f"❌ Erreur pour {config['direction']} : {e}")


def load_model_hierarchical(direction, model_type, base_dir='models', filename=None):
    """Charge un modèle depuis la structure hiérarchique"""
    if filename is None:
        filename = f"{direction}.pkl"

    filepath = os.path.join(base_dir, direction, model_type, filename)

    try:
        with open(filepath, 'rb') as f:
            model = dill.load(f)
        print(f"✅ Modèle chargé depuis {filepath}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None


def load_all_models_hierarchical(base_dir='models'):
    """Charge tous les modèles disponibles"""
    models = {}

    if not os.path.exists(base_dir):
        print(f"❌ Dossier {base_dir} introuvable")
        return models

    for direction in os.listdir(base_dir):
        direction_path = os.path.join(base_dir, direction)
        if not os.path.isdir(direction_path):
            continue

        models[direction] = {}

        for model_type in os.listdir(direction_path):
            model_type_path = os.path.join(direction_path, model_type)
            if not os.path.isdir(model_type_path):
                continue

            # Chercher le fichier .pkl
            for filename in os.listdir(model_type_path):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(model_type_path, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            model = dill.load(f)
                        models[direction][model_type] = model
                        print(f"✅ Chargé : {direction}/{model_type}")
                        break
                    except Exception as e:
                        print(f"❌ Erreur pour {direction}/{model_type} : {e}")

    return models


def list_saved_models(base_dir='models'):
    """Liste tous les modèles sauvegardés"""
    print(f"\n📁 Modèles sauvegardés dans '{base_dir}':")

    if not os.path.exists(base_dir):
        print("   Aucun dossier de modèles trouvé")
        return

    for direction in sorted(os.listdir(base_dir)):
        direction_path = os.path.join(base_dir, direction)
        if not os.path.isdir(direction_path):
            continue

        print(f"   📂 {direction}/")

        for model_type in sorted(os.listdir(direction_path)):
            model_type_path = os.path.join(direction_path, model_type)
            if not os.path.isdir(model_type_path):
                continue

            print(f"      📂 {model_type}/")

            for filename in sorted(os.listdir(model_type_path)):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(model_type_path, filename)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"         📄 {filename} ({size_mb:.1f} MB)")