
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize
import random


class TranslationEngine:
    def __init__(self, model_fr_en=None, model_en_fr=None):
        self.model_fr_en = model_fr_en
        self.model_en_fr = model_en_fr
        self.smoothing = SmoothingFunction().method1

    def translate_sentence(self, sentence, direction='fr_to_en', method='word_by_word'):
        """Traduit une phrase avec différentes méthodes"""

        if direction == 'fr_to_en' and self.model_fr_en is None:
            raise ValueError("Modèle FR→EN non chargé")
        if direction == 'en_to_fr' and self.model_en_fr is None:
            raise ValueError("Modèle EN→FR non chargé")

        model = self.model_fr_en if direction == 'fr_to_en' else self.model_en_fr

        # Tokenisation
        tokens = word_tokenize(sentence.lower())

        if method == 'word_by_word':
            return self._translate_word_by_word(tokens, model)
        elif method == 'probabilistic':
            return self._translate_probabilistic(tokens, model)
        elif method == 'conservative':
            return self._translate_conservative(tokens, model)
        else:
            raise ValueError(
                f"Méthode '{method}' non supportée. Utilisez: 'word_by_word', 'probabilistic', 'conservative'")

    def _translate_word_by_word(self, source_tokens, model):
        """Traduction mot par mot basée sur les probabilités maximales avec filtrage"""
        translation = []

        for word in source_tokens:
            best_translation = self._get_best_translation_filtered(word, model)
            if best_translation:
                translation.append(best_translation)
            else:
                # Si pas de traduction fiable trouvée, garder le mot original
                translation.append(f"[{word}]")  # Marquer les mots non traduits

        return ' '.join(translation) if translation else "Traduction non disponible"

    def _translate_probabilistic(self, source_tokens, model, top_k=3, threshold=0.01):
        """Traduction probabiliste avec seuil de confiance"""
        translations = []

        for word in source_tokens:
            candidates = self._get_top_translations(word, model, top_k)
            if candidates:
                # Prendre la meilleure candidate si elle dépasse le seuil
                best_word, best_prob = candidates[0]
                if (best_word and
                        best_prob > threshold and
                        best_word.lower() not in ['null', '<null>', 'none']):
                    translations.append(best_word)
                else:
                    # Si la probabilité est trop faible, garder le mot original
                    translations.append(word)
            else:
                translations.append(word)

        return ' '.join(translations) if translations else "Traduction non disponible"

    def _translate_conservative(self, source_tokens, model, min_prob=0.05):
        """Traduction conservative - ne traduit que les mots avec haute confiance"""
        translation = []

        for word in source_tokens:
            best_translation = self._get_best_translation_filtered(word, model, min_prob=min_prob)
            if best_translation:
                translation.append(best_translation)
            else:
                # Garder le mot original si pas assez confiant
                translation.append(word)

        return ' '.join(translation) if translation else "Traduction non disponible"

    def _get_best_translation_filtered(self, source_word, model, min_prob=0.01, max_length_ratio=3.0):
        """Obtient la meilleure traduction avec filtres de qualité et vérifications de sécurité"""
        if not hasattr(model, 'translation_table') or source_word not in model.translation_table:
            return None

        translations = model.translation_table[source_word]
        if not translations:
            return None

        # Filtrer les traductions invalides avec vérifications de sécurité
        valid_translations = {}
        for target_word, prob in translations.items():
            # Vérifications de base
            if not target_word or not isinstance(target_word, str):
                continue

            # Filtrer les mots NULL
            if target_word.lower() in ['null', '<null>', 'none', '', ' ']:
                continue

            # Filtrer les probabilités trop faibles
            if prob < min_prob:
                continue

            # Filtrer les mots suspects (trop longs par rapport au source)
            if len(target_word) > len(source_word) * max_length_ratio:
                continue

            # Filtrer les caractères non-alphabétiques suspects
            if any(char in target_word for char in ['•', '◦', '°', '※', '§']):
                continue

            # Filtrer les mots avec des caractères bizarres
            if any(char.isdigit() for char in target_word) and not any(char.isdigit() for char in source_word):
                continue

            valid_translations[target_word] = prob

        if valid_translations:
            best_translation = max(valid_translations.items(), key=lambda x: x[1])
            # Vérifier que la probabilité est raisonnable
            if best_translation[1] > min_prob:
                return best_translation[0]

        return None

    def _get_top_translations(self, source_word, model, top_k=3):
        """Obtient les top-k traductions pour un mot avec vérifications de sécurité"""
        if not hasattr(model, 'translation_table') or source_word not in model.translation_table:
            return []

        translations = model.translation_table[source_word]
        if not translations:
            return []

        # Filtrer les traductions NULL et invalides
        valid_translations = {}
        for target_word, prob in translations.items():
            if (target_word and
                    isinstance(target_word, str) and
                    target_word.lower() not in ['null', '<null>', 'none', '', ' '] and
                    prob > 0):
                valid_translations[target_word] = prob

        if valid_translations:
            sorted_translations = sorted(valid_translations.items(),
                                         key=lambda x: x[1], reverse=True)
            return sorted_translations[:top_k]
        return []

    def get_translation_info(self, word, model):
        """Obtient des informations détaillées sur les traductions d'un mot"""
        if hasattr(model, 'translation_table') and word in model.translation_table:
            translations = model.translation_table[word]
            total_prob = sum(translations.values())

            print(f"\n📊 Traductions pour '{word}':")
            sorted_trans = sorted(translations.items(), key=lambda x: x[1], reverse=True)

            for target_word, prob in sorted_trans[:10]:  # Top 10
                percentage = (prob / total_prob) * 100 if total_prob > 0 else 0
                print(f"   {target_word}: {prob:.6f} ({percentage:.2f}%)")
        else:
            print(f"❌ Aucune traduction trouvée pour '{word}'")

    def analyze_translation_quality(self, model, direction, sample_size=100):
        """Analyse la qualité des traductions apprises par le modèle"""
        print(f"\n🔍 ANALYSE DE QUALITÉ - {direction}")

        if not hasattr(model, 'translation_table'):
            print("❌ Pas de table de traduction trouvée")
            return

        translation_table = model.translation_table
        print(f"📊 Vocabulaire source: {len(translation_table)} mots")

        # Statistiques générales
        total_translations = sum(len(translations) for translations in translation_table.values())
        avg_translations_per_word = total_translations / len(translation_table)
        print(f"📊 Traductions totales: {total_translations}")
        print(f"📊 Moyenne par mot: {avg_translations_per_word:.1f}")

        # Analyser un échantillon de traductions
        sample_words = random.sample(list(translation_table.keys()), min(sample_size, len(translation_table)))

        suspicious_translations = []
        good_translations = []

        for source_word in sample_words:
            translations = translation_table[source_word]
            if not translations:
                continue

            # Meilleure traduction
            best_target, best_prob = max(translations.items(), key=lambda x: x[1])

            # Critères de qualité
            is_suspicious = (
                    len(best_target) > len(source_word) * 3 or  # Trop long
                    any(char in best_target for char in ['•', '◦', '°', '※', '§']) or  # Caractères suspects
                    any(char.isdigit() for char in best_target) and not any(
                char.isdigit() for char in source_word) or  # Chiffres inattendus
                    best_prob < 0.001  # Probabilité très faible
            )

            if is_suspicious:
                suspicious_translations.append((source_word, best_target, best_prob))
            else:
                good_translations.append((source_word, best_target, best_prob))

        print(
            f"\n✅ Traductions correctes: {len(good_translations)} ({len(good_translations) / len(sample_words) * 100:.1f}%)")
        print(
            f"⚠️  Traductions suspectes: {len(suspicious_translations)} ({len(suspicious_translations) / len(sample_words) * 100:.1f}%)")

        # Montrer quelques exemples de bonnes traductions
        if good_translations:
            print(f"\n✅ Exemples de bonnes traductions:")
            for source, target, prob in sorted(good_translations, key=lambda x: x[2], reverse=True)[:10]:
                print(f"   {source} → {target} (prob: {prob:.4f})")

        # Montrer les traductions suspectes
        if suspicious_translations:
            print(f"\n⚠️  Traductions suspectes à vérifier:")
            for source, target, prob in sorted(suspicious_translations, key=lambda x: x[2], reverse=True)[:10]:
                print(f"   {source} → {target} (prob: {prob:.4f})")

        return {
            'total_words': len(translation_table),
            'good_translations': len(good_translations),
            'suspicious_translations': len(suspicious_translations),
            'quality_ratio': len(good_translations) / len(sample_words) if sample_words else 0
        }