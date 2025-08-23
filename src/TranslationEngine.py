from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction


class TranslationEngine:
    def __init__(self, model_fr_en=None, model_en_fr=None):
        self.model_fr_en = model_fr_en
        self.model_en_fr = model_en_fr
        self.smoothing = SmoothingFunction().method1

    def translate_sentence(self, sentence, direction='fr_to_en', method='viterbi'):
        """Traduit une phrase avec différentes méthodes"""

        if direction == 'fr_to_en' and self.model_fr_en is None:
            raise ValueError("Modèle FR→EN non chargé")
        if direction == 'en_to_fr' and self.model_en_fr is None:
            raise ValueError("Modèle EN→FR non chargé")

        model = self.model_fr_en if direction == 'fr_to_en' else self.model_en_fr

        # Tokenisation
        tokens = word_tokenize(sentence.lower())

        if method == 'probabilistic':
            return self._translate_probabilistic(tokens, model)
        elif method == 'word_by_word':
            return self._translate_word_by_word(tokens, model)
        elif method == 'best_alignment':
            return self._translate_best_alignment(tokens, model)
        elif method == "viterbi":
            return self._translate_viterbi_alignment(tokens, model)
        else:
            raise ValueError(f"Méthode '{method}' non supportée")

    def _translate_best_alignment(self, source_tokens, model):
        """Traduction basée sur le meilleur alignement"""
        try:
            # Obtenir l'alignement le plus probable
            alignments = model.align([source_tokens])
            best_alignment = alignments[0] if alignments else []

            # Extraire les traductions basées sur l'alignement
            translation = []
            for i, source_word in enumerate(source_tokens):
                # Chercher la meilleure traduction pour ce mot
                best_target = self._get_best_translation(source_word, model)
                if best_target and best_target != 'NULL':
                    translation.append(best_target)

            return ' '.join(translation) if translation else "Traduction non disponible"

        except Exception as e:
            print(f"Erreur dans best_alignment: {e}")
            return self._translate_word_by_word(source_tokens, model)

    def _translate_viterbi_alignment(self, source_tokens, model):
        """Traduction basée sur l'algorithme de Viterbi (alignement optimal)"""
        try:
            # Créer une phrase bilingue fictive pour l'alignement
            # Le modèle IBM attend des paires de phrases alignées
            dummy_target = [self._get_best_translation(word, model) or word for word in source_tokens]

            # Créer un objet AlignedSent si nécessaire
            from nltk.translate import AlignedSent
            aligned_sent = AlignedSent(source_tokens, dummy_target)

            # Essayer d'utiliser la méthode d'alignement du modèle
            try:
                alignment = model.align([aligned_sent])[0]

                # Utiliser l'alignement pour traduire
                translation = []
                for i, source_word in enumerate(source_tokens):
                    # Chercher l'alignement pour ce mot
                    aligned_indices = [j for j, aligned_i in alignment if aligned_i == i]
                    if aligned_indices:
                        target_word = dummy_target[aligned_indices[0]]
                        if target_word.lower() not in ['null', '<null>', 'none']:
                            translation.append(target_word)
                    else:
                        best_trans = self._get_best_translation(source_word, model)
                        if best_trans and best_trans.lower() not in ['null', '<null>', 'none']:
                            translation.append(best_trans)

                return ' '.join(translation) if translation else self._translate_word_by_word(source_tokens, model)

            except (AttributeError, TypeError, IndexError) as e:
                print(f"Alignement non disponible ({e}), utilisation word-by-word")
                return self._translate_word_by_word(source_tokens, model)

        except Exception as e:
            print(f"Erreur dans Viterbi: {e}")
            return self._translate_word_by_word(source_tokens, model)

    def _translate_word_by_word(self, source_tokens, model):
        """Traduction mot par mot basée sur les probabilités"""
        translation = []

        for word in source_tokens:
            best_translation = self._get_best_translation(word, model)
            if best_translation and best_translation != 'NULL':
                translation.append(best_translation)

        return ' '.join(translation) if translation else "Traduction non disponible"

    def _translate_probabilistic(self, source_tokens, model, top_k=3):
        """Traduction probabiliste avec plusieurs options"""
        translations = []

        for word in source_tokens:
            candidates = self._get_top_translations(word, model, top_k)
            if candidates:
                # Prendre la meilleure candidate
                best_word = max(candidates, key=lambda x: x[1])[0]
                if best_word != 'NULL':
                    translations.append(best_word)

        return ' '.join(translations) if translations else "Traduction non disponible"

    def _get_best_translation(self, source_word, model):
        """Obtient la meilleure traduction pour un mot"""
        if source_word in model.translation_table:
            translations = model.translation_table[source_word]
            if translations:
                return max(translations.items(), key=lambda x: x[1])[0]
        return None

    def _get_top_translations(self, source_word, model, top_k=3):
        """Obtient les top-k traductions pour un mot"""
        if source_word in model.translation_table:
            translations = model.translation_table[source_word]
            if translations:
                sorted_translations = sorted(translations.items(), key=lambda x: x[1], reverse=True)
                return sorted_translations[:top_k]
        return []