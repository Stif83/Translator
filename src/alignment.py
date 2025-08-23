from nltk.translate import AlignedSent

def aligned_sentences_fr_to_en(data_fr, data_en):

    assert len(data_fr) == len(data_en), "Tailles des listes différentes !"
    aligned_sentences = []

    for fr_tokens, en_tokens in zip(data_fr, data_en):
        aligned_sentences.append(AlignedSent(fr_tokens,en_tokens))

    return aligned_sentences

def aligned_sentences_en_to_fr(data_fr, data_en):

    assert len(data_fr) == len(data_en), "Tailles des listes différentes !"
    aligned_sentences = []

    for fr_tokens, en_tokens in zip(data_fr, data_en):
        aligned_sentences.append(AlignedSent(en_tokens, fr_tokens))

    return aligned_sentences


