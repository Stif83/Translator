from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from tqdm import tqdm


def preprocess_data():
    with open("data/raw/en-fr small.tmx", "r", encoding="utf-8") as corpus:
        dataset = BeautifulSoup(corpus, "xml")

    data_fr = []
    data_en = []
    skipped = 0

    for tu in tqdm(dataset.find_all("tu"), desc="Lecture des segements"):
        fr = tu.find("tuv", {"xml:lang": "fr"})
        en = tu.find("tuv", {"xml:lang": "en"})

        if not (fr and en):
            skipped += 1
            continue

        try:
            fr_tokens = word_tokenize(fr.seg.text.strip())
            en_tokens = word_tokenize(en.seg.text.strip())

            data_fr.append(fr_tokens)
            data_en.append(en_tokens)

        except Exception as e:
            skipped += 1
            continue

    print(f"Segments traités : {len(data_fr)}")
    print(f"Segments ignorés : {skipped}")

    return data_fr, data_en

"""
data_fr, data_en = preprocess_data()

# Tests de cohérence à faire :
print(f"Nombre de phrases FR : {len(data_fr)}")
print(f"Nombre de phrases EN : {len(data_en)}")
print(f"Tailles égales ? {len(data_fr) == len(data_en)}")

# Exemples à examiner :
for i in range(min(3, len(data_fr))):
    print(f"FR[{i}]: {data_fr[i]}")
    print(f"EN[{i}]: {data_en[i]}")
    print("-" * 40)
    
    """