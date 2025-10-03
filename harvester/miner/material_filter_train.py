import re

import joblib
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from spacy.tokens import DocBin

CHEMISH = re.compile(
    r"(?:[A-Z][a-z]?\d+)+|oxide|nitrate|acetate|chloride|sulfate|hydroxide|phosphate|carbonate|perovskite|aluminate",
    re.I,
)


def feats(s):
    t = s.strip()
    tl = t.lower()
    return {
        "len": len(t),
        "has_digit": any(c.isdigit() for c in t),
        "caps_ratio": sum(c.isupper() for c in t) / max(1, len(t)),
        "chemish": int(bool(CHEMISH.search(t))),
        "ends_tail": int(
            bool(
                re.search(
                    r"(film|powder|slurry|solution|composite|substrate|support)s?$", tl
                )
            )
        ),
        **{f"c3={tl[i:i+3]}": 1 for i in range(len(tl) - 2)},  # tiny char-grams
    }


# Load gold and predictions on the same dev set
nlp = spacy.load("./ner_model/model-best")
gold = DocBin().from_disk("silver_all/dev.spacy")
X, Y = [], []
for d in gold.get_docs(nlp.vocab):
    gset = {(e.start_char, e.end_char, e.label_) for e in d.ents}
    pred = nlp(d.text)
    for e in pred.ents:
        if e.label_ != "MATERIAL":
            continue
        X.append(feats(e.text))
        Y.append(int((e.start_char, e.end_char, "MATERIAL") in gset))

vec = DictVectorizer()
Xv = vec.fit_transform(X)
clf = LogisticRegression(max_iter=200)
clf.fit(Xv, Y)
joblib.dump({"vec": vec, "clf": clf}, "material_filter.joblib")
print("saved material_filter.joblib")
