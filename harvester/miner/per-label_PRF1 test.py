import spacy
from spacy.tokens import DocBin
from spacy.training import Example

MODEL_DIR = r"miner/ner_model/model-best"
DEV_PATH = r"miner/ceder_all/dev.spacy"

nlp = spacy.load(MODEL_DIR)
gold_db = DocBin().from_disk(DEV_PATH)
gold_docs = list(gold_db.get_docs(nlp.vocab))

examples = []
for gold in gold_docs:
    pred = nlp(gold.text)
    examples.append(Example(gold, pred))  

scores = nlp.evaluate(examples)
print("ents_p = {:.3f}  ents_r = {:.3f}  ents_f = {:.3f}".format(
    scores["ents_p"], scores["ents_r"], scores["ents_f"]
))
print("\nPer-label:")
for label, m in sorted(scores["ents_per_type"].items()):
    print(f"{label:>10}  P={m['p']:.3f}  R={m['r']:.3f}  F1={m['f']:.3f}")
