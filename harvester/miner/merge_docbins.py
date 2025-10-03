import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")


def merge_docbins(paths, out_path):
    out = DocBin(store_user_data=False)
    for p in paths:
        db = DocBin().from_disk(p)
        for doc in db.get_docs(nlp.vocab):
            out.add(doc)
    out.to_disk(out_path)


merge_docbins(
    [
        "miner/ceder_1/train.spacy",
        "miner/ceder_2/train.spacy",
        "miner/ceder_3/train.spacy",
    ],
    "miner/ceder_all/train.spacy",
)
merge_docbins(
    ["miner/ceder_1/dev.spacy", "miner/ceder_2/dev.spacy", "miner/ceder_3/dev.spacy"],
    "miner/ceder_all/dev.spacy",
)
