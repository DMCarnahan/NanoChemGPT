import importlib.util, spacy
from pathlib import Path

def _import_linker(path):
    p=Path(path); spec=importlib.util.spec_from_file_location('heuristic_linker', p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def load_pipeline(model_dir, linker_path):
    nlp=spacy.load(model_dir); linker=_import_linker(linker_path); return nlp, linker.link_doc

def run_ner_link(nlp, link_doc, texts):
    out=[]
    for t in texts:
        doc=nlp(t)
        out.append({'text':t,'ents':[{'start':e.start_char,'end':e.end_char,'label':e.label_,'text':e.text} for e in doc.ents], 'links': link_doc(doc)})
    return out
