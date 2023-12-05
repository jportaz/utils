# Assumes:
#  (1) continuity (no ";" in the offset field)
#  (2) no label overlapping

# Arreglar: .<SENT>»

import spacy
from spacy.util import compile_prefix_regex
from spacy.util import compile_infix_regex
from spacy.util import compile_suffix_regex
from spacy.pipeline import Sentencizer
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.language import Language

def custom_sentencizer(doc: Doc) -> Doc:
    sentencizer = Sentencizer(punct_chars=["\n"])
    return sentencizer(doc)

def custom_tokenizer(nlp):
    prefixes = list(nlp.Defaults.prefixes)
    prefixes.append('\\-')
    prefixes.append('\u00AD')
    prefix_re = compile_prefix_regex(prefixes)
    
    infixes = list(nlp.Defaults.prefixes)
    infixes.append('-')
    infixes.append('\\/')
    infixes.append('\u00AD')
    infix_re  = compile_infix_regex(infixes)
    
    suffixes = list(nlp.Defaults.suffixes)
    suffixes.append('-')
    suffixes.append('\u00AD')
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab,
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)

@Language.component("line_sent")
def line_senter(doc):
    for tok in doc:
        if tok.text == "\n" or tok.text == "\n\n":
            tok.is_sent_start = True
    return doc

def load(model):
    nlp = spacy.load("es_core_news_sm", exclude=["tok2vec", "morphologizer", "parser", "attribute_ruler", "lemmatizer", "ner"])

    nlp.tokenizer = custom_tokenizer(nlp)
    nlp.add_pipe("line_sent", first=True)
    return nlp

