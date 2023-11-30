# Assumes:
#  (1) continuity (no ";" in the offset field)
#  (2) no label overlapping

# Arreglar: .<SENT>»
# Hacer "\n\n" EOS

import sys
import json
import spacy
from spacy.util import compile_prefix_regex
from spacy.util import compile_infix_regex
from spacy.util import compile_suffix_regex
from spacy.tokenizer import Tokenizer
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--ann", required=True)
parser.add_argument("--txt", required=True)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

debug = args.debug

def custom_tokenizer(nlp):
    #infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
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

def skip(text):
    return text in ['', ' ', ' \n', '\n', '\n\n', '\n \n', '\n\n\n', '•', '\u00AD']

# Load and parse .ann file
file = open(args.ann, "r") 

labels = file.readlines()
labels = [line.split("\t")[1].split(" ") for line in labels]
labels = [[int(start), int(end), label] for label, start, end in labels]
labels.sort(key=lambda x : x[0])

file.close()

nlp = spacy.load("es_core_news_sm")
nlp.tokenizer = custom_tokenizer(nlp)

file = open(args.txt, "r")

text = file.read()

sentence = []

doc = nlp(text)

i = 0
j = 0

while i < len(doc) and j < len(labels):
    if doc[i].idx == labels[j][0] + 1 and text[labels[j][0]] == ' ':
        labels[j][0] += 1
    if len(text) > labels[j][1] and text[labels[j][1]] == ' ':
        labels[j][1] -= 1
    if doc[i].idx == labels[j][0]:
        pref = 'B-'
        while i < len(doc) and doc[i].idx < labels[j][1]:
            if debug:
                print(1, doc[i].is_sent_start, doc[i].text, doc[i].idx, pref + labels[j][2], file=sys.stderr)
            if doc[i].is_sent_start and sentence:
                line = {
                    'text': text[sentence[0][1]:sentence[-1][1]+len(sentence[-1][0])],
                    'words': [w[0] for w in sentence],
                    'tags': [w[2] for w in sentence],
                    'idx': [w[1] for w in sentence]
                }
                assert(len(line['words']) == len(line['tags']))
                assert(len(line['idx']) == len(line['tags']))
                print(json.dumps(line, ensure_ascii=False))
                sentence = []
            if not skip(doc[i].text):
                sentence.append([doc[i].text, doc[i].idx, pref + labels[j][2]])
            i += 1
            pref = 'I-'
        j += 1
    elif doc[i].idx > labels[j][0]:
        if debug:
            print(2, line, file=sys.stderr)
            for tok in doc:
                print(3, tok.text, tok.idx, file=sys.stderr)
            for label in labels:
                print(4, label, '"' + text[label[0]:label[1]] + '"', file=sys.stderr)
        print(f'Skip {doc[i].idx} > {labels[j]} at {id}: "{text[labels[j][0]:labels[j][1]]}"', file=sys.stderr)
        j += 1
    else:
        if debug:
            print(4, doc[i].is_sent_start, doc[i].text, doc[i].idx, 'O', file=sys.stderr)
        if doc[i].is_sent_start and sentence:
            line = {
                'text': text[sentence[0][1]:sentence[-1][1]+len(sentence[-1][0])],
                'words': [w[0] for w in sentence],
                'tags': [w[2] for w in sentence],
                'idx': [w[1] for w in sentence]
            }
            assert(len(line['words']) == len(line['tags']))
            assert(len(line['idx']) == len(line['tags']))
            print(json.dumps(line, ensure_ascii=False))
            sentence = []
        if not skip(doc[i].text):
            sentence.append([doc[i].text, doc[i].idx, 'O'])
        i += 1
        
if j >= len(labels):
    assert j >= len(labels), f'id:{id}, {[(token, token.idx) for token in doc]}, Stopped at {labels[j]}, {labels}'
    while i < len(doc):
        if debug:
            print(5, doc[i].is_sent_start, doc[i].text, doc[i].idx, 'O', file=sys.stderr)
        if doc[i].is_sent_start and sentence:
            line = {
                'text': text[sentence[0][1]:sentence[-1][1]+len(sentence[-1][0])],
                'words': [w[0] for w in sentence],
                'tags': [w[2] for w in sentence],
                'idx': [w[1] for w in sentence]
            }
            assert(len(line['words']) == len(line['tags']))
            assert(len(line['idx']) == len(line['tags']))
            print(json.dumps(line, ensure_ascii=False))
            sentence = []
        if not skip(doc[i].text):
            sentence.append([doc[i].text, doc[i].idx, 'O'])
        i += 1
    if sentence:
        line = {
            'text': text[sentence[0][1]:sentence[-1][1]+len(sentence[-1][0])],
            'words': [w[0] for w in sentence],
            'tags': [w[2] for w in sentence],
            'idx': [w[1] for w in sentence]
        }
        assert(len(line['words']) == len(line['tags']))
        assert(len(line['idx']) == len(line['tags']))
        print(json.dumps(line, ensure_ascii=False))
        if debug:
            print(file=sys.stderr)

file.close()
