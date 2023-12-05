import sys
import json

words = []
tags = []

ambig = set(
"""creo
descreo
aburro
paro
consumo
vengo
ve
fundo
siento
revierta
fue
fué
fuera
fuerais
fueran
fueras
fuere
fuereis
fueren
fueres
fueron
fuese
fueseis
fuesen
fueses
fui
fuimos
fuiste
fuisteis
fuistes
fuéramos
fuéremos
fuésemos
vivo
competía
asiento
vendo""".split("\n"))

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("#"):
        if words:
            print(json.dumps({"words": words, "tags": tags}, ensure_ascii=False))
            words = []
            tags = []
    else:
        splits = line.split("\t")
        if splits[3][0] == 'V' and splits[1].lower() in ambig:
            words.append(splits[1])        
            tags.append(splits[4][:7]+"+"+splits[2])
        elif " " not in splits[1]:
            words.append(splits[1])        
            tags.append(splits[4][:7])
        else:
            for i, word in enumerate(splits[1].split()):
                words.append(word)                        
                tags.append(("B-" if i == 0 else "I-") + splits[4][:7])

if words:
    print(json.dumps({"words": words, "tags": tags}, ensure_ascii=False))

