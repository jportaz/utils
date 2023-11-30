python=python3.8

all:
	$(python) bin/brat2dataset.py \
		--ann examples/CIE_riesgos.ann \
		--txt examples/CIE_riesgos.txt \
	> /tmp/CIE_riesgos.json
	#@
	$(python) bin/tokenizer.py \
		--json /tmp/CIE_riesgos.json \
	> examples/CIE_riesgos.json

