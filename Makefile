python=python3

TARGETS= \
	examples/CIE_modelo_negocio.json \
	examples/IBERDROLA_modelo_negocio.json \
	examples/MAPFRE_modelo_negocio.json \
	examples/CIE_riesgos.json \
	examples/IBERDROLA_riesgos.json \
	examples/MAPFRE_riesgos.json \
	examples/CIE_RSC_medioambiente.json \
	examples/IBERDROLA_RSC.json \
	examples/MAPFRE_RSC_medioambiente.json

%.json:
	$(python) bin/brat2jsonl.py --ann $*.ann --txt $*.txt > $*.json

datasets: $(TARGETS)
	cat $(TARGETS) | shuf | split -l 400
	mv xaa examples/train.json
	mv xab examples/eval.json

train:
	$(python) bin/train_t2tc.py \
		--train_dataset examples/train.json \
		--eval_dataset examples/eval.json \
		--num_train_epochs 100
