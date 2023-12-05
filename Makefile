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

#MODEL=PlanTL-GOB-ES/roberta-base-bne
MODEL=bert-base-multilingual-uncased
#MODEL=microsoft/deberta-v3-base
EPOCHS=200
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2

#TRAIN_DATASET=examples/train.json
#EVAL_DATASET=examples/eval.json
#TEST_DATASET=examples/eval.json
#METRIC=seqeval

TRAIN_DATASET=/tmp/train.json
EVAL_DATASET=/tmp/valid.json
TEST_DATASET=/tmp/test.json
METRIC=poseval

%.json:
	$(python) bin/brat2jsonl.py --ann $*.ann --txt $*.txt > $*.json

datasets: $(TARGETS)
	cat $(TARGETS) | shuf | split -l 400
	mv xaa examples/train.json
	mv xab examples/eval.json

train:
	$(python) bin/train_t2tc.py \
		--model $(MODEL) \
		--train_dataset $(TRAIN_DATASET) \
		--eval_dataset $(EVAL_DATASET) \
		--test_dataset $(TEST_DATASET) \
		--num_train_epochs $(EPOCHS) \
		--per_device_train_batch_size $(TRAIN_BATCH_SIZE) \
		--per_device_eval_batch_size $(EVAL_BATCH_SIZE) \
		--metric $(METRIC)
