TRIALS = $(sort $(wildcard trials/*.ipynb))
EXP    = $(sort $(wildcard *.ipynb))

NBRUN   = jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1
NBCLEAN = jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True

all: exp trials

exp:
	$(NBRUN) $(EXP)

trials:
	$(NBRUN) $(TRIALS)

install:
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	$(NBCLEAN) $(TRIALS)
	$(NBCLEAN) $(EXP)

readme:
	grip README.md

.PHONY: all exp trials install clean readme
