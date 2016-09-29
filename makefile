NB = $(sort $(wildcard *.ipynb))

all: run
	$(MAKE) -C trials run

run:
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $(NB)

install:
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	$(MAKE) -C trials clean
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	#rm -fr data

readme:
	grip README.md

.PHONY: run all install clean readme
