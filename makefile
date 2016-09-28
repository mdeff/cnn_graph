NB = $(sort $(wildcard toolkit/*.ipynb))

run:
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $(NB)

install:
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	#rm -fr data

readme:
	grip README.md

.PHONY: run install clean readme
