NB = $(sort $(wildcard *.ipynb))

run:
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $(NB)

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

.PHONY: run clean
