NB = $(sort $(wildcard *.ipynb))
DIRS = nips2016 trials

CLEANDIRS = $(DIRS:%=clean-%)

run: $(NB) $(DIRS)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

$(DIRS):
	$(MAKE) -C $@

clean: $(CLEANDIRS)
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)
	#rm -rf **/*.pyc

$(CLEANDIRS):
	$(MAKE) clean -C $(@:clean-%=%)

install:
	pip install --upgrade pip
	pip install -r requirements.txt

readme:
	grip README.md

.PHONY: run $(NB) $(DIRS) clean $(CLEANDIRS) install readme
