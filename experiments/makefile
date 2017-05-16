NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	@for nb in $(NB); do \
		echo "$$(jq --indent 1 ' \
			.metadata = {} \
			| (.cells[] | select(has("outputs")) | .outputs) = [] \
			| (.cells[] | select(has("execution_count")) | .execution_count) = null \
			| .cells[].metadata = {} \
			' $$nb)" > $$nb; \
	done

.PHONY: run $(NB) clean
