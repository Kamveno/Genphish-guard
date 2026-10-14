.PHONY: setup run run-lr test clean demo
# genphish-guard — common commands

setup:            ## install pinned dependencies
	pip install -r requirements.txt

run:              ## run the full pipeline (xgboost), regenerate tables + figures
	python -m genphish.run_all

run-lr:           ## run with the logistic-regression baseline
	python -m genphish.run_all --logreg

test:             ## run the reproducibility + invariant test suite
	pytest -q

demo: run         ## alias used in the live review

clean:            ## remove generated results and caches
	rm -rf results/figures/*.png results/*.csv results/summary.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
