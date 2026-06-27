.PHONY: help build publish serve

help:
	@echo "Targets: build publish serve"

build:
	python3 scripts/build.py

publish: build
	python3 scripts/publish_docs.py

serve:
	bash scripts/serve.sh 8765
