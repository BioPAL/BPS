.PHONY: help generate build publish serve

help:
	@echo "Targets: generate build publish serve"

generate:
	python3 scripts/generate_slides.py

build:
	python3 scripts/build.py

publish: build
	python3 scripts/publish_docs.py

serve:
	bash scripts/serve.sh 8765
