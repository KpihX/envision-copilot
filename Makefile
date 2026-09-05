# Makefile for Envision PSC Project (INF01)
# Main entry point for building deliverables and syncing

SHELL := /bin/zsh

FLYER_DIR = envision-copilot-presentation/flyer
ASSETS_DIR = envision-copilot-presentation/assets
FLYER_OUT = $(ASSETS_DIR)/flyer.pdf
COMMIT_MSG = "chore: finalize psc deliverables for INF01"
DOCS_COMMIT_MSG = "docs: update documentation and assets"

.PHONY: all flyer push push-docs clean help sync-docs sync-submodules

all: flyer sync-docs clean push

help:
	@echo "Usage:"
	@echo "  make flyer            Compile the flyer PDF and move to $(ASSETS_DIR)/"
	@echo "  make sync-docs        Copy markdown files into envision-copilot-presentation/ for Docsify"
	@echo "  make sync-submodules  Initialize and update submodules recursively"
	@echo "  make push             Commit and push all changes to github and gitlab remotes"
	@echo "  make push-docs        Commit and push only the presentation submodule"
	@echo "  make clean            Remove LaTeX temporary files"
	@echo "  make all              Run flyer, sync-docs, clean, and then push everything"

sync-submodules:
	git submodule update --init --recursive

sync-docs: sync-submodules
	@echo "Syncing Markdown files to envision-copilot-presentation/..."
	mkdir -p envision-copilot-presentation/agents envision-copilot-presentation/pipeline/benchmarks envision-copilot-presentation/pipeline/agent_workflow envision-copilot-presentation/rag/embedders envision-copilot-presentation/rag/chunkers envision-copilot-presentation/rag/parsers envision-copilot-presentation/rag/query_transformers envision-copilot-presentation/rag/retrievers
	cp README.md envision-copilot-presentation/README.md 2>/dev/null || true
	cp agents/AGENTS.md envision-copilot-presentation/agents/AGENTS.md 2>/dev/null || true
	cp pipeline/PIPELINE.md envision-copilot-presentation/pipeline/PIPELINE.md 2>/dev/null || true
	cp pipeline/benchmarks/BENCHMARKS.md envision-copilot-presentation/pipeline/benchmarks/BENCHMARKS.md 2>/dev/null || true
	cp pipeline/agent_workflow/AGENTIC_WORKFLOW.md envision-copilot-presentation/pipeline/agent_workflow/AGENTIC_WORKFLOW.md 2>/dev/null || true
	cp rag/embedders/EMBEDDERS.md envision-copilot-presentation/rag/embedders/EMBEDDERS.md 2>/dev/null || true
	cp rag/chunkers/CHUNKERS.md envision-copilot-presentation/rag/chunkers/CHUNKERS.md 2>/dev/null || true
	cp rag/RAG.md envision-copilot-presentation/rag/RAG.md 2>/dev/null || true
	cp rag/parsers/PARSER.md envision-copilot-presentation/rag/parsers/PARSER.md 2>/dev/null || true
	cp rag/query_transformers/QUERY_TRANSFORMERS.md envision-copilot-presentation/rag/query_transformers/QUERY_TRANSFORMERS.md 2>/dev/null || true
	cp rag/retrievers/RETRIEVERS.md envision-copilot-presentation/rag/retrievers/RETRIEVERS.md 2>/dev/null || true
	@echo "Presentation docs successfully synced!"

flyer:
	@echo "Compiling flyer..."
	cd $(FLYER_DIR) && pdflatex -interaction=nonstopmode flyer.tex
	mv $(FLYER_DIR)/flyer.pdf $(FLYER_OUT)
	@echo "Flyer PDF updated in $(FLYER_OUT)"

push: sync-docs
	@echo "Pushing all changes to github and gitlab remotes..."
	git push github main
	git push gitlab main
	@echo "Successfully pushed all remotes."

push-docs:
	@echo "Pushing documentation changes..."
	cd envision-copilot-presentation && git add . && git commit -m $(DOCS_COMMIT_MSG) 2>/dev/null || true && git push github main && git push gitlab main
	git add envision-copilot-presentation
	git commit -m "chore(submodule): update presentation pointer" 2>/dev/null || true
	git push github main
	git push gitlab main
	@echo "Successfully pushed docs to remotes."

clean:
	@echo "Cleaning up debris in $(FLYER_DIR)..."
	cd $(FLYER_DIR) && rm -f *.aux *.log *.out *.toc *.nav *.snm *.fdb_latexmk *.fls *.synctex.gz
