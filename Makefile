SHELL := /bin/zsh

.PHONY: sync-submodules sync-docs push

sync-submodules:
	git submodule update --init --recursive

sync-docs: sync-submodules
	@echo "Syncing Markdown files to envision-copilot-presentation..."
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

push: sync-docs
	git push github main
	git push gitlab main
