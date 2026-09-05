SHELL := /bin/zsh

.PHONY: sync-submodules push

sync-submodules:
	git submodule update --init --recursive

push: sync-submodules
	git push github graph
	git push gitlab graph
