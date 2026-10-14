OLLAMA_MODELS=$(HOME)/goinfre/ollama/.models
MODEL=qwen3:0.6b
export OLLAMA_MODELS

.PHONY: all install serve run clean kill

all: run

install: serve
	ollama list | grep -q $(MODEL) || ollama pull $(MODEL)

serve:
	mkdir -p $(OLLAMA_MODELS)
	pgrep -x ollama >/dev/null || \
		gnome-terminal -- bash -c "export OLLAMA_MODELS=$(OLLAMA_MODELS); ollama serve"
	until ollama list >/dev/null 2>&1; do sleep 1; done

run: install
	ollama run $(MODEL)

clean:
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "processed" -exec rm -rf {} +
	find . -type d -name "output" -exec rm -rf {} +

kill:
	-pkill -x ollama
