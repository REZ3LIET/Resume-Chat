#! /bin/bash

curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama2

ollama pull nomic-embed-text