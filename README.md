# Resume-Chat | [Blog](https://medium.com/@rezeliet/enriching-resume-interview-and-modify-your-resume-with-ollama-and-langchain-09aa462adff6)

Resum Chat helps in interacting with your resume in 2 modes:
- Improve Resume:
    In Improve Resume the agent rates your resume and suggests improvement based on the provides job summary.
- Dummy Interview
    In Dummy Interview the agent will ask you questions based on you resume similar to an interview setting.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
1. Installation of LangChain packages and other dependencies
```
pip install -r requirements.txt
```
2. Install Ollama from [Ollama Github](https://github.com/ollama/ollama)

3. Pulling Llama2 for llm
```
ollama pull llama2
```

4. Pulling Embeddings from Ollama
```
ollama pull nomic-embed-text
```

## Usage

To execute this project from the source directory.

```
# Linux
streamlit run Resume_Chat/ui_chat.py
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---
