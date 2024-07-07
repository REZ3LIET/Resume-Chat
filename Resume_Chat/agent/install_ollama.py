import subprocess

def install_ollama():
    status = subprocess.run(["bash", "./Resume_Chat/tools/ollama_preq.sh"])
    return True