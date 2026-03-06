import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARQUIVO_POLITICAS = os.path.join(DATA_DIR, "politicas_empresa.txt")

# Configurações da IA
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
