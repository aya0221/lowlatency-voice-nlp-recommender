from dotenv import load_dotenv
import os

load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
MODEL_NAME = os.getenv("MODEL_NAME")
SPACY_MODEL = os.getenv("SPACY_MODEL")
