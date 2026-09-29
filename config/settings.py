import os
from dotenv import load_dotenv

load_dotenv()
DOCS_DIR = os.environ.get("DOCUMENTS_DIR", ".docs")
