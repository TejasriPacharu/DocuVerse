import os


class Settings:
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    db_name = "llm-chat-db"
    files_dir = os.environ.get("FILES_STORAGE_DIR", "../uploaded_files")


CONFIG = Settings()
