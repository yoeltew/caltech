from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://petapp:petapp@db:5432/petapp"
    models_dir: str = "models"

    class Config:
        env_file = ".env"


settings = Settings()
