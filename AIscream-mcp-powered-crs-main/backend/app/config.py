from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / '.env'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        extra='ignore'
    )
    app_name: str = 'MCP-Powered CRS Backend'
    backend_host: str = '127.0.0.1'
    backend_port: int = 8000
    openai_api_key: str = ''
    openai_model: str = 'gpt-4o-mini'
    cors_origins: str = 'http://127.0.0.1:3000,http://localhost:3000'

    tmdb_api_key: str = ''
    omdb_api_key: str = ''
    google_books_api_key: str = ''
    lastfm_api_key: str = ''

    supabase_url: str = ''
    supabase_anon_key: str = ''
    supabase_service_role_key: str = ''

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(',') if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
