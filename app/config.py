"""Central configuration, read once from the environment."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/ragdb"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    chat_model: str = "qwen2.5:14b"

    # Retrieval knobs. Every one of these is an experiment you can run
    # against the eval harness, which is the whole point of having them here.
    vector_top_k: int = 30
    keyword_top_k: int = 30
    rrf_k: int = 60

    # Weights for the two arms of the fusion.
    #
    # Textbook RRF gives both arms an equal vote. Measured on this corpus that
    # cost precision: vector alone reaches 100% recall@1, unweighted fusion drops
    # it to 89%. The lexical arm only manages 28% recall@1 on its own, so its
    # mediocre candidates were displacing the correct passage from first place.
    # Sweeping the weight recovers to 94% at 0.2 and to 100% only at 0, which is
    # vector-only.
    #
    # 0.2 is a deliberate compromise rather than the argmax: the lexical arm is
    # kept alive for exact-token queries (SKUs, error codes, CLI flags) that this
    # golden set happens not to stress, while no longer outvoting a retriever
    # that is far stronger here. Treat it as provisional. It is tuned on 18
    # cases, which is small enough to overfit, and it should be re-swept against
    # a larger golden set and any change of embedding model. See eval/sweep.py.
    rrf_vector_weight: float = 1.0
    rrf_keyword_weight: float = 0.2
    rerank_top_n: int = 8
    context_chunks: int = 5


settings = Settings()
