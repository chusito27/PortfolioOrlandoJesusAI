"""Central configuration, read once from the environment."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/ragdb"

    # "ollama" or "bedrock". Ollama is the default so the project runs with no
    # cloud account. Switching to bedrock also means changing embed_dim and
    # re-ingesting, because the chunks table declares a fixed vector width.
    provider: str = "ollama"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    chat_model: str = "qwen2.5:14b"

    # AWS Bedrock. Titan v2 embeddings are 1024-dimensional, so embed_dim has to
    # change with the provider and the corpus has to be re-ingested.
    aws_region: str = "us-east-1"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_chat_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

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

    # "cross-encoder", "llm" or "none". The cross-encoder is the default because
    # it measured the same recall as the LLM re-ranker at a fraction of the
    # latency, and unlike the LLM it cannot return unparseable output. See
    # eval/sweep.py and the README.
    reranker: str = "cross-encoder"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


settings = Settings()
