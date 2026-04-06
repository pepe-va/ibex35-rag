"""Pydantic v2 request/response models."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, description="Financial question")
    company_filter: str | None = Field(
        default=None,
        description="Filter results to a specific IBEX35 company (e.g. 'IBERDROLA')",
    )
    use_agent: bool = Field(
        default=False,
        description="Use the ReAct agent (combines RAG + real-time data). Slower but more capable.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Session ID for agent conversation memory. Same thread_id resumes the conversation.",
    )


class SourceDoc(BaseModel):
    company: str
    ticker: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc] = Field(default_factory=list)
    latency_seconds: float
    from_cache: bool = False
    query: str


class AgentResponse(BaseModel):
    answer: str
    latency_seconds: float
    steps_taken: int
    tools_used: list[str]
    query: str


class AskResponse(BaseModel):
    answer: str
    route: str  # "rag" | "agent"
    latency_seconds: float
    query: str
    # RAG-specific (optional)
    sources: list[SourceDoc] = Field(default_factory=list)
    from_cache: bool = False
    # Agent-specific (optional)
    tools_used: list[str] = Field(default_factory=list)
    steps_taken: int = 0


class IngestionRequest(BaseModel):
    pdf_dir: str | None = Field(
        default=None,
        description="Override default PDF directory path",
    )


class IngestionResponse(BaseModel):
    success: bool
    total_documents: int
    total_nodes: int
    total_companies: int
    duration_seconds: float
    companies: list[str]
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    redis: str
    ollama: str
    collection_count: int
