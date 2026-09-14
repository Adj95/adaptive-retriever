"""All Pydantic models for domain data crossing module boundaries."""

from enum import Enum
from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DocClass(str, Enum):
    DIGITAL_PROSE = "digital_prose"
    TABLE_HEAVY = "table_heavy"
    SCANNED = "scanned"
    HTML = "html"
    CODE = "code"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


class DocumentFingerprint(BaseModel):
    """Computed WITHOUT an LLM. Deterministic: same bytes -> same fingerprint."""
    source_path: str
    mime_type: str
    page_count: Optional[int] = None
    char_count: int
    text_to_image_ratio: Optional[float] = None
    has_extractable_text: bool
    table_region_count: int = 0
    heading_depth: int = 0
    avg_line_length: float = 0.0
    column_count_estimate: int = 1
    code_block_ratio: float = 0.0
    speaker_turn_ratio: float = 0.0
    language: str = "en"
    doc_class: DocClass
    class_confidence: float = Field(ge=0.0, le=1.0)
    fingerprint_version: str = "1.0"


class Document(BaseModel):
    doc_id: str
    fingerprint: DocumentFingerprint
    parser_used: str
    parse_ms: float
    parse_error: Optional[str] = None
    ingested_at: str
    raw_text_path: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    chunker_used: str
    section_path: List[str] = Field(default_factory=list)
    contains_table: bool = False
    table_header: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    token_count: int


class RepresentationKind(str, Enum):
    RAW = "raw"
    SUMMARY = "summary"
    QUESTION = "question"
    KEYWORDS = "keywords"


class Representation(BaseModel):
    rep_id: str
    chunk_id: str
    kind: RepresentationKind
    text: str
    generated_by: Optional[str] = None
    tier: Literal["high", "low"]


class QueryClass(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    CONCEPTUAL = "conceptual"
    COMPOUND = "compound"
    MULTIHOP = "multihop"
    AGGREGATION = "aggregation"
    FILTERABLE = "filterable"


class QueryFingerprint(BaseModel):
    raw_query: str
    token_count: int
    has_exact_identifiers: bool
    identifier_spans: List[str] = Field(default_factory=list)
    has_numeric_intent: bool
    has_comparative_language: bool
    multihop_signal_score: float = 0.0
    corpus_vocab_overlap: float = 0.0
    extracted_filters: Dict[str, Any] = Field(default_factory=dict)
    query_class: QueryClass
    class_confidence: float


class QueryProfile(BaseModel):
    raw_query: str
    doc_class: Optional[DocClass] = None
    contains_table: Optional[bool] = None
    entities: List[str] = Field(default_factory=list)
    date_refs: List[str] = Field(default_factory=list)
    access_tags: List[str] = Field(default_factory=list)
    kind: Optional[RepresentationKind] = None
    tier: Optional[Literal["high", "low"]] = None
    extracted_filters: Dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    chunk_id: str
    rep_id: Optional[str] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelevantPassage(BaseModel):
    doc_id: str
    passage_text: str
    passage_norm: str
    canonical_char_start: int
    canonical_char_end: int
    canonical_parser: str


class EvalItem(BaseModel):
    item_id: str
    query: str
    relevant_passages: List[RelevantPassage]
    doc_class_of_source: DocClass
    query_class_expected: Optional[QueryClass] = None
    answer_reference: Optional[str] = None
    provenance: Literal["synthetic", "synthetic_verified", "synthetic_rejected", "human", "correction"]
    split: Literal["dev", "test"]
    verified_at: Optional[str] = None
    verified_by: Optional[str] = None


class StageTiming(BaseModel):
    stage: str
    ms: float
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd_cost: float = 0.0


class EscalationLog(BaseModel):
    rungs_fired: List[str] = Field(default_factory=list)
    confidence_history: List[float] = Field(default_factory=list)
    stop_reason: str
    hops_used: int = 0
    usd_spent: float = 0.0
    ms_spent: float = 0.0


class GenerationRequest(BaseModel):
    request_id: str          # sha256(chunk_id | kind | index | prompt_version)[:16]
    chunk_id: str
    kind: RepresentationKind # SUMMARY | QUESTION | KEYWORDS
    index: int = 0           # question ordinal; 0 for summary/keywords
    prompt: str              # fully rendered, ready to send
    prompt_version: str
    model: str               # the model this request was planned for


class GenerationResult(BaseModel):
    request_id: str
    chunk_id: str
    kind: RepresentationKind
    index: int
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    generated_at: str        # ISO-8601


class Trace(BaseModel):
    trace_id: str
    timestamp: str
    raw_query: str
    query_fingerprint: QueryFingerprint
    config_hash: str
    transformation_applied: Optional[str] = None
    funnel_counts: Dict[str, int] = Field(default_factory=dict)
    escalation_rungs_fired: List[str] = Field(default_factory=list)
    escalation_stop_reason: str
    final_chunk_ids: List[str] = Field(default_factory=list)
    final_confidence: float
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    flagged_for_review: bool = False
    flag_reasons: List[str] = Field(default_factory=list)
    stages: List[StageTiming] = Field(default_factory=list)
    total_ms: float
    total_usd: float

