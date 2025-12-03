# Agentic Search Engine - Workflow Diagram

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI_ST[Streamlit App<br/>app.py]
        UI_NX[Next.js Client<br/>web-client/]
    end

    subgraph API["API Layer"]
        EP[FastAPI Server<br/>api.py]
        EP_SEARCH[POST /search]
        EP_EMBED[POST /embed]
        EP_INGEST[POST /ingest/*]
        EP_CACHE[GET/POST /cache/*]
    end

    subgraph Pipeline["RAG Pipeline"]
        PIPE[RAGPipeline<br/>rag_pipeline.py]

        subgraph CacheLayer["Cache Layer"]
            CACHE_CHECK{Cache<br/>Lookup}
            CACHE_HIT[Cache HIT<br/>Return cached response]
            CACHE_STORE[Store in Cache]
            SC[SemanticCache<br/>FAISS + JSON]
            RC[RedisSemanticCache<br/>FAISS + Redis]
        end

        subgraph Routing["Query Routing"]
            ROUTER[CompositeRouter]
            RBR[RuleBasedRouter<br/>Keyword matching]
            LLMR[LLMRouter<br/>OpenAI classification]
            RD[RoutingDecision<br/>intent, target_retrievers]
        end

        subgraph Retrieval["Document Retrieval"]
            RET_QDRANT[QdrantRetriever<br/>10-K & OpenAI docs]
            RET_WEB[WebSearchRetriever<br/>DuckDuckGo]
            DOCS[Retrieved Documents]
        end

        subgraph Generation["Response Generation"]
            LLM[OpenAI LLM<br/>gpt-4o-mini]
            PROMPT[RAG Prompt Template]
            RESP[Generated Response]
        end
    end

    subgraph Storage["Storage Layer"]
        QDRANT[(Qdrant<br/>Vector DB)]
        FAISS[(FAISS<br/>In-memory index)]
        REDIS[(Redis<br/>Distributed cache)]
        JSON[(JSON Files<br/>Local cache)]
    end

    subgraph Shared["Shared Components"]
        EMB[SentenceTransformer<br/>nomic-embed-text-v1.5]
    end

    %% Client connections
    UI_ST --> EP
    UI_NX --> EP

    %% API endpoints
    EP --> EP_SEARCH
    EP --> EP_EMBED
    EP --> EP_INGEST
    EP --> EP_CACHE

    %% Main search flow
    EP_SEARCH --> PIPE

    %% Pipeline steps
    PIPE --> CACHE_CHECK
    CACHE_CHECK -->|similarity > 0.78| CACHE_HIT
    CACHE_CHECK -->|no match| ROUTER

    %% Routing flow
    ROUTER --> RBR
    RBR -->|rules match| RD
    RBR -->|no match| LLMR
    LLMR --> RD

    %% Retrieval based on intent
    RD -->|LOCAL_10K| RET_QDRANT
    RD -->|LOCAL_OPENAI| RET_QDRANT
    RD -->|WEB_SEARCH| RET_WEB
    RD -->|HYBRID| RET_QDRANT
    RD -->|HYBRID| RET_WEB

    RET_QDRANT --> DOCS
    RET_WEB --> DOCS

    %% Generation
    DOCS --> PROMPT
    PROMPT --> LLM
    LLM --> RESP

    %% Cache store
    RESP --> CACHE_STORE
    CACHE_STORE --> SC
    CACHE_STORE --> RC

    %% Storage connections
    SC --> FAISS
    SC --> JSON
    RC --> FAISS
    RC --> REDIS
    RET_QDRANT --> QDRANT

    %% Embedding usage
    EMB -.->|encode queries| CACHE_CHECK
    EMB -.->|encode queries| RET_QDRANT
    EMB -.->|encode for storage| CACHE_STORE

    %% Final output
    CACHE_HIT --> OUTPUT[SearchResult<br/>response, sources, timing]
    RESP --> OUTPUT
```

---

## Sequence Diagram (Request Flow)

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant P as RAGPipeline
    participant C as SemanticCache
    participant R as CompositeRouter
    participant Q as QdrantRetriever
    participant W as WebSearchRetriever
    participant L as OpenAI LLM
    participant E as Embeddings

    U->>API: POST /search {query}
    API->>P: search(query)

    rect rgb(200, 230, 200)
        Note over P,C: Step 1: Cache Lookup
        P->>E: encode(query)
        E-->>P: query_embedding
        P->>C: lookup(query_embedding)
        alt Cache HIT
            C-->>P: (cached_response, metadata)
            P-->>API: SearchResult(cache_hit=true)
            API-->>U: Response
        end
    end

    rect rgb(200, 200, 230)
        Note over P,R: Step 2: Query Routing
        P->>R: route(query, retrievers)
        R->>R: RuleBasedRouter.route()
        alt No rule match
            R->>L: classify_query()
            L-->>R: intent
        end
        R-->>P: RoutingDecision
    end

    rect rgb(230, 200, 200)
        Note over P,W: Step 3: Document Retrieval
        par Parallel Retrieval
            P->>Q: retrieve(query, top_k)
            Q->>E: encode(query)
            Q-->>P: documents[]
        and
            P->>W: retrieve(query, top_k)
            W-->>P: web_results[]
        end
    end

    rect rgb(230, 230, 200)
        Note over P,L: Step 4: Response Generation
        P->>P: format_context(documents)
        P->>L: generate(prompt + context)
        L-->>P: response
    end

    rect rgb(200, 230, 230)
        Note over P,C: Step 5: Cache Storage
        P->>E: encode(query)
        P->>C: store(query, response, embedding)
    end

    P-->>API: SearchResult
    API-->>U: Response + Sources + Timing
```

---

## Component Diagram (Interfaces & Implementations)

```mermaid
graph LR
    subgraph Interfaces["Base Interfaces"]
        BC[BaseCache]
        BR[BaseRouter]
        BRT[BaseRetriever]
        BL[BaseLLM]
        BE[BaseEmbedding]
    end

    subgraph Implementations["Implementations"]
        SC[SemanticCache] --> BC
        RSC[RedisSemanticCache] --> BC

        RBR[RuleBasedRouter] --> BR
        LLMR[LLMRouter] --> BR
        CR[CompositeRouter] --> BR

        QR[QdrantRetriever] --> BRT
        WR[WebSearchRetriever] --> BRT

        OL[OpenAILLM] --> BL

        STE[SentenceTransformerEmbedding] --> BE
    end

    subgraph Factory["Factory Pattern"]
        F[create_pipeline<br/>factory.py]
    end

    F --> SC
    F --> RSC
    F --> CR
    F --> QR
    F --> WR
    F --> OL
    F --> STE
```

---

## Data Flow Summary

| Step | Component | Input | Output | Timing |
|------|-----------|-------|--------|--------|
| 1 | Cache Lookup | query | cached_response OR miss | ~5ms |
| 2 | Query Routing | query | RoutingDecision (intent, retrievers) | 10-500ms |
| 3 | Document Retrieval | query, top_k | List[Document] | 100-2000ms |
| 4 | LLM Generation | context, query | response text | 500-3000ms |
| 5 | Cache Storage | query, response | stored entry | ~20ms |
| **Total** | | | SearchResult | **615-5525ms** |

---

## Key Components

### RAGPipeline (`rag_pipeline.py`)
- Orchestrates the complete search flow
- Dependency injection for all components
- Handles timing and error recovery

### SemanticCache (`semantic_cache.py`)
- FAISS index for similarity search
- Cosine similarity threshold: 0.78
- LRU eviction when max_size reached

### CompositeRouter (`router.py`)
- Tries RuleBasedRouter first (fast keyword matching)
- Falls back to LLMRouter (OpenAI classification)
- Returns: LOCAL_10K, LOCAL_OPENAI, WEB_SEARCH, or HYBRID

### Retrievers
- **QdrantRetriever**: Vector similarity search on local documents
- **WebSearchRetriever**: DuckDuckGo search for current information

### OpenAILLM (`openai_llm.py`)
- Model: gpt-4o-mini
- Temperature: 0.3
- RAG prompt template for context-aware generation

---

## Query Intent Types

| Intent | Description | Retrievers Used |
|--------|-------------|-----------------|
| `LOCAL_10K` | Financial 10-K filing questions | qdrant_10k_data |
| `LOCAL_OPENAI` | OpenAI documentation questions | qdrant_opnai_data |
| `WEB_SEARCH` | Current events, news, general web | web_search |
| `HYBRID` | Complex queries needing multiple sources | qdrant + web_search |

---

## Storage Systems

| System | Purpose | Persistence | Use Case |
|--------|---------|-------------|----------|
| **Qdrant** | Document vectors | Disk | 10-K filings, OpenAI docs |
| **FAISS** | Cache query index | Memory | Fast similarity lookup |
| **Redis** | Distributed cache | Disk + Memory | Scalable caching |
| **JSON** | Local cache backup | Disk | Single-instance caching |
