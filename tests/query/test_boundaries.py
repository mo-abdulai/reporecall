from pathlib import Path


def test_query_understanding_does_not_import_retrieval_or_rag_layers():
    query_package = Path("src/reporecall/query")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(query_package.glob("*.py"))
    )

    forbidden_references = (
        "FaissVectorIndex",
        "VectorRetriever",
        "BM25Index",
        "KeywordRetriever",
        "HybridRetriever",
        "ReciprocalRankFusion",
        "CrossEncoderReranker",
        "RAGPipeline",
        "RAGContextBuilder",
    )

    for reference in forbidden_references:
        assert reference not in source
