from src.generator import synthesize_answer
from src.schemas import RetrievedChunk


def test_synthesize_answer_returns_source():
    chunks = [
        RetrievedChunk(
            chunk_id=0,
            filename="sample.txt",
            text="Acme Analytics builds analytics tools for small businesses.",
            score=0.91,
        )
    ]

    result = synthesize_answer("What does Acme Analytics build?", chunks)

    assert "analytics tools" in result.answer
    assert result.sources[0].filename == "sample.txt"
