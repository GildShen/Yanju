import asyncio
from unittest.mock import patch, MagicMock
from research_agent.config import AppConfig
from research_agent.web import make_config
from research_agent.db import get_connection, initialize_database, fetch_paper
from research_agent.methodology_analysis import analyze_paper_methodology, PROMPT_VERSION, OpenAIAnswerClient

MOCK_LLM_RESPONSE = """## Abstract
This is a fake abstract extracted by the LLM. It shows that the parsing logic works.

## Research Problem
Fake problem.

## Theory And Context
Fake theory.

## Data And Sample
Fake data.

## Research Design
Fake design.

## Measures And Constructs
Fake measures.

## Analysis Method
Fake analysis.

## Main Findings
Fake findings.

## Limitations
Fake limitations.

## Notes For Future Research
Fake notes.
"""

def test_abstract_extraction():
    config = make_config()
    db_path = config.db_path
    
    # 1. Create a dummy paper
    entry_id = "test_abstract_update_123"
    with get_connection(config) as conn:
        initialize_database(conn)
        conn.execute("INSERT OR REPLACE INTO papers (entry_id, title, published, source, feed_url, note_path, added_at, doi, pdf_url, abstract) VALUES (?, ?, '2025-01-01', 'test', 'test', 'test', '2025-01-01', '', '', '')", (entry_id, "Test Paper"))
        # ensure pdf_catalog points to a fake or real file? 
        # Actually, methodology_analysis needs a PDF to analyze. If no PDF is found, it skips the LLM call!
        # Let's mock _resolve_source_pages to return some pages!
    
    with patch("research_agent.methodology_analysis._resolve_source_pages") as mock_resolve:
        mock_resolve.return_value = ([{"page": 1, "text": "Fake paper content here."}], "local_pdf", "dummy.pdf")
        
        with patch.object(OpenAIAnswerClient, "create_answer") as mock_answer:
            class FakeResponse:
                text = MOCK_LLM_RESPONSE
            mock_answer.return_value = FakeResponse()
            
            # 2. Call analyze (force=True to skip cache)
            result = analyze_paper_methodology(config, entry_id, force=True)
            
            # 3. Check if DB was updated
            with get_connection(config) as conn:
                paper = fetch_paper(conn, entry_id)
                abstract = paper.get("abstract")
                print(f"Status: {result.get('status')}")
                print(f"Abstract in DB: {abstract}")
                assert abstract == "This is a fake abstract extracted by the LLM. It shows that the parsing logic works.", "Abstract was not updated correctly!"
                
            print("Test passed successfully!")

if __name__ == "__main__":
    test_abstract_extraction()
