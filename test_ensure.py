import asyncio
from research_agent.config import AppConfig
from research_agent.web import make_config
from research_agent.pipeline import ingest_feeds

async def main():
    config = make_config()
    try:
        entries = ingest_feeds(config)
        print("Success! Ingested:", len(entries))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
