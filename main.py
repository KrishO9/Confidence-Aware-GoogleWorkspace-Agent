"""
Autonomous Email Assistant - Main Entry Point
A sophisticated agentic system using LangGraph, Azure OpenAI, and Gmail APIs
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.workflows import EmailAssistant
from src.utils import setup_logger, get_logger
from src.config import get_settings

# Setup logger
setup_logger()
logger = get_logger()


async def interactive_mode():
    """Run assistant in interactive mode"""
    print("=" * 80)
    print("🤖 Autonomous Email Assistant")
    print("=" * 80)
    print("\nInitializing system...")
    
    # Initialize assistant
    assistant = EmailAssistant()
    settings = get_settings()
    
    print("✓ System initialized successfully!")
    print("\n📋 Commands:")
    print("  - Type your query to get assistance")
    print("  - 'index' - Manually index recent emails")
    print("  - 'autoindex' - Run automatic indexing process (last 7 days)")
    print("  - 'status' - Show indexer service status")
    print("  - 'stats' - Show memory statistics")
    print("  - 'cleardb' - Clear vector database (delete all indexed emails)")
    print("  - 'clear' - Clear conversation history")
    print("  - 'quit' - Exit the assistant")
    print("\n" + "=" * 80)
    
    while True:
        try:
            # Get user input
            print("\n")
            query = input("You: ").strip()
            
            if not query:
                continue
            
            # Handle special commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif query.lower() == 'index':
                print("\n📥 Manually indexing recent emails (last 7 days)...")
                stats = await assistant.index_emails(max_emails=100)
                print(f"\n✓ Manual indexing completed!")
                print(f"  - Total fetched: {stats['total_fetched']}")
                print(f"  - Indexed: {stats['indexed']}")
                print(f"  - Skipped: {stats['skipped']}")
                print(f"  - Errors: {stats['errors']}")
                continue
            
            elif query.lower() == 'autoindex':
                print("\n🔄 Running automatic indexing process...")
                stats = await assistant.force_index_now()
                print(f"\n✓ Automatic indexing completed!")
                print(f"  - Duration: {stats['duration_seconds']:.2f} seconds")
                print(f"  - Total fetched: {stats['total_fetched']}")
                print(f"  - Newly indexed: {stats['indexed']}")
                print(f"  - Skipped (already indexed): {stats['skipped']}")
                print(f"  - Errors: {stats['errors']}")
                print(f"  - Date range: Last {stats['days_back']} days")
                continue
            
            elif query.lower() == 'status':
                print("\n📊 Indexer Service Status:")
                status = assistant.get_indexer_status()
                print(f"  - Running: {status['is_running']}")
                print(f"  - Last indexed: {status['last_index_time'] or 'Never'}")
                print(f"  - Next index: {status['next_index_time'] or 'N/A'}")
                print(f"\n⚙️  Configuration:")
                print(f"  - Interval: Every {status['config']['interval_hours']} hours")
                print(f"  - Days back: {status['config']['days_back']} days")
                print(f"  - Max emails per run: {status['config']['max_emails']}")
                continue
            
            elif query.lower() == 'stats':
                print("\n📊 Memory Statistics:")
                stats = assistant.get_memory_stats()
                print(f"  - Vector store documents: {stats['vector_store']['total_documents']}")
                print(f"  - Conversation messages: {stats['conversation_messages']}")
                print(f"  - Has summary: {stats['has_summary']}")
                print(f"  - User patterns tracked: {stats['user_patterns']}")
                continue
            
            elif query.lower() == 'cleardb':
                print("\n⚠️  WARNING: This will delete ALL indexed emails from the vector database!")
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    print("\n🗑️  Clearing vector database...")
                    assistant.clear_vector_db()
                    print("✓ Vector database cleared successfully!")
                    print("💡 Run 'autoindex' to re-index your emails")
                else:
                    print("❌ Operation cancelled")
                continue
            
            elif query.lower() == 'clear':
                assistant.clear_conversation()
                print("\n✓ Conversation history cleared!")
                continue
            
            # Process query
            print("\n🤔 Processing your query...")
            response = await assistant.run(query)
            
            # Display response
            print(f"\n🤖 Assistant: {response['answer']}")
            
            # Show sources if available
            if response.get('sources'):
                print(f"\n📎 Sources ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. {source.get('subject', 'N/A')}")
                    print(f"     From: {source.get('sender', 'N/A')}")
                    print(f"     Date: {source.get('date', 'N/A')}")
            
            # Show metadata
            if response.get('used_rag'):
                print(f"\n💡 Used RAG with semantic search")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {e}")


async def single_query_mode(query: str):
    """Run assistant with single query"""
    print(f"\n🤖 Processing query: {query}\n")
    
    # Initialize assistant
    assistant = EmailAssistant()
    
    # Process query
    response = await assistant.run(query)
    
    # Display response
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(response['answer'])
    
    if response.get('sources'):
        print("\n" + "=" * 80)
        print("SOURCES:")
        print("=" * 80)
        for i, source in enumerate(response['sources'], 1):
            print(f"\n{i}. {source.get('subject', 'N/A')}")
            print(f"   From: {source.get('sender', 'N/A')}")
            print(f"   Date: {source.get('date', 'N/A')}")
            print(f"   Relevance: {source.get('relevance_score', 0):.2%}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

