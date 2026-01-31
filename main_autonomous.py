"""
Autonomous Email Assistant - Main Entry Point
True agentic system with LLM-driven tool selection
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.workflows.autonomous_agent import AutonomousEmailAgent
from src.services import EmailIndexerService
from src.utils import setup_logger, get_logger
from src.config import get_settings

# Setup logger
setup_logger()
logger = get_logger()


async def interactive_mode():
    """Run autonomous agent in interactive mode"""
    print("=" * 80)
    print("🤖 Autonomous Email Assistant (LLM-Driven)")
    print("=" * 80)
    print("\nInitializing autonomous agent...")
    
    # Initialize agent
    agent = AutonomousEmailAgent()
    indexer = EmailIndexerService()
    settings = get_settings()
    
    print("✓ Autonomous agent initialized successfully!")
    print("\n📋 Commands:")
    print("  - Type your query naturally - LLM will decide which tools to use")
    print("  - 'autoindex' - Index emails from last 7 days")
    print("  - 'status' - Show indexer status")
    print("  - 'stats' - Show memory statistics")
    print("  - 'cleardb' - Clear vector database")
    print("  - 'clear' - Clear conversation history")
    print("  - 'quit' - Exit")
    print("\n💡 Agent Features:")
    print("  - LLM decides which tools to use autonomously")
    print("  - Can call multiple tools for one query")
    print("  - Maintains conversation memory")
    print("  - Learns from interactions")
    print("\n" + "=" * 80)
    
    while True:
        try:
            print("\n")
            query = input("You: ").strip()
            
            if not query:
                continue
            
            # Handle special commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif query.lower() == 'autoindex':
                print("\n🔄 Running automatic indexing (last 7 days)...")
                stats = await indexer.run_indexing()
                print(f"\n✓ Indexing completed!")
                print(f"  - Duration: {stats['duration_seconds']:.2f}s")
                print(f"  - Fetched: {stats['total_fetched']}")
                print(f"  - Newly indexed: {stats['indexed']}")
                print(f"  - Skipped: {stats['skipped']}")
                print(f"  - Errors: {stats['errors']}")
                continue
            
            elif query.lower() == 'status':
                print("\n📊 Indexer Status:")
                status = indexer.get_status()
                print(f"  - Running: {status['is_running']}")
                print(f"  - Last run: {status['last_index_time'] or 'Never'}")
                print(f"  - Next run: {status['next_index_time'] or 'N/A'}")
                continue
            
            elif query.lower() == 'stats':
                print("\n📊 Memory Statistics:")
                stats = agent.get_memory_stats()
                print(f"  - Vector store emails: {stats['vector_store']['total_documents']}")
                print(f"  - Conversation messages: {stats['conversation_messages']}")
                print(f"  - User patterns: {stats['user_patterns']}")
                continue
            
            elif query.lower() == 'cleardb':
                print("\n⚠️  WARNING: Clear all indexed emails and email storage?")
                confirm = input("Type 'yes' to confirm: ").strip().lower()
                if confirm == 'yes':
                    agent.vector_store.clear_all_data()
                    agent.clear_email_storage()
                    print("✓ Vector database and email storage cleared!")
                continue
            
            elif query.lower() == 'clear':
                agent.clear_conversation()
                print("\n✓ Conversation cleared!")
                continue
            
            # Process query with autonomous agent
            print("\n🤔 Agent is thinking...")
            response = await agent.process_query(query)
            
            # Display response
            print(f"\n🤖 Agent: {response['answer']}")
            
            # Show tool calls if any
            if response.get('tool_calls'):
                print(f"\n🔧 Tools Used ({len(response['tool_calls'])}):")
                for i, tool_call in enumerate(response['tool_calls'], 1):
                    print(f"  {i}. {tool_call['tool']} → {str(tool_call['result'])[:100]}...")
            
            print(f"\n💭 Iterations: {response.get('iterations', 0)}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {e}")


async def single_query_mode(query: str):
    """Run agent with single query"""
    print(f"\n🤖 Processing: {query}\n")
    
    agent = AutonomousEmailAgent()
    response = await agent.process_query(query)
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(response['answer'])
    
    if response.get('tool_calls'):
        print("\n" + "=" * 80)
        print(f"TOOLS USED ({len(response['tool_calls'])}):")
        print("=" * 80)
        for i, tool_call in enumerate(response['tool_calls'], 1):
            print(f"\n{i}. {tool_call['tool']}")
            print(f"   Parameters: {tool_call['parameters']}")
            print(f"   Result: {str(tool_call['result'])[:200]}...")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

