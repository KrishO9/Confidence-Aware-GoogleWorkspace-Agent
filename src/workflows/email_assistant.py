"""
Email Assistant - Main LangGraph Workflow
Autonomous agentic system with parallel execution and memory
"""

from typing import Dict, Any, Optional, List, TypedDict, Annotated
import asyncio
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

from src.config import get_settings
from src.utils import get_logger, setup_logger
from src.api import AzureOpenAIClient
from src.memory import MemoryManager
from src.rag import EmailRAG, RAGAgent
from src.tools import EmailTools, CalendarTools, TaskTools, SearchStrategyPlanner
from .execution_planner import ExecutionPlanner
from .decision_engine import DecisionEngine
from src.services import EmailIndexerService

# Setup logger
setup_logger()
logger = get_logger()


class AgentState(TypedDict):
    """State for the email assistant agent"""
    query: str
    messages: Annotated[List[Any], operator.add]
    user_context: str
    search_strategy: Optional[Dict[str, Any]]
    execution_plan: Optional[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]
    final_answer: str
    iteration_count: int
    needs_rag: bool
    needs_search: bool
    completed: bool


class EmailAssistant:
    """
    Autonomous Email Assistant with LangGraph
    Implements TinyAgent principles and LLMCompiler parallel execution
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize components
        self.azure_client = AzureOpenAIClient()
        self.memory_manager = MemoryManager()
        self.email_rag = EmailRAG()
        self.rag_agent = RAGAgent()
        self.search_planner = SearchStrategyPlanner()
        self.execution_planner = ExecutionPlanner()
        self.decision_engine = DecisionEngine()  # Enhanced decision making
        
        # Initialize background services
        self.indexer_service = EmailIndexerService()
        
        # Initialize tools
        self.email_tools = EmailTools()
        self.calendar_tools = CalendarTools()
        self.task_tools = TaskTools()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        # Start background indexing if enabled
        if self.settings.auto_index_enabled:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.indexer_service.start())
                else:
                    loop.run_until_complete(self.indexer_service.start())
            except RuntimeError:
                # No event loop running, will start later
                logger.info("Will start indexer service when event loop is available")
        
        logger.info("Email Assistant initialized successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        # Define workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("plan_search", self._plan_search)
        workflow.add_node("execute_parallel", self._execute_parallel)
        workflow.add_node("retrieve_with_rag", self._retrieve_with_rag)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "understand_query",
            self._route_after_understanding,
            {
                "plan_search": "plan_search",
                "direct_answer": "synthesize_answer"
            }
        )
        
        workflow.add_conditional_edges(
            "plan_search",
            self._route_after_planning,
            {
                "execute": "execute_parallel",
                "rag": "retrieve_with_rag"
            }
        )
        
        workflow.add_edge("execute_parallel", "synthesize_answer")
        workflow.add_edge("retrieve_with_rag", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile()
    
    async def _understand_query(self, state: AgentState) -> AgentState:
        """
        Understand user query and determine approach using enhanced decision engine
        """
        try:
            logger.info(f"Understanding query: {state['query']}")
            
            # Get personalized context from memory
            user_context = await self.memory_manager.get_personalized_context(state['query'])
            state['user_context'] = user_context
            
            # Use enhanced decision engine for comprehensive analysis
            analysis = await self.decision_engine.analyze_query(state['query'])
            
            # Update state with analysis results
            state['needs_rag'] = analysis['needs_rag']
            state['needs_search'] = analysis['needs_search']
            
            # Store analysis details for later use
            if 'analysis' not in state:
                state['analysis'] = analysis
            
            # Add system message
            state['messages'].append(
                SystemMessage(content=f"Processing query: {state['query']}\nAnalysis: {analysis.get('reasoning', '')}")
            )
            
            logger.info(f"Enhanced query analysis completed:")
            logger.info(f"  - needs_rag: {analysis['needs_rag']}")
            logger.info(f"  - needs_search: {analysis['needs_search']}")
            logger.info(f"  - confidence: {analysis['confidence']}")
            # Fix: Handle None time_constraint
            time_constraint = analysis.get('time_constraint')
            time_desc = time_constraint.get('description', 'None') if time_constraint else 'None'
            logger.info(f"  - time_constraint: {time_desc}")
            logger.info(f"  - reasoning: {analysis['reasoning']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error understanding query: {e}")
            # Fallback to safe defaults
            state['needs_rag'] = True
            state['needs_search'] = True
            return state
    
    async def _plan_search(self, state: AgentState) -> AgentState:
        """
        Plan search strategy with parallel execution
        """
        try:
            logger.info("Planning search strategy")
            
            # Generate search strategy
            strategy = await self.search_planner.plan_search_strategy(state['query'])
            state['search_strategy'] = strategy
            
            # Generate execution plan for parallel operations
            available_tools = [
                "search_emails",
                "get_recent_emails",
                "search_calendar",
                "list_tasks"
            ]
            
            execution_tasks = await self.execution_planner.generate_execution_plan(
                user_query=state['query'],
                available_tools=available_tools
            )
            
            state['execution_plan'] = {
                "tasks": execution_tasks,
                "strategy": strategy
            }
            
            logger.info(f"Generated plan with {len(execution_tasks)} tasks")
            
            return state
            
        except Exception as e:
            logger.error(f"Error planning search: {e}")
            return state
    
    async def _execute_parallel(self, state: AgentState) -> AgentState:
        """
        Execute search operations in parallel using LLMCompiler approach
        """
        try:
            logger.info("Executing parallel operations")
            
            if not state.get('execution_plan'):
                return state
            
            tasks = state['execution_plan']['tasks']
            
            # Create tool executor with all available methods
            class ToolExecutor:
                def __init__(self, email_tools, calendar_tools, task_tools):
                    self.email_tools = email_tools
                    self.calendar_tools = calendar_tools
                    self.task_tools = task_tools
                
                def search_emails(self, query: str = "", **kwargs):
                    return self.email_tools.search_emails(query=query, **kwargs)
                
                def get_recent_emails(self, days: int = 7, **kwargs):
                    return self.email_tools.get_recent_emails(days=days, **kwargs)
                
                def search_calendar(self, query: str, **kwargs):
                    return self.calendar_tools.search_events(query=query, **kwargs)
                
                def get_upcoming_events(self, days: int = 7, **kwargs):
                    return self.calendar_tools.get_upcoming_events(days=days, **kwargs)
                
                def list_tasks(self, **kwargs):
                    return self.task_tools.list_tasks(**kwargs)
            
            tool_executor = ToolExecutor(
                self.email_tools,
                self.calendar_tools,
                self.task_tools
            )
            
            # Execute plan
            results = await self.execution_planner.execute_plan(tasks, tool_executor)
            
            # Collect search results
            all_results = []
            for task_id, result in results.get('results', {}).items():
                if result and isinstance(result, list):
                    all_results.extend(result)
            
            state['search_results'] = all_results
            
            logger.info(f"Parallel execution completed: {results.get('completed_tasks', 0)}/{results.get('total_tasks', 0)} tasks")
            
            return state
            
        except Exception as e:
            logger.error(f"Error executing parallel operations: {e}")
            state['search_results'] = []
            return state
    
    async def _retrieve_with_rag(self, state: AgentState) -> AgentState:
        """
        Retrieve information using RAG
        """
        try:
            logger.info("Retrieving with RAG")
            
            # Extract filters from strategy
            filters = {}
            if state.get('search_strategy'):
                strategy = state['search_strategy']
                if strategy.get('sender_filters'):
                    filters['sender'] = strategy['sender_filters'][0]
                if strategy.get('date_filters', {}).get('after'):
                    filters['date_after'] = strategy['date_filters']['after']
            
            # Process with RAG
            rag_response = await self.email_rag.process_query_with_rag(
                query=state['query'],
                filters=filters,
                additional_context=state.get('user_context', '')
            )
            
            state['rag_response'] = rag_response
            
            logger.info(f"RAG retrieval completed: {rag_response.get('total_sources', 0)} sources")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            state['rag_response'] = None
            return state
    
    async def _synthesize_answer(self, state: AgentState) -> AgentState:
        """
        Synthesize final answer from all gathered information
        """
        try:
            logger.info("Synthesizing final answer")
            
            # Gather all available information
            context_parts = []
            
            # Add user context
            if state.get('user_context'):
                context_parts.append(f"User Context:\n{state['user_context']}")
            
            # Add RAG response
            if state.get('rag_response'):
                rag_resp = state['rag_response']
                context_parts.append(f"RAG Answer:\n{rag_resp.get('answer', '')}")
                
                if rag_resp.get('sources'):
                    sources_text = "\n".join([
                        f"- {s.get('subject', 'N/A')} from {s.get('sender', 'N/A')}"
                        for s in rag_resp['sources'][:3]
                    ])
                    context_parts.append(f"Sources:\n{sources_text}")
            
            # Add search results
            if state.get('search_results'):
                results_summary = self._summarize_search_results(state['search_results'])
                context_parts.append(f"Search Results:\n{results_summary}")
            
            # Build synthesis prompt
            context = "\n\n".join(context_parts)
            
            synthesis_prompt = f"""Based on the following information, provide a comprehensive and helpful answer to the user's query.

User Query: {state['query']}

Available Information:
{context}

Provide a clear, concise, and actionable answer:"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an intelligent email assistant. Provide helpful, accurate, and actionable answers."
                },
                {
                    "role": "user",
                    "content": synthesis_prompt
                }
            ]
            
            final_answer = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=1500
            )
            
            state['final_answer'] = final_answer
            state['completed'] = True
            
            # Learn from this interaction
            await self.memory_manager.learn_from_interaction(
                user_query=state['query'],
                assistant_response=final_answer,
                context={
                    "used_rag": state.get('needs_rag', False),
                    "search_results_count": len(state.get('search_results', []))
                }
            )
            
            logger.info("Answer synthesis completed")
            
            return state
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            state['final_answer'] = "I encountered an error while processing your request. Please try again."
            state['completed'] = True
            return state
    
    def _summarize_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize search results"""
        if not results:
            return "No results found."
        
        summary_parts = [f"Found {len(results)} items:"]
        
        for i, result in enumerate(results[:5], 1):
            if 'subject' in result:
                # Email result
                summary_parts.append(
                    f"{i}. Email: {result.get('subject', 'N/A')} from {result.get('from', 'N/A')}"
                )
            elif 'summary' in result:
                # Calendar event
                summary_parts.append(
                    f"{i}. Event: {result.get('summary', 'N/A')}"
                )
            elif 'title' in result:
                # Task
                summary_parts.append(
                    f"{i}. Task: {result.get('title', 'N/A')}"
                )
        
        if len(results) > 5:
            summary_parts.append(f"... and {len(results) - 5} more")
        
        return "\n".join(summary_parts)
    
    def _route_after_understanding(self, state: AgentState) -> str:
        """Route after understanding query"""
        if state.get('needs_search') or state.get('needs_rag'):
            return "plan_search"
        return "direct_answer"
    
    def _route_after_planning(self, state: AgentState) -> str:
        """Route after planning"""
        if state.get('needs_rag'):
            return "rag"
        return "execute"
    
    async def run(
        self,
        query: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Run the email assistant
        
        Args:
            query: User query
            stream: Whether to stream responses (future feature)
            
        Returns:
            Response with answer and metadata
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Initialize state
            initial_state: AgentState = {
                "query": query,
                "messages": [HumanMessage(content=query)],
                "user_context": "",
                "search_strategy": None,
                "execution_plan": None,
                "search_results": [],
                "rag_response": None,
                "final_answer": "",
                "iteration_count": 0,
                "needs_rag": False,
                "needs_search": False,
                "completed": False
            }
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Prepare response
            response = {
                "answer": final_state.get('final_answer', ''),
                "query": query,
                "used_rag": final_state.get('needs_rag', False),
                "search_results_count": len(final_state.get('search_results', [])),
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Add sources if RAG was used
            if final_state.get('rag_response'):
                response['sources'] = final_state['rag_response'].get('sources', [])
            
            logger.info("Query processing completed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error running email assistant: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def index_emails(self, max_emails: int = 100) -> Dict[str, Any]:
        """
        Index recent emails for RAG
        
        Args:
            max_emails: Maximum emails to index
            
        Returns:
            Indexing statistics
        """
        return await self.email_rag.index_emails(max_emails=max_emails)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.memory_manager.get_stats()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory_manager.clear_conversation()
    
    async def start_indexer(self):
        """Start the background indexing service"""
        await self.indexer_service.start()
        logger.info("Background indexing service started")
    
    def get_indexer_status(self) -> Dict[str, Any]:
        """Get indexer service status"""
        return self.indexer_service.get_status()
    
    async def force_index_now(self) -> Dict[str, Any]:
        """Force immediate email indexing"""
        logger.info("Forcing immediate email indexing")
        return await self.indexer_service.run_indexing()
    
    def clear_vector_db(self):
        """Clear all data from vector database"""
        logger.warning("Clearing all data from vector database")
        self.memory_manager.vector_store.clear_all_data()
        logger.info("Vector database cleared successfully")

