"""
Base Agent with ReAct Pattern
LLM-driven autonomous decision making
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient
import json

logger = get_logger()


class BaseAgent:
    """
    Base autonomous agent using ReAct (Reasoning + Acting) pattern
    LLM decides which tools to use based on the query
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
        self.available_tools = {}
        self.max_iterations = 10
    
    def register_tool(self, name: str, function: callable, description: str, parameters: Dict[str, str]):
        """
        Register a tool that the agent can use
        
        Args:
            name: Tool name
            function: Callable function
            description: What the tool does
            parameters: Parameter descriptions
        """
        self.available_tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }
        logger.debug(f"Registered tool: {name}")
    
    def get_tools_description(self) -> str:
        """Get formatted description of all available tools"""
        tools_desc = []
        for name, tool in self.available_tools.items():
            params = ", ".join([f"{k}: {v}" for k, v in tool['parameters'].items()])
            tools_desc.append(f"- {name}({params}): {tool['description']}")
        return "\n".join(tools_desc)
    
    async def run(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run agent with autonomous tool selection
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            
        Returns:
            Agent response with tool calls and final answer
        """
        try:
            # Initialize
            if conversation_history is None:
                conversation_history = []
            
            iteration = 0
            tool_results = []
            thoughts = []
            
            # Agent loop
            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
                
                # Check for repeated tool calls (prevent loops)
                if iteration > 3:
                    # Check if last 3 tool calls were the same
                    if len(tool_results) >= 3:
                        last_three = [tr['tool'] for tr in tool_results[-3:]]
                        if len(set(last_three)) == 1 and last_three[0] == tool_results[-1]['tool']:
                            logger.warning(f"Agent is looping on tool {last_three[0]}, forcing final answer")
                            # Force agent to synthesize results
                            decision = await self._get_agent_decision(
                                query=query,
                                conversation_history=conversation_history,
                                tool_results=tool_results,
                                thoughts=thoughts,
                                force_synthesize=True
                            )
                            if decision['action'] == 'final_answer':
                                return {
                                    'answer': decision['content'],
                                    'tool_calls': tool_results,
                                    'iterations': iteration,
                                    'thoughts': thoughts,
                                    'completed': True,
                                    'forced_stop': True
                                }
                
                # Get LLM decision
                decision = await self._get_agent_decision(
                    query=query,
                    conversation_history=conversation_history,
                    tool_results=tool_results,
                    thoughts=thoughts
                )
                
                # Check if agent is done
                if decision['action'] == 'final_answer':
                    return {
                        'answer': decision['content'],
                        'tool_calls': tool_results,
                        'iterations': iteration,
                        'thoughts': thoughts,
                        'completed': True
                    }
                
                # Execute tool
                if decision['action'] == 'use_tool':
                    tool_name = decision['tool_name']
                    tool_params = decision['parameters']
                    
                    logger.info(f"Executing tool: {tool_name} with params: {tool_params}")
                    
                    result = await self._execute_tool(tool_name, tool_params)
                    
                    tool_results.append({
                        'tool': tool_name,
                        'parameters': tool_params,
                        'result': result
                    })
                    
                    thoughts.append(decision.get('thought', ''))
                    
                    # Check if we have sufficient results to answer
                    if iteration >= 2 and tool_results:
                        # If we have results from RAG search, we likely have enough
                        rag_results = [tr for tr in tool_results if tr['tool'] == 'search_emails_rag']
                        if rag_results and rag_results[0]['result'].get('count', 0) > 0:
                            # We have RAG results, suggest synthesizing
                            logger.info("RAG results found, agent should synthesize answer")
                
                # Store thought
                elif decision['action'] == 'think':
                    thoughts.append(decision['thought'])
            
            # Max iterations reached
            return {
                'answer': "I've exhausted my thinking iterations. Please rephrase your query.",
                'tool_calls': tool_results,
                'iterations': iteration,
                'completed': False
            }
            
        except Exception as e:
            logger.error(f"Error in agent run: {e}")
            return {
                'answer': f"I encountered an error: {str(e)}",
                'error': str(e),
                'completed': False
            }
    
    async def _get_agent_decision(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        tool_results: List[Dict[str, Any]],
        thoughts: List[str],
        force_synthesize: bool = False
    ) -> Dict[str, Any]:
        """
        Get LLM's decision on what to do next
        
        Returns:
            Decision dict with action and parameters
        """
        tools_description = self.get_tools_description()
        
        # Build context
        context_parts = []
        
        if conversation_history:
            # Include more context from conversation history
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history[-10:]  # Last 10 messages for better context
            ])
            context_parts.append(f"CONVERSATION HISTORY (Use this to understand context and previous questions):\n{history_text}\n\nIMPORTANT: If the user asks about 'previous question', 'last question', 'what did I ask', etc., refer to the conversation history above.")
        
        if tool_results:
            results_text = []
            for tr in tool_results:
                result = tr['result']
                # Format result more clearly
                if isinstance(result, dict):
                    if result.get('complete') and result.get('body'):
                        # Email details result - highlight that it's complete
                        results_text.append(
                            f"Tool: {tr['tool']} (email_id: {result.get('email_id', 'N/A')})\n"
                            f"Status: ✅ COMPLETE EMAIL RETRIEVED\n"
                            f"Body length: {result.get('body_length', len(result.get('body', '')))} characters\n"
                            f"Message: {result.get('message', 'Full email content available')}\n"
                            f"Subject: {result.get('subject', 'N/A')}\n"
                            f"Body preview: {result.get('body', '')[:500]}...\n"
                            f"⚠️ DO NOT call get_email_details again for this email_id - you already have the complete email!"
                        )
                    elif result.get('emails'):
                        # Search results
                        email_count = result.get('count', 0)
                        results_text.append(
                            f"Tool: {tr['tool']}\n"
                            f"Found {email_count} emails matching query.\n"
                            f"Email IDs: {[e.get('email_id') for e in result.get('emails', [])[:5]]}\n"
                            f"To get full content, use get_email_details with one of these email_ids."
                        )
                    else:
                        results_text.append(f"Tool: {tr['tool']}\nResult: {str(result)[:500]}")
                else:
                    results_text.append(f"Tool: {tr['tool']}\nResult: {str(result)[:500]}")
            
            context_parts.append(f"Previous Tool Results:\n{chr(10).join(results_text)}")
        
        if thoughts:
            context_parts.append(f"Previous Thoughts:\n{chr(10).join(thoughts[-3:])}")
        
        context = "\n\n".join(context_parts) if context_parts else "No previous context."
        
        # Add warning if forcing synthesis
        synthesis_warning = ""
        if force_synthesize:
            synthesis_warning = "\n\n⚠️ CRITICAL: You have called the same tool multiple times. You MUST synthesize the results you have and provide a final_answer. Do NOT call any more tools!"
        
        # Create agent prompt
        agent_prompt = f"""You are an autonomous email assistant agent. You can use tools to help answer the user's query.
{synthesis_warning}

User Query: {query}

{context}

Available Tools:
{tools_description}

Based on the query and context, decide what to do next. Respond in JSON format with ONE of these actions:

1. If you need to use a tool:
{{
    "action": "use_tool",
    "tool_name": "tool_name_here",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "thought": "Why I'm using this tool"
}}

2. If you have enough information to answer:
{{
    "action": "final_answer",
    "content": "Your complete answer to the user",
    "thought": "Why this is the final answer"
}}

3. If you need to think more:
{{
    "action": "think",
    "thought": "What I'm analyzing or considering"
}}

CRITICAL RULES:
1. **For email queries, ALWAYS use search_emails_rag FIRST** (emails are indexed daily via autoindex)
2. **If search_emails_rag returns results, use get_email_details ONCE for full content** - DO NOT call it multiple times!
3. **If get_email_details returns 'complete: True' and 'body' field, you have the FULL email** - STOP and synthesize answer!
4. **Only use search_emails_gmail if RAG didn't find what you need**
5. **STOP and give final_answer when you have sufficient information** - don't keep searching!
6. **If tool result shows '✅ COMPLETE EMAIL RETRIEVED', you MUST synthesize and give final_answer** - do NOT call get_email_details again!
7. **Use conversation history** - If user asks about "previous question" or "what did I ask", check conversation history above
8. **Maximum 3-4 tool calls per query** - if you've called tools 3+ times, you likely have enough info

IMPORTANT:
- Use tools when you need to fetch or manipulate data
- You can call MULTIPLE tools if needed (one at a time)
- Give final_answer when you have sufficient information - DON'T keep searching!
- If previous tool results already answered the query, synthesize them and give final_answer
- Be autonomous - make decisions based on the query

Your decision (JSON only):"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an autonomous agent. Think step by step and use tools when needed. Always respond with valid JSON."
            },
            {
                "role": "user",
                "content": agent_prompt
            }
        ]
        
        response = await self.azure_client.generate_response(
            messages=messages,
            max_tokens=500
        )
        
        # Parse JSON decision
        try:
            decision = json.loads(response)
            logger.info(f"Agent decision: {decision['action']} - {decision.get('thought', '')[:100]}")
            return decision
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent decision: {response}")
            # Fallback
            return {
                "action": "final_answer",
                "content": "I had trouble deciding what to do. Please rephrase your query.",
                "thought": "JSON parse error"
            }
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters"""
        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            tool = self.available_tools[tool_name]
            function = tool['function']
            
            # Execute function (handle both sync and async)
            result = function(**parameters)
            
            if hasattr(result, '__await__'):
                result = await result
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

