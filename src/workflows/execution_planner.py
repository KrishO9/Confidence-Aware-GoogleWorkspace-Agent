"""
Execution Planner
LLMCompiler-inspired parallel execution planning
Generates DAG of operations and executes them in parallel where possible
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient

logger = get_logger()


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ExecutionTask:
    """Represents a single execution task"""
    task_id: str
    operation: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # Task IDs that must complete first
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None


class ExecutionPlanner:
    """
    Plans and executes operations in parallel using DAG-based approach
    Inspired by LLMCompiler paradigm
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
        self.available_operations = {
            "search_emails": "Search for emails with given criteria",
            "get_recent_emails": "Get recent emails from last N days",
            "search_calendar": "Search calendar events",
            "get_upcoming_events": "Get upcoming calendar events",
            "list_tasks": "List tasks",
            "create_task": "Create a new task",
            "semantic_search": "Perform semantic search on indexed emails",
            "extract_information": "Extract specific information from results"
        }
    
    async def generate_execution_plan(
        self,
        user_query: str,
        available_tools: List[str]
    ) -> List[ExecutionTask]:
        """
        Generate execution plan with parallel operations
        
        Args:
            user_query: User's query
            available_tools: List of available tool names
            
        Returns:
            List of execution tasks in DAG order
        """
        try:
            operations_desc = "\n".join([
                f"- {op}: {desc}"
                for op, desc in self.available_operations.items()
                if op in available_tools or not available_tools
            ])
            
            planning_prompt = f"""You are an execution planner. Given a user query, break it down into a sequence of operations that can be executed.

User Query: {user_query}

Available Operations:
{operations_desc}

Create an execution plan in JSON format. Each task should have:
- task_id: Unique identifier (task_1, task_2, etc.)
- operation: Name of the operation to perform
- parameters: Dictionary of parameters for the operation
- dependencies: List of task_ids that must complete before this task (empty for parallel tasks)

IMPORTANT: Identify operations that can run in PARALLEL (no dependencies on each other).

Example format:
[
    {{
        "task_id": "task_1",
        "operation": "search_emails",
        "parameters": {{"query": "project deadline", "max_results": 20}},
        "dependencies": []
    }},
    {{
        "task_id": "task_2",
        "operation": "search_calendar",
        "parameters": {{"query": "project meeting"}},
        "dependencies": []
    }},
    {{
        "task_id": "task_3",
        "operation": "extract_information",
        "parameters": {{"focus": "deadlines and dates"}},
        "dependencies": ["task_1", "task_2"]
    }}
]

Notice task_1 and task_2 can run in PARALLEL (no dependencies), while task_3 depends on both.

Now generate the execution plan for the query above:"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert execution planner. Always respond with valid JSON array."
                },
                {
                    "role": "user",
                    "content": planning_prompt
                }
            ]
            
            response = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=2000
            )
            
            # Parse JSON response
            plan_data = json.loads(response)
            
            # Convert to ExecutionTask objects
            tasks = []
            for task_data in plan_data:
                task = ExecutionTask(
                    task_id=task_data["task_id"],
                    operation=task_data["operation"],
                    parameters=task_data.get("parameters", {}),
                    dependencies=task_data.get("dependencies", [])
                )
                tasks.append(task)
            
            logger.info(f"Generated execution plan with {len(tasks)} tasks")
            self._log_plan_summary(tasks)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error generating execution plan: {e}")
            # Fallback: simple single-task plan
            return [
                ExecutionTask(
                    task_id="task_1",
                    operation="search_emails",
                    parameters={"query": user_query},
                    dependencies=[]
                )
            ]
    
    def _log_plan_summary(self, tasks: List[ExecutionTask]):
        """Log execution plan summary"""
        # Find parallel tasks (no dependencies)
        parallel_tasks = [t for t in tasks if not t.dependencies]
        sequential_tasks = [t for t in tasks if t.dependencies]
        
        logger.info(f"Execution Plan Summary:")
        logger.info(f"  - Parallel tasks (wave 1): {len(parallel_tasks)}")
        logger.info(f"  - Sequential/dependent tasks: {len(sequential_tasks)}")
        
        for task in tasks:
            deps = f" [depends on: {', '.join(task.dependencies)}]" if task.dependencies else " [parallel]"
            logger.info(f"  - {task.task_id}: {task.operation}{deps}")
    
    async def execute_plan(
        self,
        tasks: List[ExecutionTask],
        tool_executor: Any
    ) -> Dict[str, Any]:
        """
        Execute the plan with parallel execution
        
        Args:
            tasks: List of execution tasks
            tool_executor: Object with methods matching operation names
            
        Returns:
            Execution results
        """
        try:
            task_map = {task.task_id: task for task in tasks}
            completed_tasks = set()
            results = {}
            
            # Execute in waves (topological order)
            while len(completed_tasks) < len(tasks):
                # Find tasks ready to execute (all dependencies met)
                ready_tasks = [
                    task for task in tasks
                    if task.status == TaskStatus.PENDING
                    and all(dep in completed_tasks for dep in task.dependencies)
                ]
                
                if not ready_tasks:
                    # Check for blocked tasks
                    pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
                    if pending_tasks:
                        logger.error("Execution blocked: circular dependencies or failed dependencies")
                        for task in pending_tasks:
                            task.status = TaskStatus.BLOCKED
                    break
                
                logger.info(f"Executing wave with {len(ready_tasks)} parallel tasks")
                
                # Execute ready tasks in parallel
                await self._execute_parallel_tasks(ready_tasks, tool_executor)
                
                # Update completed tasks
                for task in ready_tasks:
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task.task_id)
                        results[task.task_id] = task.result
            
            # Compile final results
            final_result = {
                "success": len(completed_tasks) == len(tasks),
                "completed_tasks": len(completed_tasks),
                "total_tasks": len(tasks),
                "results": results,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "operation": t.operation,
                        "status": t.status.value,
                        "error": t.error
                    }
                    for t in tasks
                ]
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return {
                "success": False,
                "error": str(e),
                "completed_tasks": len(completed_tasks) if 'completed_tasks' in locals() else 0,
                "total_tasks": len(tasks)
            }
    
    async def _execute_parallel_tasks(
        self,
        tasks: List[ExecutionTask],
        tool_executor: Any
    ):
        """
        Execute multiple tasks in parallel
        
        Args:
            tasks: Tasks to execute
            tool_executor: Tool executor object
        """
        async def execute_single_task(task: ExecutionTask):
            try:
                task.status = TaskStatus.RUNNING
                logger.info(f"Executing {task.task_id}: {task.operation}")
                
                # Get the method from tool_executor
                if hasattr(tool_executor, task.operation):
                    method = getattr(tool_executor, task.operation)
                    
                    # Execute with parameters
                    result = method(**task.parameters)
                    
                    # Handle async methods
                    if asyncio.iscoroutine(result):
                        result = await result
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    logger.info(f"Completed {task.task_id}")
                else:
                    raise ValueError(f"Operation not found: {task.operation}")
                    
            except Exception as e:
                logger.error(f"Error executing {task.task_id}: {e}")
                task.error = str(e)
                task.status = TaskStatus.FAILED
        
        # Execute all tasks in parallel
        await asyncio.gather(*[execute_single_task(task) for task in tasks])
    
    def optimize_plan(self, tasks: List[ExecutionTask]) -> List[ExecutionTask]:
        """
        Optimize execution plan by identifying more parallelization opportunities
        
        Args:
            tasks: Execution tasks
            
        Returns:
            Optimized tasks
        """
        # Simple optimization: remove unnecessary dependencies
        # In production, implement more sophisticated optimizations
        
        for task in tasks:
            # Remove dependencies that don't affect this task
            necessary_deps = []
            for dep in task.dependencies:
                # Check if dependency result is actually used
                # This is simplified - real implementation would analyze parameter usage
                necessary_deps.append(dep)
            
            task.dependencies = necessary_deps
        
        return tasks

