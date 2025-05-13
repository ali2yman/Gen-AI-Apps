from crewai import Agent, Task, Crew, Process
from config import get_llama_llm
from agents.manager import create_manager_agent
from agents.researcher import create_researcher_agent
from agents.analyst import create_analyst_agent
from agents.developer import create_developer_agent
from tools.web_search import WebSearchTool
from tools.document_tool import DocumentAnalyzerTool
from tools.code_tool import CodeGeneratorTool
from tools.memory_tool import MemoryTool
from utils.helpers import create_delegated_tasks
 # Initialize our tools
web_search_tool = WebSearchTool()
document_analyzer = DocumentAnalyzerTool()
code_generator = CodeGeneratorTool()
memory_tool = MemoryTool()
 # Create our agents
manager = create_manager_agent(tools=[memory_tool])
researcher = create_researcher_agent(tools=[web_search_tool, document_analyzer, memory_tool])
analyst = create_analyst_agent(tools=[document_analyzer, memory_tool])
developer = create_developer_agent(tools=[code_generator, memory_tool])
 # Organize agents in a dictionary for easy reference
worker_agents = {
    "researcher": researcher,
    "analyst": analyst,
    "developer": developer
}
 # Create the main task for the manager
main_task = create_delegated_tasks(
    manager,
    worker_agents,
    """Research the current state of RAG (Retrieval-Augmented Generation) systems 
    in 2025, analyze their strengths and limitations compared to fine-tuning 
    approaches, and develop a simple implementation of a RAG system."""
 )
 # Create the crew with hierarchical structure
crew = Crew(
    agents=[manager, researcher, analyst, developer],
    tasks=[main_task],
    verbose=2,  # Increased verbosity for debugging
    process=Process.sequential,  # Tasks will be executed in order
    manager_agent=manager  # Set the manager as the coordinator
 )
 # Start the crew's work
result = crew.kickoff()
print("\nFinal Result:")
print(result)