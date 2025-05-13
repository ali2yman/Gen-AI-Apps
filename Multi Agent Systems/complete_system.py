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
import os
def create_rag_implementation_system(project_description, human_feedback=False):
    """
    Create a complete multi-agent system for implementing a RAG solution.
    
    Args:
        project_description: Detailed description of the RAG project
        human_feedback: Whether to enable human feedback
        
    Returns:
        The CrewAI crew ready to execute
    """
    
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
    
    # Create tasks
    
    # 1. Initial research task
    research_task = Task(
        description=f"""
        Research the current state of RAG (Retrieval-Augmented Generation) systems in 2025.
        Focus on:
        - Core components and architecture
        - Best practices for implementation
        - Latest innovations and techniques
        - Performance benchmarks
        - Real-world applications
        
        Project context: {project_description}
        
        Use the memory tools to store important facts and information for later use.
        Be thorough and ensure you collect comprehensive information from multiple sources.
        """,
        expected_output="""
        A comprehensive research report on RAG systems that includes:
        1. Detailed explanation of RAG architecture and components
        2. Current best practices for implementation
        3. Key innovations and techniques in 2025
        4. Relevant performance metrics and benchmarks
        5. Notable real-world applications and case studies
        6. Citations and references to sources
        """,
        agent=researcher,
        human_approval=human_feedback
    )
    
    # 2. Comparative analysis task
    analysis_task = Task(
        description=f"""
        Analyze the research findings to compare RAG systems with fine-tuning approaches.
        
        Your analysis should:
        1. Identify key strengths and limitations of RAG systems
        2. Compare RAG with fine-tuning across multiple dimensions:
           - Performance
           - Cost
           - Implementation complexity
           - Maintenance requirements
           - Use case suitability
        3. Develop a decision framework for when to use RAG vs. fine-tuning
        4. Identify hybrid approaches that combine both methods
        
        Project context: {project_description}
        
        Use the memory tools to retrieve facts gathered during research and to store your insights.
        """,
        expected_output="""
        A comprehensive analysis report that includes:
        1. Detailed comparison of RAG systems vs. fine-tuning
        2. Clear articulation of strengths and limitations of each approach
        3. Decision framework with specific recommendations
        4. Analysis of hybrid approaches
        5. Supporting evidence for all conclusions
        """,
        agent=analyst,
        context=[research_task],
        human_approval=human_feedback
    )
    
    # 3. Implementation design task
    design_task = Task(
        description=f"""
        Based on the research and analysis, design a RAG system architecture.
        
        Your design should include:
        1. High-level system architecture with all components
        2. Data flow diagram
        3. Component specifications:
           - Document processing pipeline
           - Embedding generation
           - Vector storage
           - Retrieval mechanism
           - Response generation
        4. Technology stack recommendations
        5. Implementation considerations and best practices
        
        Project context: {project_description}
        
        Use the memory tools to retrieve facts and insights from previous tasks.
        """,
        expected_output="""
        A detailed RAG system design document that includes:
        1. System architecture diagram (described in text)
        2. Component specifications
        3. Data flow explanation
        4. Technology recommendations
        5. Implementation guidelines
        """,
        agent=developer,
        context=[research_task, analysis_task],
        human_approval=human_feedback
    )
    
    # 4. Implementation task
    implementation_task = Task(
        description=f"""
        Implement a simple but functional RAG system based on the approved design.
        
        Your implementation should include:
        1. Code for all core components:
           - Document processing
           - Embedding generation
           - Vector storage
           - Retrieval mechanism
           - Response generation
        2. Clear documentation for each component
        3. Error handling and edge cases
        4. Usage examples
        
        Project context: {project_description}
        
        Use the code_generator tool to create well-structured, documented code.
        Use the memory tools to retrieve information from previous tasks.
        """,
        expected_output="""
        A complete implementation of a simple RAG system, including:
        1. Well-structured, documented code for all components
        2. Integration code showing how components work together
        3. Usage examples demonstrating the system in action
        4. Implementation notes including limitations and potential improvements
        """,
        agent=developer,
        context=[research_task, analysis_task, design_task],
        human_approval=human_feedback
    )
    
    # 5. Manager's coordination task (uses delegation)
    manager_task = create_delegated_tasks(
        manager,
        worker_agents,
        f"""
        Coordinate the implementation of a RAG (Retrieval-Augmented Generation) system
        based on the following requirements: {project_description}
        
        Your team needs to:
        1. Research the current state of RAG systems
        2. Analyze the strengths and limitations compared to fine-tuning
        3. Design a RAG system architecture
        4. Implement a simple but functional RAG system
        
        As the manager, ensure all tasks are completed properly and the final 
        deliverable meets all requirements.
        """
    )
    
    # Create the crew with hierarchical structure
    crew = Crew(
        agents=[manager, researcher, analyst, developer],
        tasks=[manager_task if human_feedback else research_task, 
               analysis_task, design_task, implementation_task],
        verbose=2,
        process=Process.hierarchical if human_feedback else Process.sequential,
        manager_agent=manager
    )
    
    return crew
 # Example usage
if __name__ == "__main__":
    project_description = """
    Build a RAG system optimized for technical documentation. The system should
    be able to process software documentation, API references, and technical guides,
    then answer specific technical questions with accurate citations to the source
    material. Focus on maximizing retrieval relevance and response accuracy.
    """
    
    # Create and run the system
    crew = create_rag_implementation_system(project_description, human_feedback=True)
    result = crew.kickoff()
    
    # Save the results
    with open("rag_implementation_results.txt", "w") as f:
        f.write(result)
    
    print("\nFinal Result:")
    print(result)