from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import TaskOutput
from crewai.agents.agent_report import AgentReport
from config import get_llama_llm
 # Create a human feedback function
def get_human_feedback(agent_name, task_description, output):
    print("\n" + "="*50)
    print(f"Human Feedback Required for {agent_name}")
    print("="*50)
    print(f"\nTask: {task_description}")
    print(f"\nCurrent Output:\n{output}")
    
    feedback = input("\nPlease provide feedback or guidance (press Enter to approve): ")
    
    if feedback.strip():
        return False, feedback
    else:
        return True, "Output approved"
 # Create a task with human-in-the-loop approval
def create_human_approval_task(agent, description, expected_output):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        human_approval=True,  # Require human approval
        human_feedback_fn=get_human_feedback  # Custom feedback function
    )
    return task
 # Example implementation of a crew with different process types
def run_process_example():
    """Demonstrate different process management approaches in CrewAI"""
    
    # Initialize agents (simplified for example)
    manager = Agent(role="Manager", goal="Coordinate the team", backstory="...", llm=get_llama_llm())
    researcher = Agent(role="Researcher", goal="Find information", backstory="...", llm=get_llama_llm())
    analyst = Agent(role="Analyst", goal="Analyze information", backstory="...", llm=get_llama_llm())
    
    # Create tasks
    task1 = create_human_approval_task(
        researcher,
        "Research the latest advancements in AI agents",
        "A comprehensive research report on AI agents"
    )
    
    task2 = Task(
        description="Analyze the research findings to identify key trends",
        expected_output="An analytical report with insights on AI agent trends",
        agent=analyst,
        context=[task1]  # This task depends on the research task
    )
    
    # 1. Sequential Process (one after another)
    crew_sequential = Crew(
        agents=[manager, researcher, analyst],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )
    
    # 2. Hierarchical Process (manager coordinates)
    crew_hierarchical = Crew(
        agents=[manager, researcher, analyst],
        tasks=[task1, task2],
        verbose=True,
        process=Process.hierarchical,
        manager_agent=manager
    )
    
    # Execute with chosen process
    result = crew_sequential.kickoff()
    # OR
    # result = crew_hierarchical.kickoff()
    
    return result