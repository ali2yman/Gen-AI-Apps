from langchain_core.tools import tool
import json
import os
from typing import Dict, List, Any, Optional
import time
class MemoryTool:
    """Tool for agent memory management and contextual awareness."""
    
    def __init__(self, memory_file="agent_memory.json"):
        """
        Initialize the memory tool.
        
        Args:
            memory_file: File to store agent memories
        """
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """
        Load memory from file or initialize if it doesn't exist.
        
        Returns:
            Dictionary containing agent memories
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                # If there's an error loading, initialize new memory
                return self._initialize_memory()
        else:
            return self._initialize_memory()
    
    def _initialize_memory(self) -> Dict[str, Any]:
        """
        Initialize a new memory structure.
        
        Returns:
            New memory dictionary
        """
        memory = {
            "facts": [],
            "entities": {},
            "conversations": [],
            "tasks": {},
            "reflections": [],
            "last_updated": time.time()
        }
        
        # Save the initialized memory
        self._save_memory(memory)
        
        return memory
    
    def _save_memory(self, memory: Optional[Dict[str, Any]] = None) -> None:
        """
        Save memory to file.
        
        Args:
            memory: Memory dictionary to save (uses self.memory if None)
        """
        if memory is None:
            memory = self.memory
        
        memory["last_updated"] = time.time()
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory, f, indent=2)
    
    @tool("store_fact")
    def store_fact(self, fact: str) -> str:
        """
        Store an important fact in memory.
        
        Args:
            fact: The fact to remember
            
        Returns:
            Confirmation message
        """
        self.memory["facts"].append({
            "content": fact,
            "timestamp": time.time()
        })
        
        self._save_memory()
        
        return f"Fact stored in memory: '{fact}'"
    
    @tool("retrieve_facts")
    def retrieve_facts(self, query: str = None) -> str:
        """
        Retrieve facts from memory, optionally filtered by query.
        
        Args:
            query: Optional search term to filter facts
            
        Returns:
            String containing retrieved facts
        """
        facts = self.memory["facts"]
        
        if not facts:
            return "No facts stored in memory."
        
        if query:
            # Simple filtering by substring match
            filtered_facts = [f for f in facts if query.lower() in f["content"].lower()]
            
            if not filtered_facts:
                return f"No facts found matching query: '{query}'"
            
            result = f"Facts related to '{query}':\n\n"
            for i, fact in enumerate(filtered_facts, 1):
                result += f"{i}. {fact['content']}\n"
            
            return result
        else:
            # Return all facts
            result = "All stored facts:\n\n"
            for i, fact in enumerate(facts, 1):
                result += f"{i}. {fact['content']}\n"
            
            return result
    
    @tool("store_entity_information")
    def store_entity_information(self, entity_name: str, information: str) -> str:
        """
        Store or update information about a specific entity.
        
        Args:
            entity_name: Name of the entity (person, concept, project, etc.)
            information: Information to store about the entity
            
        Returns:
            Confirmation message
        """
        if entity_name not in self.memory["entities"]:
            self.memory["entities"][entity_name] = {
                "information": [information],
                "first_mentioned": time.time(),
                "last_updated": time.time()
            }
        else:
            self.memory["entities"][entity_name]["information"].append(information)
            self.memory["entities"][entity_name]["last_updated"] = time.time()
        
        self._save_memory()
        
        return f"Information about '{entity_name}' stored in memory."
    
    @tool("retrieve_entity")
    def retrieve_entity(self, entity_name: str) -> str:
        """
        Retrieve information about a specific entity.
        
        Args:
            entity_name: Name of the entity to retrieve
            
        Returns:
            String containing entity information
        """
        if entity_name not in self.memory["entities"]:
            return f"No information found for entity: '{entity_name}'"
        
        entity = self.memory["entities"][entity_name]
        
        result = f"Information about '{entity_name}':\n\n"
        for i, info in enumerate(entity["information"], 1):
            result += f"{i}. {info}\n"
        
        return result
    
    @tool("log_task")
    def log_task(self, task_name: str, status: str, details: str) -> str:
        """
        Log information about a task or subtask.
        
        Args:
            task_name: Name or identifier for the task
            status: Status of the task (e.g., "started", "in_progress", "completed", "failed")
            details: Additional details about the task
            
        Returns:
            Confirmation message
        """
        if task_name not in self.memory["tasks"]:
            self.memory["tasks"][task_name] = {
                "status": status,
                "history": [{
                    "status": status,
                    "details": details,
                    "timestamp": time.time()
                }],
                "created": time.time(),
                "last_updated": time.time()
            }
        else:
            self.memory["tasks"][task_name]["status"] = status
            self.memory["tasks"][task_name]["history"].append({
                "status": status,
                "details": details,
                "timestamp": time.time()
            })
            self.memory["tasks"][task_name]["last_updated"] = time.time()
        
        self._save_memory()
        
        return f"Task '{task_name}' logged with status: {status}"
    
    @tool("get_task_status")
    def get_task_status(self, task_name: str = None) -> str:
        """
        Get status information about tasks.
        
        Args:
            task_name: Optional specific task to get status for
            
        Returns:
            String containing task status information
        """
        if not self.memory["tasks"]:
            return "No tasks have been logged."
        
        if task_name:
            if task_name not in self.memory["tasks"]:
                return f"No task found with name: '{task_name}'"
            
            task = self.memory["tasks"][task_name]
            
            result = f"Task: '{task_name}'\n"
            result += f"Status: {task['status']}\n"
            result += f"Created: {time.ctime(task['created'])}\n"
            result += f"Last Updated: {time.ctime(task['last_updated'])}\n\n"
            result += "History:\n"
            
            for i, entry in enumerate(task["history"], 1):
                result += f"{i}. [{time.ctime(entry['timestamp'])}] Status: {entry['status']}\n"
                result += f"   Details: {entry['details']}\n"
            
            return result
        else:
            # Summarize all tasks
            result = "Task Summary:\n\n"
            for name, task in self.memory["tasks"].items():
                result += f"• {name}: {task['status']} (Last updated: {time.ctime(task['last_updated'])})\n"
            
            return result
    
    @tool("add_reflection")
    def add_reflection(self, content: str) -> str:
        """
        Add a reflection or insight to memory.
        
        Args:
            content: The reflection content
            
        Returns:
            Confirmation message
        """
        self.memory["reflections"].append({
            "content": content,
            "timestamp": time.time()
        })
        
        self._save_memory()
        
        return f"Reflection added to memory."
    
    @tool("get_reflections")
    def get_reflections(self) -> str:
        """
        Retrieve all stored reflections.
        
        Returns:
            String containing all reflections
        """
        reflections = self.memory["reflections"]
        
        if not reflections:
            return "No reflections stored in memory."
        
        result = "Stored reflections:\n\n"
        for i, reflection in enumerate(reflections, 1):
            result += f"{i}. [{time.ctime(reflection['timestamp'])}]\n"
            result += f"   {reflection['content']}\n\n"
        
        return result