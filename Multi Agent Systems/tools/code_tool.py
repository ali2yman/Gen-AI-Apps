from langchain_core.tools import tool
from langchain_community.llms import Ollama
import re
import os
import json
from typing import Dict, Any, List, Optional
class CodeGeneratorTool:
    """Advanced tool for generating and analyzing code."""
    
    def __init__(self):
        """Initialize the code generator tool."""
        self.llm = Ollama(model="llama3.2:1b", temperature=0.2)
        self.code_snippets = {}  # Store generated code snippets
        self.snippet_counter = 0
    
    @tool("generate_code")
    def generate_code(self, specification: str) -> str:
        """
        Generate code based on detailed specifications.
        
        Args:
            specification: Detailed description of what the code should do
            
        Returns:
            A string containing generated code with explanations
        """
        try:
            # Create a tailored prompt for code generation
            prompt = f"""
 Generate Python code based on the following specification. 
Be sure to include:
 1. Well-structured code that follows best practices
 2. Comprehensive comments explaining the logic
 3. Error handling where appropriate
 4. Examples of how to use the code
 SPECIFICATION:
 {specification}
 Return only the properly formatted code with comments and usage examples.
 """
            # Generate the code
            response = self.llm.invoke(prompt)
            
            # Extract code blocks (assuming the response contains markdown-style code blocks)
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            
            if not code_blocks:
                # If no code blocks are found, assume the entire response is code
                code = response
            else:
                # Join multiple code blocks with separators
                code = "\n\n# ----------\n\n".join(code_blocks)
            
            # Store the generated code for future reference
            self.snippet_counter += 1
            snippet_id = f"snippet_{self.snippet_counter}"
            self.code_snippets[snippet_id] = code
            
            # Format the response
            response = f"""
 Code generated successfully (ID: {snippet_id})
 {code}
 To analyze this code, use the analyze_code tool with the snippet ID.
 To refine this code, use the refine_code tool with the snippet ID and refinement instructions.
 """
            
            return response
            
        except Exception as e:
            return f"Error generating code: {str(e)}"
    
    @tool("analyze_code")
    def analyze_code(self, code_input: str) -> str:
        """
        Analyze code to provide feedback and suggestions.
        
        Args:
            code_input: Either a snippet ID or direct code to analyze
            
        Returns:
            Analysis and recommendations
        """
        try:
            # Determine if input is a snippet ID or actual code
            if code_input.startswith("snippet_") and code_input in self.code_snippets:
                code = self.code_snippets[code_input]
            else:
                code = code_input
            
            # Create a prompt for code analysis
            prompt = f"""
 Analyze the following Python code. Provide:
 1. A brief explanation of what the code does
 2. Potential bugs or issues
 3. Performance considerations
 4. Style improvements
 5. Security concerns (if any)
 CODE:
 ```python
 {code}
 ```
 Format your response in clear sections.
 """
            # Generate the analysis
            analysis = self.llm.invoke(prompt)
            
            return f"Code Analysis:\n\n{analysis}"
            
        except Exception as e:
            return f"Error analyzing code: {str(e)}"
    
    @tool("refine_code")
    def refine_code(self, input_str: str) -> str:
        """
        Refine existing code based on new requirements or feedback.
        
        Args:
            input_str: Format should be "snippet_id: refinement instructions"
            
        Returns:
            Refined code
        """
        try:
            # Parse input
            parts = input_str.split(":", 1)
            if len(parts) != 2:
                return "Error: Input should be in format 'snippet_id: refinement instructions'"
            
            snippet_id = parts[0].strip()
            refinements = parts[1].strip()
            
            if snippet_id not in self.code_snippets:
                return f"Error: Snippet ID '{snippet_id}' not found"
            
            original_code = self.code_snippets[snippet_id]
            
            # Create a prompt for code refinement
            prompt = f"""
 Refine the following Python code based on these requirements:
 ORIGINAL CODE:
 ```python
 {original_code}
 ```
 REFINEMENT REQUIREMENTS:
 {refinements}
 Return the improved code only, keeping all the good parts of the original code while implementing the requested change
 """
            # Generate the refined code
            response = self.llm.invoke(prompt)
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            
            if not code_blocks:
                # If no code blocks are found, assume the entire response is code
                refined_code = response
            else:
                # Join multiple code blocks with separators
                refined_code = "\n\n# ----------\n\n".join(code_blocks)
            
            # Store the refined code
            self.snippet_counter += 1
            new_snippet_id = f"snippet_{self.snippet_counter}"
            self.code_snippets[new_snippet_id] = refined_code
            
            # Format the response
            response = f"""
 Code refined successfully (New ID: {new_snippet_id})
 {refined_code}
 Original code preserved as {snippet_id}.
 """
            
            return response
            
        except Exception as e:
            return f"Error refining code: {str(e)}"
    
    @tool("execute_code")
    def execute_code(self, input_str: str) -> str:
        """
        Simulates executing code (actual execution disabled for security).
        
        Args:
            input_str: Either a snippet ID or code with input values
            
        Returns:
            Simulated execution results
        """
        try:
            # Parse input - could be a snippet ID or actual code
            if ":" in input_str:
                parts = input_str.split(":", 1)
                code_ref = parts[0].strip()
                inputs = parts[1].strip()
            else:
                code_ref = input_str.strip()
                inputs = ""
            
            # Get the code
            if code_ref.startswith("snippet_") and code_ref in self.code_snippets:
                code = self.code_snippets[code_ref]
            else:
                return "Error: Please provide a valid snippet ID or use the generate_code tool first"
            
            # For security reasons, we don't actually execute the code
            # Instead, we simulate execution with LLM
            
            prompt = f"""
 I want you to simulate executing this Python code:
 ```python
 {code}
 ```
 {f"With these inputs: {inputs}" if inputs else ""}
 Show me what the output would be if this code were executed.
 Format your response as if you were showing the actual program output.
 If there would be errors, show those too.
 """
            # Simulate execution
            execution_result = self.llm.invoke(prompt)
            
            return f"Simulated Execution Result:\n\n{execution_result}"
            
        except Exception as e:
            return f"Error in code execution simulation: {str(e)}"
    
    @tool("list_code_snippets")
    def list_code_snippets(self) -> str:
        """
        List all stored code snippets with their IDs.
        
        Returns:
            A formatted list of available code snippets
        """
        if not self.code_snippets:
            return "No code snippets have been generated yet."
        
        result = "Available Code Snippets:\n\n"
        for snippet_id, code in self.code_snippets.items():
            # Extract the first line as a description (usually a comment or function def)
            first_line = code.split('\n')[0][:50] + ('...' if len(code.split('\n')[0]) > 50 else '')
            code_size = len(code.split('\n'))
            
            result += f"ID: {snippet_id}\n"
            result += f"Preview: {first_line}\n"
            result += f"Size: {code_size} lines\n\n"
        
        return result