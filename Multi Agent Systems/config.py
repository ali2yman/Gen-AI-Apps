from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Initialize Ollama LLM
def get_llama_llm(temperature=0.1, system_prompt=None):
    """
    Configure and return an Ollama LLM instance using Llama 3.2-1b

    Args:
        temperature (float): Controls randomness (lower = more deterministic)
        system_prompt (str): Optional system prompt to guide model behavior

    Returns:
        Ollama: Configured Ollama LLM instance
    """
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant working as part of a "
            "multi-agent system. You are precise, factual, and helpful. "
            "You complete tasks thoroughly and report your results clearly."
        )

    # Configure the Ollama LLM
    llm = Ollama(
        model="llama3.2:1b",
        temperature=temperature,
        system=system_prompt,
        num_ctx=4096,         # Context window size
        num_thread=4,         # Number of threads to use
        repeat_penalty=1.1,   # Penalty for repetition
        top_k=40,             # Sample from top K tokens
        top_p=0.9,            # Nucleus sampling probability
        num_predict=512       # Maximum tokens to generate
    )
    return llm
