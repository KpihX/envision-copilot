from .gemini_agent import GeminiAgent
from .mistral_agent import MistralAgent
from .gpt_agent import GPTAgent
from .llama3_agent import Llama3Agent
from .groq_agent import GroqAgent
from .qwen_agent import QwenAgent
from .base import LLMAgent
from envision_rag.config_manager import get_config

def prepare_agent(agent_name: str) -> LLMAgent:
        """Prepare and return the default LLM agent based on configuration."""
        agent : LLMAgent
        if agent_name == 'gemini':
            agent = GeminiAgent()
        elif agent_name == 'mistral':
            agent = MistralAgent()
        elif agent_name == 'gpt':
            agent = GPTAgent()
        elif agent_name == 'llama3':
            agent = Llama3Agent()
        elif agent_name == 'llama3.2':
            agent = Llama3Agent('llama3.2')
        elif agent_name == 'qwen':
            agent = QwenAgent()
        elif agent_name == 'groq':
            agent = GroqAgent()
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")
        agent.initialize()
        return agent

def prepare_default_agent() -> LLMAgent:
    """Prepare and return the default LLM agent based on configuration."""
    return prepare_agent(get_config().get_default_agent().lower())


def prepare_benchmark_agent() -> LLMAgent:
    """Prepare and return the benchmark LLM agent based on configuration."""
    return prepare_agent(get_config().get_benchmark_agent().lower())

def prepare_embedder_summary_agent() -> LLMAgent: #OLD BACKUP FUNCTION
    """Prepare and return the embedder summary LLM agent based on configuration."""
    return prepare_agent(get_config().get_embedder_summary_agent().lower())

def prepare_chunk_summary_agent() -> LLMAgent:
    """Prepare and return the embedder summary LLM agent based on configuration."""
    return prepare_agent(get_config().get_chunk_summary_agent().lower())