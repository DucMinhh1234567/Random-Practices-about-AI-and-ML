from langchain_ollama import OllamaLLM
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_classic.memory import ConversationBufferMemory
from langsmith import Client

llm = OllamaLLM(model='phi3')

# Decorator @tool của LangChain dùng docstring của hàm làm mô tả tool cho model
# Nên là phải bổ sung vào

# Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: only numbers and + - * / ** ( ) . allowed"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f'Error: {str(e)}'

# Simulated knowledge base lookup tool
@tool
def knowledge_base(query: str) -> str:
    """Look up short facts about Python, AI agents, Ollama, and RAG."""
    kbs = {
        "python": "Python is a beginner-friendly programming language widely used in AI and data science.",
        "ai agent": "An AI agent is a program that uses a language model to reason and take actions.",
        "ollama": "Ollama is a tool for running language models locally on your computer.",
        "rag": "RAG is an AI technique that improves large language models (LLMs) by fetching data from outside sources to generate answers.",
    }

    for key in kbs:
        if key in query.lower():
            return kbs[key]
    return "No information found for that query."

tools = [calculator, knowledge_base]

# Memory to track conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ReAct prompt template
prompt = Client().pull_prompt(
    "hwchase17/react-chat",
    dangerously_pull_public_prompt=True,
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

print(agent_executor.invoke({"input": "What is an AI agent?"})["output"])
print(agent_executor.invoke({"input": "Now tell me what Ollama is."})["output"])
print(agent_executor.invoke({"input": "Calculate 50 multiplied by 12 then powered by 2 then divied by 3."})["output"])
print(agent_executor.invoke({"input": "Finnaly, tell me what is RAG and how does it work."})["output"])