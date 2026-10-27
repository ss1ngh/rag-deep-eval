import os

from .objects import Answer
from .import tools as tl
from langchain_openai import ChatOpenAI
from .tools import collecting_info
from .prompt import system_agent_prompt
from langchain.agents import create_agent
from langchain.messages import HumanMessage
 

llm = ChatOpenAI(
    model='deepseek-v4.1-flash:free',
    # base_url='https://openrouter.ai/api/v1',
    base_url='https://tokenharbor.ai/v1',
    api_key=os.getenv("TOKENHARBOR_API_KEY")
    )

agent = create_agent(
    model = llm,
    tools = [collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question) -> dict:
    """ Answer the user's question about the PDF """

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages" : [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer" : structured_response.answer,
        "context" : structured_response.context
    }
