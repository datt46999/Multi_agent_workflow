import os
import functools
import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import  BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel
from agent_supervisor.system_prompts import system_prompt
from dotenv import load_dotenv
load_dotenv()


tavily_tools = TavilySearchResults(max_results = 5)
python_repl_tool = PythonREPLTool()

members = ["Researcher", "Coder"]
options = ["FINISH"] + members
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name = "messages"),
        (
            "system",
            "Give the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options =str(options), members = ", ".join(members))


class RouteResponse(BaseModel):
    next: Literal["FINISH", "Researcher", "Coder"]
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def agent_node(state, agent, name):
    """
    Proccess state through an agent and return update state
    """
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content = result["messages"][-1].content, name = name)]}

def supervisor_agent(state):
    """
    Supervisor agent that  decides which worker  should act next
    """
    llm = ChatOpenAI(model = "gpt-4", temperature =0)
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)




def create_supervisor_graph():
    """
    Create and configure the Sepervisor graph
    """

    llm = ChatOpenAI(model = "gpt-4", temperature =0)

    research_agent = create_react_agent(llm, tools = [tavily_tools])
    research_node = functools.partial(agent_node, agent = research_agent, name = "Researcher")

    code_agent = create_react_agent(llm, tools = [python_repl_tool])
    code_node = functools.partial(agent_node, agent = code_agent, name = "Coder")


    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Coder", code_node)
    workflow.add_node("supervisor", supervisor_agent)

    for member in members:
        workflow.add_edge(member, "supervisor")
    
    conditional_map ={k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_edge(START, "supervisor")

    return workflow.compile()


def main():
    """
    Run supervisor system with example queries
    """
    graph = create_supervisor_graph()

    print("Example 1: Code  Hello World")

    for s in graph.stream(
        {
            "messages":[
                HumanMessage(content="Code hello world and print it to the terminal")
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            print("---"*100)

    print("Example 2: Research Resport")

    for s in graph.stream(
        {
            "messages":[
                HumanMessage(content = "Write a brief research report on pikas.")
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            print("-----"*100)
if __name__ =="__main__":
    main()