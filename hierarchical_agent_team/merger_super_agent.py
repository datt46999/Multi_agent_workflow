import os
import operator
import functools
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import StateGraph, START, END

from hierarchical_agent_team.hierachical_agent import create_team_supervisor
from hierarchical_agent_team.research_team import research_team_agent
from hierarchical_agent_team.writing_team import writing_team_agent
from hierarchical_agent_team.system_propmpts import  agent_system_prompt

load_dotenv()
os.environ["USER_AGENT"] = "HierarchicalAgentTeams/1.0"

def create_super_agent():
    """
    create top level supervisor graph
    """

    class State(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str
    

    def get_last_messages(state: State)->dict:
        """
        get the last message from state and return it like ditionally
        """
        return {"messages": [state["messages"][-1]]}

    def join_graph(response: dict)->dict:
        """
        join the graph and respond with current state
        """    
        return {"messages": [response["messages"][-1]]}
    llm = ChatOpenAI(model = "gpt-4o", temperature = 0) 
    research_chain = research_team_agent()
    writting_chain = writing_team_agent()

    supervisor = create_team_supervisor(
        llm, 
        agent_system_prompt,
        ["ResearchTeam", "WritingTeam"]
    )
    super_graph = StateGraph(State)
    super_graph.add_node("ResearchTeam", get_last_messages | research_chain | join_graph)
    super_graph.add_node("WritingTeam", get_last_messages | writting_chain | join_graph)
    super_graph.add_node("supervisor", supervisor)
    
    super_graph.add_edge("ResearchTeam","supervisor")
    super_graph.add_edge("WritingTeam","supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"ResearchTeam": "ResearchTeam",
         "WritingTeam": "WritingTeam",
         "FINISH": END,}
    )
    super_graph.add_edge(START, "supervisor")

    return super_graph.compile()

def main():
    graph_system = create_super_agent()
    
    initial_state = {
        "messages": [
            HumanMessage(
                content = "Write a brief research report on the North American sturgeon. Include a chart."
            )
        ],
        "next": "ResearchTeam"
    }

    for s in graph_system.stream(
        initial_state,
        {"recursion_limit": 150},
    ):
        if "__end__" not in s:
            print(s)
            print("----")
        

if __name__ == "__main__":
    main()

