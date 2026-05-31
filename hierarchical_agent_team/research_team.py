import os
import operator
import functools
from typing import List, TypedDict, Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader, ArxivLoader
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent


from hierarchical_agent_team.hierachical_agent import create_team_supervisor, agent_node
from hierarchical_agent_team.system_propmpts import research_prompts
from dotenv import load_dotenv
load_dotenv()

tavil_tool = TavilySearchResults(max_results = 5)




@tool
def wiki_search(query: List[str])->str:
    """
    Search info of Wiki from query
    return maximun 2 results
    """
    search_docs = WikipediaLoader(query = query, load_max_doc= 2).load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata["source"]}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    return  "\n\n".join(
        [
            f'<Document name="{doc.metadata["source"]}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )


def research_team_agent():
    """
    create reseach agent
    """
    class ResearchState(TypedDict):
        messages : Annotated[List[BaseMessage], operator.add]
        team_members: List[str]
        next: str

    llm = ChatOpenAI(model = "gpt-4o",temperature = 0)

    search_agent = create_react_agent(llm, tools = [tavil_tool])
    search_node = functools.partial(agent_node, agent =search_agent, name = "Tavil_search")

    ScraperWeb = create_react_agent(llm, tools =[scrape_webpages, wiki_search, arxiv_search])
    ScraperWeb_node = functools.partial(agent_node, agent = ScraperWeb, name = "ScraperWeb")

    supervisor_agent = create_team_supervisor(
        llm, 
        research_prompts, 
        ["Tavil_search", "ScraperWeb"]
    )

    research_build = StateGraph(ResearchState)
    research_build.add_node("Tavil_search", search_node)
    research_build.add_node("ScraperWeb", ScraperWeb_node)
    research_build.add_node("supervisor", supervisor_agent)


    research_build.add_edge("Tavil_search","supervisor")
    research_build.add_edge("ScraperWeb","supervisor")
    research_build.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Tavil_search": "Tavil_search",
         "ScraperWeb": "ScraperWeb",
         "FINISH":END}
    )
    research_build.add_edge(START,"supervisor")
    return research_build.compile()







