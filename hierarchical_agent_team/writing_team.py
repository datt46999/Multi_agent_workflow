import os
import operator
import  functools 
from typing import Annotated, List, TypedDict, Optional, Dict
from tempfile import TemporaryDirectory
from pathlib import Path 

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent 

from hierarchical_agent_team.hierachical_agent import agent_node, create_team_supervisor
from hierarchical_agent_team.system_propmpts import writting_prompts
_TEM_DERECTORY = TemporaryDirectory()
WORKING_DERECTORY = Path(_TEM_DERECTORY.name)
@tool 
def create_outline(points:List[str], file_name:str)->str:
    """
    create outline and save
    """
    with (WORKING_DERECTORY/file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i+1}, {point}\n")
    return f"Outfile save to{file_name}"
@tool
def read_documents(file_name: str, start: Optional[int]= None, end: Optional[int] = None) ->str:
    """
    Read specified document
    """
    with (WORKING_DERECTORY/file_name).open('r') as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "".join(lines[start:end])


@tool
def writting_document(content:str, file_name:str)->str:
    """
    write and save text document
    """
    with(WORKING_DERECTORY/file_name).open('w') as file:
        file.write(content)
    return f"Document was save in {file}"
@tool
def edit_document(file_name:str, inseart:Dict[int, str])->str:
    """
    edit document by inserting text at specific line number
    """
    with (WORKING_DERECTORY/file_name).open('r') as file:
        lines = file.readlines()
    sort_inseart = sorted(inseart.items())

    for line_number, text in sort_inseart:
        if 1<= line_number<= len(lines)+1:
            lines.insert(line_number-1, text + "\n")
        else:
            return f"Error: line number {line_number} is out of range."
    with (WORKING_DERECTORY/file_name).open('w') as file:
        lines = file.writelines()

    return f"Document edited and saved to {file_name}"


@tool
def python_repl(code: str):
    """Execute python code to generate charts or perform calculations."""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

def writing_team_agent():
    """
    create the  document writting agent
    """
    class WrittingState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        team_member:str
        next: str
        current_file: str
    
    def prelude(state):
        written_files = []
        if not WORKING_DERECTORY.exists:
            WORKING_DERECTORY.mkdir()
        
        try:
            written_files = [
                f.relative_to(WORKING_DERECTORY) for f in WORKING_DERECTORY.rglob("*")
            ]
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Warning: Could not list files in working directory: {e}")
            pass
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }
    
    llm = ChatOpenAI(model=  "gpt-4o", temperature=0)

    doc_writer_agent = create_react_agent(llm, tools= [writting_document, edit_document, read_documents])
    context_doc_writer_agent = prelude | doc_writer_agent
    doc_writer_node = functools.partial(agent_node, agent = context_doc_writer_agent, name = "doc_witter")

    note_take_agent = create_react_agent(llm, tools =[create_outline, read_documents])
    context_note_take_agent = prelude | note_take_agent
    note_taking_node = functools.partial(agent_node, agent = context_note_take_agent, name = "Note_taking")

    chart_agent = create_react_agent(llm, tools = [read_documents, python_repl])
    context_chart_agent = prelude | chart_agent
    chart_agent_node = functools.partial(agent_node, agent = context_chart_agent, name = "Chart_agent")


    supervisor = create_team_supervisor(
        llm, 
        writting_prompts,
        ["doc_writter", "Note_taking", "Chart_agent"]

    )
    writting_build = StateGraph(WrittingState)
    writting_build.add_node("doc_witter", doc_writer_node)
    writting_build.add_node("Note_taking", note_taking_node)
    writting_build.add_node("Chart_agent", chart_agent_node )
    writting_build.add_node("supervisor", supervisor)

    writting_build.add_edge("doc_witter","supervisor")
    writting_build.add_edge("Note_taking","supervisor")
    writting_build.add_edge("Chart_agent","supervisor")
    writting_build.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"doc_witter": "doc_witter",
         "Note_taking": "Note_taking",
         "Chart_agent": "Chart_agent",
         "FINISH": END}
    )
    writting_build.add_edge(START, "supervisor")
    return writting_build.compile()
    

