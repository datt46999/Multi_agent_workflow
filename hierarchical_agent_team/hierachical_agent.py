from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage



    

def agent_node(state, agent, name):
    """Process state through an agent and return updated state."""
    try:
        result = agent.invoke(state)
        if not isinstance(result, dict) or "messages" not in result:
            raise ValueError(f"Agent {name} returned invalid result format: {result}")
        return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}
    except Exception as e:
        print(f"Error in agent {name}: {e}")
        return {
            "messages": [
                HumanMessage(
                    content=f"Error occurred in {name}: {str(e)}",
                    name=name
                )
            ]
        }
def create_team_supervisor(llm : ChatOpenAI, system_prompt:str, member:List[str]):
    """
    create supervisor agent
    """
    options = ["FINISH"] + member
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
             MessagesPlaceholder(variable_name = "messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                "\nRespond with ONLY the name of the next role or FINISH.",
            ),
        ]
    ).partial(options = str(options), team_members =", ".join(member))
    def parse_oupput(messages)->dict:
        """parse output to get next role"""

        if hasattr(messages, "content"):
            output = messages.content.strip()
        else:
            output = str(messages).strip()
        
        if output not in options:
            print(f"Waring: Invalid output '{output}', defauting to FINISH")
            return {'next': "FINISH"}
        return {'next': output}
    chain = prompt | llm | parse_oupput
    return chain

