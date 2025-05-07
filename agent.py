import os
import re
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt.tool_node import _get_state_args
from typing_extensions import TypedDict

loadenv = load_dotenv()




db = SQLDatabase.from_uri("sqlite:///Chinook.db")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    schema: str
    sql_query: str
    error: str

if os.path.exists("schema.txt"):
    # read the schema from the file
    with open("schema.txt", "r") as f:
        db_schema = f.read()
else:
    # get the schema from the database
    db_schema = db.get_table_info()
    # write the schema to a file
    with open("schema.txt", "w") as f:
        f.write(db_schema)


def extract_sql(sql) -> str:
    pattern = r"```sql(.*?)```"

    sql_code_snippets = re.findall(pattern, sql, re.DOTALL)

    if len(sql_code_snippets) > 0:
        sql = sql_code_snippets[-1].strip()

    return sql

def check_for_dml(query: str) -> str:

    DML_COMMANDS = ['INSERT', 'UPDATE', 'DELETE', 'REPLACE', 'CALL', 'LOAD DATA', 'MERGE']

    query_upper = query.upper()
    for command in DML_COMMANDS:
        # Match only at the beginning or after semicolon (to detect mid-query DML)
        if re.search(rf'(^|\s|;)({command})\b', query_upper):
            print(f"❌ Error: DML command '{command}' is not allowed.")
            return True
    print("✅ Query is safe (no DML detected).")
    return False


fixer_system_prompt = f"""You are a sqlite expert with a strong attention to detail. The database has the following schema: {db_schema}."""

fixer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", fixer_system_prompt),
        ("user", "The following query failed: {sql_query}. The error message was: {error}."),
        ("assistant", "Please rewrite the query to be syntactically correct."),
    ]
)
fixer = fixer_prompt | llm


@tool
def execute(sql_query: str):
    """
    Executes the given SQL query present in a message and retries if it fails.
    Args:
        sql_query_msg (str): The message containing the SQL query to execute.
    Returns:
        str: The result of the query execution.
    """
    # sql_query = extract_sql(sql_query) # not needed if used as a tool since the LLM will pass the SQL query directly and not the entire message
    max_retries = 3
    result = db.run_no_throw(sql_query)
    if check_for_dml(sql_query):
        return "Error: DML commands are not allowed."

    if result.startswith("Error:"):
        for i in range(max_retries):
            fixer_message = fixer.invoke(
                {
                    "sql_query": sql_query,
                    "error": result,
                }
            )
            print(f"Fixer Message: {fixer_message}")
            query = extract_sql(fixer_message.content)
            result = db.run_no_throw(query)
            if not result.startswith("Error:"):
                return result # if the qurey is successful after retries, return the result
        return "Could not execute the query after 3 retries." # failure after 3 retries
    return  result # if the original query is successful, return the result

db_tools = [execute]
tools_by_name: dict[str, BaseTool] = {_tool.name: _tool for _tool in db_tools}

query_gen_system = f"""You are a highly skilled SQL expert with a strong attention to detail.

Your task is to translate natural language questions into syntactically correct sqlite queries in order to answer a user's query. Use the schema provided below.

{db_schema}
Instructions:
- Carefully interpret the user's input question using only the information available in the schema. If the question is ambiguous or lacks sufficient context, state that you don’t have enough information to generate a meaningful query.
- Do not include any tool calls or code wrapping—just the raw SQL.
- Retrieve only the relevant columns necessary to answer the question.
- Never use SELECT *. Be explicit about which columns are needed.
- If the number of desired results isn't specified, limit the output to 5 rows.
- Use ORDER BY on a relevant column when appropriate to surface the most interesting or meaningful results.
- DO NOT make any modifications to the database (i.e., avoid INSERT, UPDATE, DELETE, DROP, etc.).
- Do not inlcude columns or tables that are not relevant to the question or not present in the database.

Use the execute tool get the data. Then, formulate an answer based on that data and return it to the user.
"""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)

sys_msg_db = SystemMessage(content=query_gen_system)
llm_with_db_tools = llm.bind_tools(db_tools)



def query_gen_node(state: State):
    question = state["messages"]
    message = llm_with_db_tools.invoke([sys_msg_db] + question)
    return {"messages": [message]}


max_retries = 3


def db_tool_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    """Initialise a tool node that executes the tools and writes
    all returned keys and values to the state variable.

    TODO: This should be a modular class.
    """
    messages = []
    out = {}
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        state_args = {var: state for var in _get_state_args(tool)}
        observation = tool.invoke({**tool_call['args'], **state_args},
                                    config=config)

        # observation = observation['score']
        print(f"observation: {observation}")
        message_content = observation

        messages.append(ToolMessage(content=message_content,
                                    tool_call_id=tool_call['id']))
        # out = {**out, **observation}
        out = {**out, "observation": observation}


    return {'messages': messages, **out}




def tool_call(state: State) -> Literal["db_tool_node", END]: # type: ignore
    messages = state["messages"]
    
    last_message = messages[-1]
    if last_message.tool_calls:
        return "db_tool_node"
    return END


workflow = StateGraph(State)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("db_tool_node", db_tool_node)


workflow.add_edge(START, "query_gen")
# workflow.add_edge("query_gen", END)
workflow.add_conditional_edges(
    "query_gen",
    tool_call,
    # tools_condition
)
workflow.add_edge("db_tool_node", "query_gen")



agent = workflow.compile(checkpointer=MemorySaver())

