"""Top level agent for data agent multi-agents.

-- it get data from database (e.g., BQ, AlloyDB) using NL2SQL
-- then, it use NL2Py to do further data analysis as needed
"""

from google.genai import types

from google.adk import Agent
from google.adk.tools import ToolContext
from google.adk.tools import load_artifacts
from google.adk.tools.agent_tool import AgentTool

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)

FLAT_USE_DS_AGENT_GEN_EXEC = True
DEFAULT_DATABASE = "BigQuery"


from .data_science.ds_agent import root_agent as analytics_agent
from .database.alloydb.db_agent import database_agent as pg_db_agent
from .database.bigquery.db_agent import database_agent as bq_db_agent
from .intent_understanding.agent import intent_agent
from .utils import extract_json_from_model_output

import os
from dotenv import load_dotenv

load_dotenv()

project = os.environ.get("GOOGLE_CLOUD_PROJECT", None)
location = os.environ.get("GOOGLE_CLOUD_LOCATION", None)
ressource_name = os.environ.get("CODE_INTERPRETER_EXTENSION_NAME", None)

assert project is not None, "Please set the GOOGLE_CLOUD_PROJECT environment variable."
assert (
    location is not None
), "Please set the GOOGLE_CLOUD_LOCATION environment variable."
assert (
    ressource_name is not None
), "Please set the CODE_INTERPRETER_EXTENSION_NAME environment variable."

print(f"CODE_INTERPRETER_EXTENSION_NAME {ressource_name}")
###########################################################################
# Data agents implementation with agent as tools.
###########################################################################

instruction_prompt = """

You are an AI assistant answering data-related questions using provided tools. **Never generate answers directly; always USING THE GIVEN TOOLS.**



**Workflow:**

1. **CALL THE INTENT UNDERSTANDING TOOL and understand its return (`call_intent_understanding`):**  This tool classifies the user question and returns a JSON with one of four structures:

    * **Greeting:** Contains a `greeting_message`. Return this message directly.
    * **Use Database:** Contains a `use_database`. Return "Using database XYZ".
    * **Out of Scope:**  Return: "Your question is outside the scope of this database. Please ask a question relevant to this database."
    * **SQL Query Only:** Contains `nl_to_sql_question`. Proceed to Step 2.
    * **SQL and Python Analysis:** Contains `nl_to_sql_question` and `nl_to_python_question`. Proceed to Step 2.

2. **Retrieve Data TOOL (`call_db_agent` - if applicable):**  If Step 1 returned `nl_to_sql_question` is not 'N/A', use this tool with `question: nl_to_sql_question` to query the database Don't give the SQL Query! Express your needs.

3. **Analyze Data TOOL (`call_ds_agent` - if applicable):**  If Step 1 returned `nl_to_python_question` is not 'N/A', use this tool with `question: nl_to_python_question` to perform analysis.

4. **Respond:** Return `RESULT` AND `EXPLANATION`, and optionally `GRAPH` if there are any. Please USE the MARKDOWN format (not JSON) with the following sections:

    * **Results overview:**  "Natural language summary of the data agent findings"

    * **SQL Queries** "SQL queries used to get the results. Leave empty if no SQL queries were used."

    * **Explanation:**  "Step-by-step explanation of how the result was derived.",

    * **Graph:**  (this field is optional): "Filename of any plot generated in Step 3. Leave empty if no plot was generated."

    * **Results:**  "Detailed Results in the format requested by the user."

**Tool Usage Summary:**

* **Greeting/Out of Scope:** `call_intent_understanding` only.
* **SQL Query:** `call_intent_understanding` then `call_db_agent`.
* **SQL & Python Analysis:** `call_intent_understanding`, `call_db_agent`, then `call_ds_agent`.

**Key Reminder:**
 * **ALWAYS START WITH call_intent_understanding!**
 * **Do not fabricate any answers. Rely solely on the provided tools. ALWAYS USE call_intent_understanding FIRST!**
 * **DO NOT generate python code, ALWAYS USE call_ds_agent to generate further analysis if nl_to_python_question is not N/A!**
 * **IF call_ds_agent is called with valid result, JUST SUMMARIZE ALL RESULTS FROM PREVIOUS STEPS USING RESPONE FORMAT!**
 * **Don't write the SQL query by yourself or fix the columns or the tables to use, use the provided tools. **
    """


async def call_intent_understanding(
    question: str,
    tool_context: ToolContext,
):
    """Tool to do intent understanding."""
    agent_tool = AgentTool(agent=intent_agent)
    intent_agent_output_txt = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    try:
        intent_agent_output = extract_json_from_model_output(intent_agent_output_txt)
        tool_context.state["intent_agent_output"] = intent_agent_output
        if "use_database" in intent_agent_output:
            tool_context.state["all_db_settings"]["use_database"] = intent_agent_output[
                "use_database"
            ]

        return intent_agent_output
    except Exception as e:
        return "can't decode intent_agent_output"


async def call_db_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call database (nl2sql) agent."""

    database_agent = bq_db_agent
    try:
        agent_tool = AgentTool(agent=database_agent)
        db_agent_output = await agent_tool.run_async(
            args={"request": question}, tool_context=tool_context
        )
        tool_context.state["db_agent_output"] = db_agent_output
        return db_agent_output
    except Exception as e:
        return "can't process the request"


async def call_ds_agent(
    question: str,
    tool_context: ToolContext,
):
    """Tool to call data science (nl2py) agent."""

    if question == "N/A":
        tool_context.state["ds_agent_output"] = tool_context.state["db_agent_output"]
        return tool_context.state["db_agent_output"]

    if "query_result" in tool_context.state:
        input_data = tool_context.state["query_result"]
    else:
        return "No data available for analysis."

    question_with_data = f"""
  Question to answer: {question}

  Actual data to analyze prevoius quesiton is already in the following:
  {input_data}

  """

    agent_tool = AgentTool(agent=analytics_agent)
    ds_agent_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    tool_context.state["ds_agent_output"] = ds_agent_output
    return ds_agent_output


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="faster_agent",
    instruction=instruction_prompt,
    tools=[
        call_intent_understanding,
        call_db_agent,
        call_ds_agent,
        load_artifacts,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
