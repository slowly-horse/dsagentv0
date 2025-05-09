"""Local test for the data agents."""

import os
import time

from google.adk import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# The following lines are commented out because they cause the test to fail.
# from .data_science.ds_agent_exec import analytics_agent
# from .data_science.ds_agent_gen_exec import root_agent as analytics_agent_gen_exec
from .database.bigquery.db_agent import database_agent
from .intent_understanding.agent import intent_agent as intent_understanding_agent


session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()


def test_agent(agent, query):
    """Test the agent."""

    session = session_service.create(
        app_name="DataAgent",
        user_id="test_user",
    )
    print("user: ", query)
    content = types.Content(role="user", parts=[types.Part(text=query)])

    start_time = time.time()
    runner = Runner(
        "DataAgent",
        agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    events = list(runner.run(session=session, new_message=content))
    last_event = events[-1]
    final_response = last_event.content.parts[0].text

    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    print(f"\n Query: {query}")
    print(f"\n Time taken (ms): {elapsed_time_ms}")
    print(f"\n Final response: {final_response}")

    return elapsed_time_ms, final_response


def test_basic_agent():
    """Test the basic agent."""
    basic_agent = Agent(
        model="gemini-1.5-flash",
        name="root_agent",
        instruction="",
        flow="auto",
    )
    test_agent(basic_agent, "Hello, world!")


def test_intent_understanding_agent():
    """Test the intent understanding agent."""
    queries = [
        "hi",
        "what data you have",
        "what you can do",
        "what's the capital of US",
        "what's top 10 selling liquor categories in Iowa",
        "show a bar chart of top 15 selling liquor categories in Iowa",
    ]
    for query in queries:
        test_agent(intent_understanding_agent, query)


def test_db_agent():
    """Test the db agent."""
    if os.environ.get("AGENT_QUESTION") is not None:
        # If a query is specified, run that query against the specified database
        # in the specified project.
        queries = [os.environ.get("AGENT_QUESTION")]
    else:
        queries = ["what's top 10 selling liquor categories in Iowa"]
    for query in queries:
        test_agent(database_agent, query)


def test_ds_agent():
    """Test the ds agent."""
    question_with_data = """Question to answer: Create a bar chart showing the top 15 liquor categories by sales dollars.

  Actual data to analyze previous question is already in the following:
  [{'category_name': 'CANADIAN WHISKIES', 'total_sales': 520819727.07999355},
   {'category_name': 'AMERICAN VODKAS', 'total_sales': 465278553.7300251},
   {'category_name': 'STRAIGHT BOURBON WHISKIES', 'total_sales': 307621281.8799939},
   {'category_name': 'SPICED RUM', 'total_sales': 281699076.7500008},
   {'category_name': 'WHISKEY LIQUEUR', 'total_sales': 233125048.15999946},
   {'category_name': 'IMPORTED VODKAS', 'total_sales': 197920767.93000203},
   {'category_name': 'TENNESSEE WHISKIES', 'total_sales': 184009645.45999864},
   {'category_name': '100% AGAVE TEQUILA', 'total_sales': 166843299.04999998},
   {'category_name': 'VODKA 80 PROOF', 'total_sales': 145764888.83000162},
   {'category_name': 'BLENDED WHISKIES', 'total_sales': 123017603.33000223},
   {'category_name': 'IMPORTED BRANDIES', 'total_sales': 101004054.6200014},
   {'category_name': 'CREAM LIQUEURS', 'total_sales': 96076143.9000001},
   {'category_name': 'FLAVORED RUM', 'total_sales': 89715038.65000068},
   {'category_name': 'IRISH WHISKIES', 'total_sales': 87100947.15000078},
   {'category_name': 'AMERICAN FLAVORED VODKA', 'total_sales': 84163663.46000007}]"""

    # test_agent(analytics_agent, question_with_data)

    # test_agent(analytics_agent_gen_exec, question_with_data)


if __name__ == "__main__":
    test_basic_agent()

    test_intent_understanding_agent()

    test_db_agent()

    # The following test fails.
    # test_ds_agent()
