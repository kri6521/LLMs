from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources"]
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data"]
)

agent_team = Agent(
    team=[web_agent,finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),   # use open ai model for better response
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources","Use tables to display data"]
)

agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
# for analyst recommendations it will use YFinance Tools and for latest news it will use DuckDuckGo Tools