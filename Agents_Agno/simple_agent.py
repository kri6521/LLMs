from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.cohere import Cohere
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile")
)

agent.print_response("Write a 2 sentence poem for the love between dosa and samosa")