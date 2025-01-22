import asyncio

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


browser = Browser(
    config=BrowserConfig(
        headless=True,
    )
)


async def main():
    agent = Agent(
        task="Go to Reddit, search for 'browser-use' in the search bar, click on the first post and return the first comment.",
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        browser=browser,
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
