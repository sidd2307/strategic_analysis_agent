import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool # Import the base class
from langchain_groq import ChatGroq
# from langchain_community.tools import DuckDuckGoSearchRun
# from crewai_tools import DuckDuckGoSearchRunTool
# from crewai_tools import DuckDuckGoSearchTool
from duckduckgo_search import DDGS # Import the raw search engine
import yfinance as yf

import os

# The 'Bulletproof' way to set it for the whole system
os.environ["GROQ_API_KEY"] = "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"

# Also set this for LiteLLM specifically (it's a common quirk)
os.environ["LITELLM_LOGGING"] = "False"

# --- CUSTOM TOOL DEFINITION ---
class MySearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Search the web for the latest news and information."

    # def _run(self, query: str) -> str:
    #     """Execute the search."""
    #     with DDGS() as ddgs:
    #         results = ddgs.text(query, max_results=5)
    #         return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            # Only take 2 results, and only the first 200 characters of each
            results = ddgs.text(query, max_results=2) 
            short_results = []
            for r in results:
                # We only need the headline and a snippet
                short_results.append(f"Title: {r['title']}\nSnippet: {r['body'][:200]}")
            return "\n".join(short_results)


# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Market Intelligence", page_icon="📊", layout="wide")

st.title("🚀 Strategic AI Agent Dashboard")
st.markdown("Developed for Consulting, IB, and Market Intelligence.")

# --- SIDEBAR: CONFIG & API KEYS ---
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    print('===================api key===========', groq_api_key)
    company_name = st.text_input("Target Company (e.g., Apple, NVIDIA):")
    
    st.info("Get a free key at [console.groq.com](https://console.groq.com/)")
    analyze_button = st.button("Generate Intelligence Report", type="primary")

# --- CORE LOGIC ---
if analyze_button:
    if not groq_api_key or not company_name:
        st.error("Please provide both an API key and a Company Name.")
    else:
        try:
            # Initialize Models (Free on Groq)
            # 70B for the deep thinking, 8B for the quick tasks
            llm_70b = ChatGroq(temperature=0, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="groq/llama-3.3-70b-versatile")
            # llm_8b = ChatGroq(temperature=0, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="groq/llama-3.1-8b-instant")
            
            # search_tool = DuckDuckGoSearchTool()
            # Initialize the custom tool
            search_tool = MySearchTool()

            # AGENTS
            # scraper = Agent(
            #     role='Market Researcher',
            #     goal=f'Find latest news and strategic moves for {company_name}',
            #     backstory='Specialist in business intelligence and competitive landscape mapping.',
            #     tools=[search_tool], llm=llm_8b, verbose=True
            # )
            scraper = Agent(
                role='Researcher',
                goal='Find news',
                backstory='Concise assistant.', # That's it. Don't add more.
                tools=[search_tool],
                llm=llm_70b,
                max_iter=1, # FORCE it to only search ONCE
                verbose=True
            )

            analyst = Agent(
                role='Financial Strategist',
                goal=f'Identify 3 Strengths and 3 Weaknesses for {company_name}',
                backstory='Former IB associate who synthesizes financial health from market data.',
                llm=llm_70b, verbose=True
            )

            synthesizer = Agent(
                role='Managing Partner',
                goal='Create a SWOT and Porter’s Five Forces analysis',
                backstory='Consulting veteran skilled at creating boardroom-ready frameworks.',
                llm=llm_70b, verbose=True
            )

            # TASKS
            # t1 = Task(description=f"Summarize major {company_name} news from the last 6 months.", agent=scraper, expected_output="List of top 5 business updates.")
            t1 = Task(
                description=f"Provide a brief summary of the most recent news for {company_name}.",
                expected_output="A list of 3 bullet points with headlines and dates.",
                agent=scraper
            )
            t2 = Task(description=f"Assess {company_name}'s market position and financial stability.", agent=analyst, expected_output="Bullet points of financial pros/cons.")
            t3 = Task(description="Synthesize all data into a SWOT and Porter's Five Forces report.", agent=synthesizer, expected_output="Markdown formatted report.")

            # EXECUTION
            crew = Crew(agents=[scraper, analyst, synthesizer], tasks=[t1, t2, t3], process=Process.sequential, verbose=True, max_rpm=3)

            with st.spinner("Agents are collaborating... This usually takes 30-60 seconds."):
                final_report = crew.kickoff()

            # DISPLAY RESULTS
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Strategic Intelligence Report")
                st.markdown(final_report)
            
            with col2:
                st.subheader("Live Market Context")
                # Basic YFinance Info for extra 'flair'
                ticker = company_name.upper()[:4] # Crude ticker guess
                st.metric("Ticker (Estimated)", ticker)
                st.write("Tip: Use the report on the left for your slide deck or strategy memo.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.write("Enter a company and click 'Generate' to wake up the agents.")