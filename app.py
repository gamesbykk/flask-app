from flask import Flask, render_template_string, request
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import markdown2
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize the Language Model
llm = ChatOpenAI(
    model="openrouter/deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7,
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# Custom DuckDuckGo Search Tool
class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

# Define the Agents
market_researcher = Agent(
    role="Stock Market Researcher",
    goal="Identify the top performing stocks and sectors with strong growth potential for the current year",
    backstory="You are a financial analyst with expertise in identifying high-growth stocks across different sectors.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[MyCustomDuckDuckGoTool()]
)

stock_analyst = Agent(
    role="Stock Analyst",
    goal="Analyze and select the top 10 stocks to invest in for the current year",
    backstory="You are a seasoned stock analyst with 10+ years of experience in fundamental and technical analysis.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[MyCustomDuckDuckGoTool()],
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Create a compelling investment recommendation report for the top 10 stocks",
    backstory="You are a professional investment advisor who helps clients make informed decisions.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[MyCustomDuckDuckGoTool()],
)

# Define the Tasks
market_research_task = Task(
    description="Research current stock market trends and identify sectors with strong growth potential for the current year.",
    expected_output="A list of 5-7 promising sectors with explanations of why they have growth potential this year",
    agent=market_researcher
)

stock_selection_task = Task(
    description="Based on the sector research, identify the top 10 stocks to invest in this year.",
    expected_output="A list of 10 stocks with their basic information and investment rationale",
    agent=stock_analyst,
    context=[market_research_task]
)

investment_report_task = Task(
    description="Create a comprehensive investment report presenting the top 10 stocks to invest in this year.",
    expected_output="A well-formatted investment report in markdown with detailed analysis of each recommended stock",
    agent=investment_advisor,
    context=[market_research_task, stock_selection_task]
)

# Create the Crew
investment_crew = Crew(
    agents=[market_researcher, stock_analyst, investment_advisor],
    tasks=[market_research_task, stock_selection_task, investment_report_task],
    verbose=True,
    process=Process.sequential
)

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the last result (for demo purposes)
# In production, you'd want to use a proper database or caching solution
last_result = None
last_updated = None

def generate_recommendations():
    global last_result, last_updated
    try:
        result = investment_crew.kickoff()
        last_result = str(result)
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return last_result
    except Exception as e:
        raise e

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_result, last_updated
    
    if request.method == 'POST' and 'refresh' in request.form:
        try:
            result_str = generate_recommendations()
        except Exception as e:
            return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error</title>
                <style>
                    .error { color: red; }
                </style>
            </head>
            <body>
                <h1>Error Generating Recommendations</h1>
                <p class="error">{{ error }}</p>
                <a href="/">Back to recommendations</a>
            </body>
            </html>
            """, error=str(e))
    else:
        if last_result is None:
            result_str = generate_recommendations()
        else:
            result_str = last_result
    
    # Convert the Markdown result to HTML
    result_html = markdown2.markdown(result_str)
    
    # Render the result with Markdown
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Top 10 Stocks to Invest In This Year</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css">
        <style>
            .markdown-body {
                box-sizing: border-box;
                min-width: 200px;
                max-width: 980px;
                margin: 0 auto;
                padding: 45px;
            }
            .refresh-btn {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 20px 0;
                cursor: pointer;
                border-radius: 4px;
            }
            .refresh-btn:hover {
                background-color: #45a049;
            }
            .timestamp {
                color: #666;
                font-style: italic;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <article class="markdown-body">
            <h1>Top 10 Stocks to Invest In This Year</h1>
            <form method="post">
                <button type="submit" name="refresh" class="refresh-btn">Refresh Recommendations</button>
            </form>
            {{ result_html|safe }}
            <div class="timestamp">Last updated: {{ last_updated }}</div>
            <footer style="margin-top: 2rem; color: #586069; font-size: 0.9rem;">
                <p>Note: Stock recommendations are based on current market analysis and may change.</p>
            </footer>
        </article>
    </body>
    </html>
    """, result_html=result_html, last_updated=last_updated)

if __name__ == '__main__':
    app.run(debug=True, port=2076,host='0.0.0.0')
