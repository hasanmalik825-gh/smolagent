from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import yfinance as yf
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
from typing import Dict, Any

@tool
def get_stock_info(symbol: str) -> Dict[str, Any]:
    """A tool that fetches current stock information for a given stock symbol.
    Args:
        symbol: A string representing the stock ticker symbol (e.g., 'AAPL' for Apple, 'GOOG' for Google).
    Returns:
        Dict[str, Any]: A dictionary containing:
            - Stock: The stock symbol
            - Current Price: Current trading price
            - Today's High: Highest price of the day
            - Today's Low: Lowest price of the day
            - Volume: Trading volume
            - MarketCap: Market capitalization
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        # Get current stock info
        info = ticker.info
        current_price = info.get('currentPrice', 'NA')
        market_cap = info.get('marketCap', 'NA')        
        # Get today's data
        today_data = ticker.history(period='1d')
            
        return {
            "Stock": symbol.upper(),
            "Current Price": current_price,
            "Today's High": today_data['High'].iloc[0] or "NA",
            "Today's Low": today_data['Low'].iloc[0] or "NA",
            "Volume": today_data['Volume'].iloc[0] or "NA",
            "MarketCap": market_cap
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return {"Error": str(e)}  # Return the error in a friendly format

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_stock_info, image_generation_tool, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()