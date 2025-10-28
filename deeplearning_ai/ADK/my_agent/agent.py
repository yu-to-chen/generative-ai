from google.adk.agents import Agent
from datetime import datetime
import pytz

def get_current_time(city: str) -> dict:
    """Return the current local time in the given city."""
    try:
        city_map = {
            "new york": "America/New_York",
            "san francisco": "America/Los_Angeles",
            "london": "Europe/London",
            "paris": "Europe/Paris",
            "tokyo": "Asia/Tokyo",
            "taipei": "Asia/Taipei",
            "singapore": "Asia/Singapore",
            "sydney": "Australia/Sydney",
        }
        tz_name = city_map.get(city.lower(), city)
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
        time_str = now.strftime("%Y-%m-%d %I:%M %p")
        return {"status": "success", "city": city, "time": time_str}
    except Exception as e:
        return {"status": "error", "city": city, "message": str(e)}

"""
root_agent = Agent(
    model="gemini-2.0-flash-live-001",
    name="city_time_agent",
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the get_current_time tool for this purpose.",
    tools=[get_current_time],
)
"""

# Speed it up
root_agent = Agent(
    model="gemini-2.0-flash-live-001",   # fast speech latency
    name="city_time_agent",
    description="Tells the current time in a specified city.",
    instruction="Be brief. Speak in 1–2 sentences.",
    tools=[get_current_time],            # keep only truly needed tools
)

