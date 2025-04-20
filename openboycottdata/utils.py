import re
import time
import datetime
from traceback import print_exc as tb
from bs4 import BeautifulSoup
from .config import GOOGLE_SEARCH_DAILY_LIMIT_RESET_HOUR

def string_standard_formatting(string: str) -> str:
    """Standardizes a string by converting to lowercase, stripping whitespace,
    and removing non-alphanumeric characters."""
    string = string.lower().strip()
    string = re.sub(r'[^a-z0-9]', '', string)
    return string

def wait_until_4am():
    """Waits until 4:00 AM local time (or the next day if it's already past 4 AM)."""
    print("Waiting until the Google Search API daily limit resets...")
    now = datetime.datetime.now()
    reset_hour = GOOGLE_SEARCH_DAILY_LIMIT_RESET_HOUR
    target_time = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)

    if now >= target_time:
        # If it's already past reset time today, wait until reset time tomorrow
        target_time += datetime.timedelta(days=1)

    remaining_time = (target_time - now).total_seconds()
    if remaining_time > 0:
        print(f"Sleeping for {remaining_time:.0f} seconds until {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(remaining_time)
    print("Resuming operations.")


def extract_text_from_html(html_string: str) -> str:
    """Extracts plain text from an HTML string."""
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        # Use newline as separator and strip whitespace from each line
        text = soup.get_text(separator='\n', strip=True)
        # Further clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text
    except Exception:
        tb()
        return ""

def sum_weights(data: dict[str, list[float]]) -> float:
    """Calculates the sum of weights from a metrics dictionary."""
    if not data:
        return 0.0
    return sum(metric_data[0] for metric_data in data.values() if metric_data and len(metric_data) > 0)

def empty_function_add_data(data: dict):
    """Placeholder function for adding data incrementally."""
    print(f"Processing data for: {list(data.keys())}")
    pass

def empty_function_skip_company(company_std_name: str) -> bool:
    """Placeholder function to determine if a company should be skipped."""
    return False