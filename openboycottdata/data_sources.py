import requests
import time
from urllib.parse import quote
from traceback import print_exc as tb
from google import genai

from .config import (
    FMP_API_KEY, GOOGLE_API_KEY, ISSUES, BASE_GOOGAPI_URL, FMP_ESG_URL,
    REQUEST_TIMEOUT, ARTICLE_REQUEST_TIMEOUT, GOOGLE_SEARCH_REQUEST_COOLDOWN
)
from .utils import wait_until_4am, extract_text_from_html
from .gemini_utils import ask_about_article, get_gemini_client # Need ask_about_article here

# Initialize Gemini client once if needed by data_google
# This assumes data_google might need the client. If not, remove this.
# gemini_client_for_google = get_gemini_client()

def data_fmp(symbol: str) -> dict:
    """
    Fetches Environmental, Social, and Governance (ESG) scores from Financial Modeling Prep.

    Args:
        symbol: The company stock symbol.

    Returns:
        A dictionary with 'ENV' and 'PAY' scores and weights, or {} on failure.
        Example: {"ENV": [100, 75.0], "PAY": [50, 65.0]} (Weight, Score)
    """
    print(f"Getting FMP ESG data for {symbol}...")
    if not FMP_API_KEY:
        print("Warning: FMP_API_KEY not found. Skipping FMP data fetch.")
        return {}

    url = FMP_ESG_URL.format(symbol=symbol, key=FMP_API_KEY)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()

        if not json_data or not isinstance(json_data, list):
             print(f"Warning: Unexpected FMP response format for {symbol}: {json_data}")
             return {}

        # Assuming the first element contains the relevant data
        data = json_data[0]
        output = {}
        env_score = data.get("environmentalScore")
        social_score = data.get("socialScore")

        # Add scores only if they are valid numbers
        if isinstance(env_score, (int, float)):
            output["ENV"] = [100.0, float(env_score)] # Assign full weight if score exists
        if isinstance(social_score, (int, float)):
             # Using socialScore for 'PAY' as per original logic
            output["PAY"] = [50.0, float(social_score)] # Assign 50 weight as per original logic

        if not output:
            print(f"Warning: No valid ESG scores found in FMP data for {symbol}.")

        return output

    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP data for {symbol}: {e}")
        # Optionally check status code for specific handling, e.g., 404 Not Found
        # if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
        #     print(f"No FMP ESG data found for symbol {symbol}.")
        return {}
    except (IndexError, KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP data for {symbol}: {e}")
        tb()
        return {}
    except Exception as e:
        print(f"Unexpected error during FMP data fetch for {symbol}: {e}")
        tb()
        return {}


def data_google(company_name: str, gemini_client: genai.Client) -> dict:
    """
    Performs Google Custom Searches for articles related to the company and issues,
    analyzes content using Gemini, and returns aggregated scores.

    Args:
        company_name: The name of the company.
        gemini_client: Initialized Gemini client for article analysis.


    Returns:
        A dictionary containing aggregated 'data' (metrics) and 'sources' (links).
        Example: {'data': {'DEI_L': [50.0, 20.0], ...}, 'sources': ['http://...']}
    """
    print(f"Performing Google searches and analysis for {company_name}...")
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not found. Skipping Google search.")
        return {"data": {}, "sources": []}
    if not gemini_client:
         print("Warning: Gemini client not available for Google data analysis. Skipping.")
         return {"data": {}, "sources": []}

    all_links = set()
    issue_articles = {} # Store articles per issue: {issue_id: [content1, content2]}

    for issue_id, description in ISSUES.items():
        start_time = time.time()
        query = quote(f'"{company_name}" "{description}"') # More specific query
        final_googapi_url = BASE_GOOGAPI_URL.format(key=GOOGLE_API_KEY, query=query)
        print(f"Searching for: {company_name} {description}")

        article_content_list = []
        link_list_for_issue = []
        retries = 2

        for attempt in range(retries):
            try:
                r = requests.get(final_googapi_url, timeout=REQUEST_TIMEOUT)

                if r.status_code == 429:
                    print("Google Search API rate limit hit.")
                    wait_until_4am()
                    # Need to retry the request after waiting
                    if attempt < retries - 1:
                         print("Retrying Google search request...")
                         continue # Retry the request in the next loop iteration
                    else:
                         print("Max retries reached after rate limit wait. Skipping issue.")
                         break # Break inner loop if max retries hit

                r.raise_for_status() # Check for other HTTP errors
                result = r.json()

                if "error" in result:
                    print(f"Google API Error for '{description}': {result['error']}")
                    break # Don't retry on API errors other than 429

                if 'items' not in result or not result['items']:
                    print(f"No Google search results found for '{description}'.")
                    break # No results, move to next issue

                result_items = result["items"]
                print(f"Found {len(result_items)} Google sources for '{description}'.")
                failed_articles = 0

                for item in result_items:
                    link = item.get("link")
                    if not link or link in all_links: # Skip duplicates across issues
                        continue

                    link_list_for_issue.append(link)
                    all_links.add(link) # Add to overall set

                    try:
                        # Fetch article content
                        article_response = requests.get(link, timeout=ARTICLE_REQUEST_TIMEOUT, headers={'User-Agent': 'OpenBoycottBot/1.0'})
                        article_response.raise_for_status()
                        # Extract text
                        text_response = extract_text_from_html(article_response.text)
                        if text_response:
                            # Limit text size to avoid overly large Gemini prompts (e.g., first 10k chars)
                            max_chars = 10000
                            article_content_list.append(text_response[:max_chars])
                        else:
                             print(f"Could not extract text from: {link}")
                             failed_articles += 1
                    except requests.exceptions.RequestException as article_err:
                        print(f"Failed to fetch/process article {link}: {article_err}")
                        failed_articles += 1
                    except Exception as proc_err:
                         print(f"Error processing article {link}: {proc_err}")
                         failed_articles += 1


                if failed_articles > 0:
                    print(f"Failed to retrieve content for {failed_articles}/{len(result_items)} articles for '{description}'.")

                # If successful, break the retry loop for this issue
                break

            except requests.exceptions.ReadTimeout:
                 print(f"Google Search API request timed out for '{description}'. Retrying...")
                 if attempt == retries - 1: print("Max retries reached for timeout.")
                 time.sleep(5 * (attempt + 1)) # Exponential backoff for timeout
                 continue # Retry
            except requests.exceptions.RequestException as e:
                print(f"Google Search API request failed for '{description}': {e}")
                # Don't retry most request exceptions unless specific handling is needed
                break
            except Exception as e:
                 print(f"Unexpected error during Google search for '{description}': {e}")
                 tb()
                 break # Don't retry unknown errors

        # Store fetched articles for this issue
        if article_content_list:
            issue_articles[issue_id] = article_content_list

        # Ensure minimum delay between Google API calls
        elapsed = time.time() - start_time
        if elapsed < GOOGLE_SEARCH_REQUEST_COOLDOWN:
            time.sleep(GOOGLE_SEARCH_REQUEST_COOLDOWN - elapsed)

    # --- Article Analysis with Gemini ---
    datasets = []
    print("Analyzing collected articles with Gemini...")
    # Analyze articles grouped by issue for potentially better context
    for issue_id, articles in issue_articles.items():
        if not articles:
            continue

        # Combine articles for the prompt, consider size limits
        # Simple combination for now:
        combined_article_text = "\n\n---\n\n".join(articles)
        # Truncate if necessary, though individual article truncation helps more
        max_prompt_chars = 30000 # Adjust based on model limits
        truncated_text = combined_article_text[:max_prompt_chars]

        prompt = f"COMPANY NAME: {company_name}\nISSUE CONTEXT: {ISSUES[issue_id]}\n\nARTICLE(S) CONTENT:\n{truncated_text}"

        # Call Gemini to analyze the combined text for this issue
        # Note: This sends one Gemini request per issue with articles found
        response = ask_about_article(prompt, gemini_client) # Removed model_id, uses default from gemini_utils
        if response:
            # We might want to adjust the response structure or how it's aggregated later.
            # For now, just append the dictionary returned by ask_about_article.
            # This assumes ask_about_article returns scores for *all* issues based on the input,
            # which might not be the intended use here.
            # Consider modifying ask_about_article or how prompts are structured if analysis
            # should be strictly per-issue based on the search query.
            datasets.append(response)
        else:
             print(f"Gemini analysis failed for articles related to issue: {issue_id}")


    # Aggregation happens outside this function in analyze_companies
    # Return the raw datasets from Gemini analysis and the list of unique source links
    # The structure returned by ask_about_article needs to be handled by aggregate_metrics
    return {
        "datasets": datasets, # List of dictionaries from ask_about_article
        "sources": list(all_links)
    }
