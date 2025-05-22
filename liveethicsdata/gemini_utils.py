import time
import re
from traceback import print_exc as tb
from google import genai
from google.genai import errors, types
from .config import (
    GEMINI_MODEL_ID, VERTEXAI_PROJECT_NAME, ISSUES,
    ISSUES_SIGNIFICANCE_TOOL, RESEARCH_SCORING_TOOL, RESEARCH_COMPETITION_INFO_TOOL,
    MAX_RETRIES, DEFAULT_RETRY_DELAY, RATE_LIMIT_COOLDOWN_BASE, RATE_LIMIT_COOLDOWN_MULTIPLIER
)

def get_gemini_client() -> genai.Client | None:
    """Initializes and returns the Gemini client using Vertex AI."""
    if VERTEXAI_PROJECT_NAME:
        try:
            return genai.Client(
                vertexai=True,
                project=VERTEXAI_PROJECT_NAME,
                location="us-central1" # Or your preferred location
            )
        except Exception as e:
            print(f"Error initializing Vertex AI client: {e}")
            tb()
            raise ValueError("Failed to initialize Gemini client. Check your environment variables and permissions.")
    else:
        print("Warning: VERTEXAI_PROJECT_NAME not found in environment variables. Cannot initialize Gemini client.")
        raise ValueError("VERTEXAI_PROJECT_NAME not set in environment variables.")

def _handle_gemini_api_call(api_call_func, *args, **kwargs):
    """Generic handler for Gemini API calls with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            return api_call_func(*args, **kwargs)
        except errors.ClientError as e:
            if e.code == 429: # Rate limit error
                if attempt == MAX_RETRIES - 1:
                    print(f"FATAL: Gemini API rate limit exceeded on final attempt ({attempt + 1}/{MAX_RETRIES}).")
                    return None
                cooldown = RATE_LIMIT_COOLDOWN_BASE + attempt * RATE_LIMIT_COOLDOWN_MULTIPLIER
                print(f"Gemini API rate limit hit (429). Retrying attempt {attempt + 2}/{MAX_RETRIES} in {cooldown} seconds...")
                time.sleep(cooldown)
                continue
            else: # Other client errors
                print(f"Gemini Client Error (code {e.code}): {e.message}")
                tb()
                return None # Don't retry other client errors for now
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"FATAL: Unhandled exception during Gemini API call on final attempt ({attempt + 1}/{MAX_RETRIES}).")
                tb()
                return None
            print(f"Unhandled exception during Gemini API call. Retrying attempt {attempt + 2}/{MAX_RETRIES} in {DEFAULT_RETRY_DELAY} seconds...")
            tb()
            time.sleep(DEFAULT_RETRY_DELAY)
            continue
    return None # Should not be reached if MAX_RETRIES > 0

def ask_about_article(input_text: str, gemini_client: genai.Client, model_id: str = GEMINI_MODEL_ID) -> dict:
    """
    Analyzes article text using Gemini to score relevance and sentiment for predefined issues.

    Args:
        input_text: The text content of the article(s) prefixed with company name.
        gemini_client: Initialized Gemini client.
        model_id: The Gemini model ID to use.

    Returns:
        A dictionary mapping issue IDs to [weight, score] lists, or {} on failure.
    """
    if not gemini_client: return {}

    def api_call():
        return gemini_client.models.generate_content(
            model=model_id,
            contents=input_text,
            config=types.GenerateContentConfig(
                tools=[ISSUES_SIGNIFICANCE_TOOL],
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )

    response = _handle_gemini_api_call(api_call)
    if not response or not response.candidates or not response.candidates[0].content:
        print("Invalid response structure received from Gemini.")
        return {}

    output = {}
    response_parts = response.candidates[0].content.parts
    if not response_parts:
        print("No response parts found in Gemini response.")
        return {}

    for part in response_parts:
        try:
            if "function_call" in part.model_dump() and "args" in part.function_call.model_dump():
                func_call = part.function_call
                issue_id = func_call.name.replace('_INDEX', '')
                if "score" in func_call.args and "weight" in func_call.args:
                     # Ensure weight and score are numbers, default to 0 if not
                    weight = float(func_call.args.get("weight", 0.0))
                    score = float(func_call.args.get("score", 0.0))
                    output[issue_id] = [weight, score]
                elif "weight" in func_call.args: # Handle cases where only weight might be returned (e.g., weight is 0)
                     weight = float(func_call.args.get("weight", 0.0))
                     output[issue_id] = [weight, 0.0] # Assign default score of 0
                else:
                     output[issue_id] = [0.0, 0.0] # Default if args are missing
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            print(f"Warning: Error processing function call part: {e}. Part: {part}")
            continue # Skip this part and continue with others

    if not output:
        print("No valid function calls found in Gemini response parts.")

    return output


def data_grounded_gemini(company_name: str, gemini_client: genai.Client, model_id: str = GEMINI_MODEL_ID) -> dict[str, list[float]]:
    """
    Uses Gemini's research capabilities to score a company across predefined issues.

    Args:
        company_name: The name of the company to research.
        gemini_client: Initialized Gemini client.
        model_id: The Gemini model ID to use.

    Returns:
        A dictionary mapping issue IDs to [weight, score] lists, or {} on failure.
    """
    print(f"Getting Gemini grounded data for {company_name}...")
    if not gemini_client: return {}

    categories_list_str = ", ".join([f'"{id}": "{desc}"' for id, desc in ISSUES.items()])
    prompt = f"""Research and score the company "{company_name}" in all the \
specified categories for which you can find reliable information. Then, return your confidence (weight) and score for each category using the provided functions. \
Categories:
{{{categories_list_str}}}"""

    def api_call():
        return gemini_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[RESEARCH_SCORING_TOOL],
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )

    response = _handle_gemini_api_call(api_call)
    if not response or not response.candidates or not response.candidates[0].content:
        print("Invalid response structure received from Gemini grounded search.")
        return {}

    final_output = {}
    response_parts = response.candidates[0].content.parts
    if not response_parts:
        print("No response parts found in Gemini grounded search response.")
        return {}

    for part in response_parts:
         try:
            if "function_call" in part.model_dump() and "args" in part.function_call.model_dump():
                func_call = part.function_call
                function_name = func_call.name.replace('_INDEX', '')
                if "score" in func_call.args and "weight" in func_call.args:
                    # Ensure weight and score are numbers, default to 0 if not
                    weight = float(func_call.args.get("weight", 0.0))
                    score = float(func_call.args.get("score", 0.0))
                    final_output[function_name] = [weight, score]
                elif "weight" in func_call.args: # Handle cases where only weight might be returned
                    weight = float(func_call.args.get("weight", 0.0))
                    final_output[function_name] = [weight, 0.0] # Assign default score of 0
                else:
                    final_output[function_name] = [0.0, 0.0] # Default if args are missing
         except (AttributeError, KeyError, ValueError, TypeError) as e:
            print(f"Warning: Error processing function call part in grounded search: {e}. Part: {part}")
            continue # Skip this part

    if not final_output:
        print("No valid function calls found in Gemini grounded search response parts.")

    return final_output


def ask_competitors(company_name: str, gemini_client: genai.Client, model_id: str = GEMINI_MODEL_ID) -> list:
    """
    Asks Gemini to identify competitors and product information for a given company.

    Args:
        company_name: The name of the company.
        gemini_client: Initialized Gemini client.
        model_id: The Gemini model ID to use.

    Returns:
        A list containing competitor data structure as returned by Gemini, or [] on failure.
    """
    print(f"Getting competitors for {company_name}...")
    if not gemini_client: return []

    prompt = (
        f"""\
COMPANY NAME: \"{company_name}\"
List information about the competition for this company's most valuable products or services and \
compile any data you find in the list_competition function. This function must be called exactly once."""
    )

    def api_call():
        return gemini_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[RESEARCH_COMPETITION_INFO_TOOL],
                tool_config=types.FunctionCallingConfig(mode="any"), # Ensure function is called
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )

    response = _handle_gemini_api_call(api_call)

    # Extract arguments from the function call in the response
    try:
        # Check if response and function_calls exist
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             # Find the first part that is a function call
             function_call_part = next((part for part in response.candidates[0].content.parts if part.function_call), None)
             if function_call_part and function_call_part.function_call.name == "list_competition":
                 # Safely access args, defaulting to an empty dict if not present
                 competitors_data = getattr(function_call_part.function_call, 'args', {})
                 # Return the value associated with 'products' key, or empty list if key missing or data is None
                 return competitors_data.get('products', []) if competitors_data else []
    except (AttributeError, IndexError, TypeError) as e:
        print(f"Error extracting competitor data from Gemini response: {e}")
        tb()

    print("Could not extract competitor data from Gemini response.")
    return []


def ask_alt_names(company_name: str, gemini_client: genai.Client, model_id: str = GEMINI_MODEL_ID) -> list[str]:
    """
    Asks Gemini to list alternative names for a given company.

    Args:
        company_name: The name of the company.
        gemini_client: Initialized Gemini client.
        model_id: The Gemini model ID to use.

    Returns:
        A list of alternative names, or [] on failure.
    """
    print(f"Getting alternative names for {company_name}...")
    if not gemini_client: return []

    prompt = f"""List as many alternate names by which the company \"{company_name}\" is known as you can, including:
- Common abbreviations (e.g. IBM for International Business Machines)
- Former names (e.g. Google before it became Alphabet)
- Parent/subsidiary relationships (e.g. Instagram being owned by Meta)
- Common misspellings or variations
- Stock ticker symbols
- Versions with or without corporate suffixes (Co, corp, Inc, LLC, etc.)
Do not include generic terms. Do not include the names of any of the company's products or services unless that product or service name was a previous name of the company.
Your response should be ONLY a Python-style list of strings (e.g., ["Name1", "Name2"]). Do not include any other text before or after the list. Remove any commas from within the names themselves.
Example response for Meta Platforms: ["Meta", "Facebook", "Face Book", "Meta Inc"]"""

    def api_call():
        return gemini_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )

    response = _handle_gemini_api_call(api_call)

    try:
        if response and response.text:
            # Use regex to find the list structure more reliably
            match = re.search(r'\[\s*(".*?"|\'.*?\')(?:\s*,\s*(".*?"|\'.*?\'))*\s*\]', response.text)
            if match:
                list_str = match.group(0)
                # Safely evaluate the string list
                alt_names = eval(list_str)
                if isinstance(alt_names, list):
                    # Clean names: strip whitespace and quotes
                    return [str(name).strip().strip('"\'') for name in alt_names]
            else:
                 print(f"Could not parse alt names list from response: {response.text}")

    except (SyntaxError, ValueError, AttributeError, Exception) as e:
        print(f"Error parsing alt names list from Gemini response: {e}")
        print(f"Response text: {getattr(response, 'text', 'N/A')}")
        tb()

    return []