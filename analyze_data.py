import requests
import json
from urllib.parse import quote
from traceback import print_exc as tb
from bs4 import BeautifulSoup
from google import genai
from google.genai.errors import ClientError
from google.genai import types
from google.genai.types import Part
import time
import re
from google.genai.types import HttpOptions

issues = {
    "DEI_L": "DEI in leadership",
    "DEI_H": "DEI in hiring",
    "QUEER": "LGBTQ support",
    "BIPOC": "BIPOC support",
    "PAY": "Fair wages",
    "ENV": "Low environmental impact",
    "CHARITY": "Charitable donations and support",
    "POLI": "Progressive or Democratic political engagement"
}

TEST_MODE = False  # Set to True to use mock data for API calls
model_id = "gemini-2.0-flash"

issues_funcs: list[types.FunctionDeclaration] = []

for issue_id, issue_desc in issues.items():
    issues_funcs.append(types.FunctionDeclaration(
      name=f"{issue_id}_INDEX",
      description=(
          f"""Given the article(s) in the prompt, indicate how strongly the article(s) relate to \
\"{issue_desc}\" in regard to the company defined in the prompt as COMPANY NAME. \
This weight should be a value from 0-100, with 0 meaning it doesn't mention that issue in regards to the company at all, \
and 100 means that issue as it relates to the company is the only thing the article(s) talk about. \
Then, score the company in the \"{issue_desc}\" category, from 0-100, given the content of the article, \
where 50 means a net-neutral impact, 100 means that the company is a world leader in the category, \
and 0 means they're doing extensive, lasting damage. If the significance weight is 0, don't include the score."""
      ),
      parameters=types.Schema(
        type="OBJECT",
        required=["weight"],
        properties={
          "weight": types.Schema(
            type='NUMBER',
          ),
          "score": types.Schema(
            type='NUMBER',
          ),
        },
      ),
    ))

research_scoring_tool_funcs: list[types.FunctionDeclaration] = []
for function in issues_funcs:
    modified_function = function.model_copy()
    this_issue_id = modified_function.name.replace("_INDEX", "")
    modified_function.description = f"""\
Regarding "{issues[this_issue_id]}", research score the company defined in the prompt, from 0-100, \
where 50 means a net-neutral impact, 100 means \
that the company is a world leader in the category, \
and 0 means they're doing extensive, lasting damage. \
Also assign a "weight" from 0-100 based on your confidence in this score. \
If you found 10+ sources about the company regarding that issue, set the weight to 100.
If you couldn't find any information about the company, set the score and weight to 0, \
"""
    research_scoring_tool_funcs.append(modified_function)

issues_significance_tool = types.Tool(function_declarations=issues_funcs)
research_and_scoring_tool = types.Tool(google_search=types.GoogleSearch(), function_declarations=issues_funcs)
grounding_tool = types.Tool(google_search=types.GoogleSearch())
research_scoring_tool = types.Tool(function_declarations=research_scoring_tool_funcs)

def ask_about_article(input_text: str, gemini_client: genai.Client):
    try:
        response = gemini_client.models.generate_content(
            model=model_id,
            contents=input_text,
            config=types.GenerateContentConfig(
                tools=[issues_significance_tool],
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )
    except Exception as e:
        tb()
        return {}
    response_parts = response.candidates[0].content.parts
    output = {}
    for part in response_parts:
        if "function_call" in part.__dict__.keys():
            if "score" in part.function_call.args.keys():
                output[part.function_call.name.replace('_INDEX', '')] = [
                    part.function_call.args["weight"],
                    part.function_call.args["score"]
                ]
            else:
                output[part.function_call.name.replace('_INDEX', '')] = [0, 0]
    if not output:
        print("No output found")
    return output

def extract_text_from_html(html_string):
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        tb()
        return ""

def get_test_fmp_data() -> dict:
    return {
        "ENV": [80, 75],
        "PAY": [50, 65]
    }

def get_test_google_data(company_name: str) -> dict:
    if company_name:
        return {
            "DEI_L": [50, 20],
            "DEI_H": [60, 30],
            "QUEER": [70, 40],
            "BIPOC": [80, 50],
            "PAY": [90, 60]
        }
    return {}

def get_test_gemini_response(company_name: str) -> dict:
    if company_name:
        return {
            "DEI_L": [50, 75],
            "DEI_H": [50, 80],
            "QUEER": [50, 70],
            "BIPOC": [50, 65],
            "PAY": [50, 60],
            "ENV": [50, 85]
        }
    return {}

def get_test_competitors(company_name: str) -> list:
    test_competitors = {
        "Apple": ["Samsung", "Microsoft", "Google"],
        "Google": ["Microsoft", "Apple", "Amazon"],
        "Meta": ["Twitter", "TikTok", "LinkedIn"]
    }
    return test_competitors.get(company_name, ["Competitor 1", "Competitor 2", "Competitor 3"])

def aggregate_metrics(metrics_list: list[dict[str, list[float, float]]]) -> dict[str, list[float, float]]:
    aggregated_metrics = {}
    
    # Combine all metrics into a single structure
    combined_metrics: dict = {}
    for metrics in metrics_list:
        for issue_id, data in metrics.items():
            if issue_id not in combined_metrics:
                combined_metrics[issue_id] = []
                
            # Ensure data is in correct format [weight, score]
            if isinstance(data, list) and len(data) == 2:
                try:
                    weight, score = float(data[0]), float(data[1])
                    combined_metrics[issue_id].append([weight, score])
                except (ValueError, TypeError):
                    continue
    
    # Calculate weighted averages for each issue
    for issue_id, data_points in combined_metrics.items():
        if not data_points:
            continue
        
        try:
            total_weight = sum(point[0] for point in data_points)
            if total_weight <= 0:
                continue
                
            weighted_sum = sum(point[0] * point[1] for point in data_points)
            final_score = weighted_sum / total_weight
            
            aggregated_metrics[issue_id] = [
                round(total_weight, 3),
                round(final_score, 3)
            ]
        except (IndexError, TypeError):
            continue
    
    return aggregated_metrics

def data_fmp(symbol: str, fmp_key: str) -> dict:
    print(f"Getting FMP data for {symbol}...")
    if TEST_MODE:
        return get_test_fmp_data()
    url = f"https://financialmodelingprep.com/stable/esg-disclosures?symbol={symbol}&apikey={fmp_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        if not json_data:
            tb()
            return {}
        data = json_data[0]
        output = {
            "ENV": [100, data.get("environmentalScore", 0)],
            "PAY": [50, data.get("socialScore", 0)]
        }
        return output
    except Exception as e:
        tb()
        return {}

def data_google(company_name: str, google_key: str, gemini_client: genai.Client) -> dict[str, list[float, float]]:
    print(f"Googling {company_name}...")
    if TEST_MODE:
        return get_test_google_data(company_name)
    
    base_url = "https://www.googleapis.com/customsearch/v1?key={key}&cx=c1bd8c831439c48db&q={query}"
    responses = {}
    for issue_id, description in issues.items():
        start_time = time.time()
        query = quote(f"{company_name} {description}")
        final_url = base_url.format(key=google_key, query=query)
        url_list = []
        try:
            r = requests.get(final_url, timeout=10)
            r.raise_for_status()
            result = r.json()
            result_items = result.get("items", [])
            for item in result_items:
                try:
                    link = item.get("link")
                    if not link:
                        continue
                    article_response = requests.get(link, timeout=10)
                    if not article_response.ok:
                        continue
                    text_response = extract_text_from_html(article_response.text)
                    if text_response:
                        url_list.append(text_response)
                except Exception as e:
                    tb()
        except Exception as e:
            tb()
        elapsed = time.time() - start_time
        if elapsed < 1:
            time.sleep(1 - elapsed)
        responses[issue_id] = url_list
    
    datasets = []
    for issue_id, articles in responses.items():
        if not articles:
            continue
        formatted_articles = [f"ARTICLE {i+1}: {article}" for i, article in enumerate(articles)]
        prompt = f"COMPANY NAME: {company_name}\nARTICLE(S): {' '.join(formatted_articles)}"
        response = ask_about_article(prompt, gemini_client)
        datasets.append(response)
    
    print(f"Google datasets: {datasets}")
    r = aggregate_metrics(datasets)
    return r

def data_grounded_gemini(company_name: str, gemini_client: genai.Client) -> dict[str, list[float, float]]:
    print(f"Getting Gemini data for {company_name}...")
    if TEST_MODE:
        return get_test_gemini_response(company_name)
    
    categoriesList = ""
    for id, desc in issues.items():
        categoriesList += f'"{id}": "{desc}", '
    try:
        response = gemini_client.models.generate_content(
            model=model_id,
            contents=f"""Research and score the company "{company_name}" in all the \
specified categories you can find information. Then, return your confidence and score for each category in the functions. \
categories:
{categoriesList}""",
            config=types.GenerateContentConfig(
                tools=[research_scoring_tool],
                temperature=0,
                top_k=1,
                top_p=0.1
            )
        )  
        final_output = {}
        response_parts = response.candidates[0].content.parts
        for part in response_parts:
            if "function_call" in part.__dict__.keys():
                try:
                    function_name = part.function_call.name.replace('_INDEX', '')
                    if "score" in part.function_call.args:
                        final_output[function_name] = [
                            float(part.function_call.args["weight"]),
                            float(part.function_call.args["score"])
                        ]
                    else:
                        final_output[function_name] = [0.0, 0.0]
                except (KeyError, ValueError) as e:
                    print(f"Warning: Error processing function response: {e}")
                    continue
        return final_output
    except Exception as e:
        print(f"Error in data_grounded_gemini: {str(e)}")
        tb()
        return {}

def ask_compeditors(company_name: str, gemini_client: genai.Client) -> list:
    print(f"Getting competitors for {company_name}...")
    if TEST_MODE:
        return get_test_competitors(company_name)
    
    max_retries = 5
    base_delay = 3
    
    for attempt in range(max_retries):
        try:
            prompt = (
                f"COMPANY NAME: {company_name}\n"
                "List 1-10 major competitors of this company, which are worth more than approximately $5M in market cap. \
                If you don't have actual data, estimate. For example, McDonald's competitors are Burger King, Wendy's, Chick-fil-A. "
                "Return the answer as a comma-separated list, wrapped in single backticks."
            )
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0,
                    top_k=1,
                    top_p=0.1
                )
            )
            compeditors_text = response.text
            matches = re.search(r'`(.*?)`', response.text, re.DOTALL)
            if not matches:
                return []
            compeditors_text: str = matches.group(1) 
            compeditors = [c.strip().replace("\n", '') for c in compeditors_text.split(",") if c.strip().replace("\n", '')]
            return compeditors
            
        except ClientError as e:
            if e.code == 429:  # Resource exhausted
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = base_delay + (attempt * 5)
                    print(f"Resource exhausted. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
            raise  # Re-raise if not 429 or final attempt
        except Exception as e:
            tb()
            return []
    
    return []  # Fallback if all retries failed

def analyze_companies(companies: list[str], keys: dict[str, str]):
    gemini_client = genai.Client(
        vertexai=True,
        project=keys["vertexai_project_name"],
        location="us-central1"
    )

    all_company_data = {}
    for company in companies:
        print(f"Analyzing {company}...")

        # Get Google search data
        google_data = data_google(company, keys["google"], gemini_client)
        print(f"GOOGLE DATA: {google_data}")
        
        # Get FMP data
        fmp_data = data_fmp(company, keys["financialmodelingprep"])
        print(f"FMP DATA: {fmp_data}")

        # Get Gemini grounded data
        gemini_response = data_grounded_gemini(company, gemini_client)
        print(f"GEMINI DATA: {gemini_response}")

        # Aggregate metrics
        metrics = aggregate_metrics([google_data, fmp_data, gemini_response])
        
        # Get competitors
        competitors = ask_compeditors(company, gemini_client)
        
        # Store results
        if metrics:
            all_company_data[company] = {
                "metrics": metrics,
                "competitors": competitors,
                "date": int(time.time())
            }

    return all_company_data

if __name__ == "__main__":
    if TEST_MODE:
        print("[TEST MODE ENABLED] Using mock data for API calls")
    
    companies = [
        "Apple",
        "Tesla",
        "Temu"
    ]

    with open("keys.json", "r") as f:
        keys = json.load(f)

    final_data = analyze_companies(companies, keys)
    print(final_data)

    try:
        with open("output.json", "r") as f:
            previous_data = json.load(f)
            for company, obj_data in final_data.items():
                previous_data[company] = obj_data
    except:
        previous_data = {}
        
    with open("output.json", "w") as f:
        json.dump(previous_data, f, indent=2)
