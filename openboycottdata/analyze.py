import requests, json, time, datetime, re
from urllib.parse import quote
from traceback import print_exc as tb
from bs4 import BeautifulSoup
from google import genai
from google.genai import errors, types

issues = {
    "DEI_L": "DEI in leadership",
    "DEI_H": "DEI in hiring",
    "QUEER": "LGBTQ support",
    "BIPOC": "BIPOC support",
    "PAY": "Fair wages",
    "ENV": "Environmental impact",
    "CHARITY": "Charitable donations",
    "POLI": "Progressive political engagement"
}

model_id = "gemini-2.0-flash"

issues_funcs: list[types.FunctionDeclaration] = []

def string_standard_formatting(string: str):
    string = string.lower().strip()
    string = re.sub(r'[^a-z0-9]', '', string)
    return string

def wait_until_4am(): # waits until the daily API limit resets
    """Waits until 4:00 AM local time."""
    print("Waiting until the Google Search API resets so that requests can be made for free")
    while True:
        now = datetime.datetime.now()
        target_time = now.replace(hour=4, minute=0, second=0, microsecond=0)

        if now >= target_time:
            # If it's already past 4 AM today, wait until 4 AM tomorrow
            if now.hour >= 4:
              target_time += datetime.timedelta(days=1)
            remaining_time = (target_time - now).total_seconds()
            if remaining_time > 0:
                time.sleep(remaining_time)
            break
        else:
            # If it's before 4 AM, calculate the remaining time and sleep
            remaining_time = (target_time - now).total_seconds()
            if remaining_time > 0:
                time.sleep(remaining_time)
            break

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
If you couldn't find any information about the company, set the score and weight to 0. \
"""
    research_scoring_tool_funcs.append(modified_function)

research_competition_info_funcs = [
    types.FunctionDeclaration(
        name="list_competition",
        description=""" \
List 1-20 of the specified company's most valuable products, services, or properties. 
For each product, list whether it's commonly available online, and whether it's commonly available in-person. \
For example, Alphabet Co's search engine Google would be their most valuable property, which is available \
online but not in-person. Apple's product iPhone would be their most valuable property, which is available for purchase both \
online and in-person. Ebay's product Ebay.com would be their most valuable property, which is available online. \
Exxon Mobil's most valuable product is their upstream oil operations, which is available in-person. \
Lockheed Martin's most valuable product would be the F-35 Jet, which is available neither online nor in-person. \
Additionally, for each product, name 1-10 competitor products in order of similarity. For example, Google's competitor \
products would be Yahoo Search, DuckDuckGo, Bing, and Yandex. \
Write the company's security name, without any class specifications and without any corporate suffixes such as LLC or Inc. \
""",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "products": types.Schema(
                    type="ARRAY",
                    items=types.Schema(
                        type="OBJECT",
                        required=["product_name"],
                        properties={
                            "product_name": types.Schema(type="STRING"),
                            "competitor_products": types.Schema(
                                type="ARRAY",
                                description="List of the most similar alternative products",
                                items=types.Schema(
                                    type="OBJECT",
                                    description="Product name and the company that owns it",
                                    required=["product_name", "parent_company"],
                                    properties={
                                        "product_name": types.Schema(type="STRING"),
                                        "parent_company": types.Schema(type="STRING"),
                                    },
                                ),
                            ),
                            "availability": types.Schema(
                                type="OBJECT",
                                description="Where this product is available",
                                properties={
                                    "online": types.Schema(type="BOOLEAN"),
                                    "in_person": types.Schema(type="BOOLEAN"),
                                },
                            ),
                        },
                    ),
                ),
            },
        ),
    )
]

research_competition_info_tool = types.Tool(function_declarations=research_competition_info_funcs, google_search=types.GoogleSearch())
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

def data_fmp(symbol: str, fmp_key: str, test_mode=False) -> dict:
    print(f"Getting FMP data for {symbol}...")
    if test_mode:
        return get_test_fmp_data()
    url = f"https://financialmodelingprep.com/stable/esg-disclosures?symbol={symbol}&apikey={fmp_key}"
    try:
        response = requests.get(url, timeout=30)
        json_data = response.json()
        data = json_data[0]
        output = {
            "ENV": [100, data.get("environmentalScore", 0)],
            "PAY": [50, data.get("socialScore", 0)]
        }
        return output
    except Exception as e:
        return {}

def data_google(company_name: str, google_key: str, gemini_client: genai.Client, test_mode=False) -> dict[str, list[float, float]]:
    print(f"Googling {company_name}...")
    if test_mode:
        return get_test_google_data(company_name)
    
    base_googapi_url = "https://www.googleapis.com/customsearch/v1?key={key}&cx=c1bd8c831439c48db&q={query}"
    responses = {}
    for issue_id, description in issues.items():
        start_time = time.time()
        query = quote(f"{company_name} {description}")
        final_googapi_url = base_googapi_url.format(key=google_key, query=query)
        article_content_list = []
        link_list = []
        max_retries = 2
        for _ in range(max_retries):
            try:
                r = requests.get(final_googapi_url, timeout=30)
            except requests.exceptions.ReadTimeout:
                continue
            except Exception as e:
                tb()
                continue
            if not r.ok:
                print(f"Request to Google API failed: {r.status_code}")
                continue
            result = r.json()
            if not 'totalResults' in result:
                print(f"API unavailable: {result}")
                wait_until_4am()
            if result['totalResults'] == '0':
                print(f"No results for {company_name} {description}")
                break
            result_items = result["items"]
            print(f"Found {result_items.__len__()} Google sources for {company_name}")
            for item in result_items:
                link = item.get("link")
                if not link:
                    continue
                link_list.append(link)
                try:
                    article_response = requests.get(link, timeout=100)
                    assert article_response.ok
                except:
                    print(f"Couldn't get article for {description}")
                    break
                text_response = extract_text_from_html(article_response.text)
                if text_response:
                    article_content_list.append(text_response)
            break
                    
        elapsed = time.time() - start_time
        if elapsed < 1:
            time.sleep(1 - elapsed)
        responses[issue_id] = article_content_list
    
    datasets = []
    for issue_id, articles in responses.items():
        if not articles:
            continue
        formatted_articles = [f"ARTICLE {i+1}: {article}" for i, article in enumerate(articles)]
        prompt = f"COMPANY NAME: {company_name}\nARTICLE(S): {' '.join(formatted_articles)}"
        response = ask_about_article(prompt, gemini_client)
        datasets.append(response)
    
    final_metrics = aggregate_metrics(datasets)

    return {
        "data": final_metrics,
        "sources": link_list
    }

def data_grounded_gemini(company_name: str, gemini_client: genai.Client, test_mode=False) -> dict[str, list[float, float]]:
    print(f"Getting Gemini data for {company_name}...")
    if test_mode:
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

def ask_compeditors(company_name: str, gemini_client: genai.Client, test_mode = False) -> list:
    print(f"Getting competitors for {company_name}...")
    if test_mode:
        return get_test_competitors(company_name)
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            prompt = (
                f"""\
COMPANY NAME: {company_name}
List information about the competition for this company's most valuable products or services and \
compile any data you find in the list_competition function. This function must be called exactly once."""
            )
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[research_competition_info_tool],
                    tool_config=types.FunctionCallingConfig(mode="any"),
                    temperature=0,
                    top_k=1,
                    top_p=0.1
                )
            )
            if response.function_calls:
                compeditors = response.function_calls[0].model_dump()
            else:
                compeditors = []
            return compeditors
            
        except errors.ClientError as e:
            if e.code == 429:  # Resource exhausted
                print(e.message)
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(20)
                    continue
            raise  # Re-raise if not 429 or final attempt
        except Exception as e:
            tb()
            return []
    
    return []  # Fallback if all retries failed

def analyze_companies(companies: list[str], keys: dict[str, str], test_mode=False):
    if "vertexai_project_name" in keys.keys():
        gemini_client = genai.Client(
            vertexai=True,
            project=keys["vertexai_project_name"],
            location="us-central1"
        )
    else:
        gemini_client = None

    all_company_data = {}
    for company in companies:
        print(f"Analyzing {company}...")

        # Get Google search data
        if "google" in keys.keys():
            google_key = keys["google"]
        else:
            google_key = ""

        google_data = data_google(company, google_key, gemini_client, test_mode=test_mode)
        
        # Get FMP data
        if "financialmodelingprep" in keys.keys():
            fmp_key = keys["financialmodelingprep"]
        else:
            fmp_key = ""

        fmp_data = data_fmp(company, fmp_key, test_mode=test_mode)

        # Get Gemini grounded data
        gemini_response = data_grounded_gemini(company, gemini_client, test_mode=test_mode)

        # Aggregate metrics
        metrics = aggregate_metrics([google_data, fmp_data, gemini_response])
        
        # Get competitors
        competitors = ask_compeditors(company, gemini_client, test_mode=test_mode)
        
        # Store results
        if metrics:
            all_company_data[string_standard_formatting(company)] = {
                "metrics": metrics,
                "full_name": company,
                "competitors": competitors,
                "date": int(time.time())
            }

    return all_company_data

if __name__ == "__main__":
    TEST_MODE = False

    if TEST_MODE:
        print("[TEST MODE ENABLED] Using mock data for API calls")
    
    companies = [
        "Apple",
        # "Tesla",
        # "Temu"
    ]

    with open("keys.json", "r") as f:
        keys = json.load(f)

    final_data = analyze_companies(companies, keys, test_mode=TEST_MODE)
    print(final_data)

    if not TEST_MODE:
        try:
            with open("output.json", "r") as f:
                previous_data = json.load(f)
                for company, obj_data in final_data.items():
                    previous_data[company] = obj_data
        except:
            previous_data = {}
            
        with open("output.json", "w") as f:
            json.dump(previous_data, f, indent=2)
