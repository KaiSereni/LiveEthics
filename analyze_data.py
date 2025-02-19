import requests
import json
from urllib.parse import quote
from traceback import print_exc as tb
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import time

def safe_print_error(msg, e=None):
    print(f"[Error] {msg}")
    if e:
        print(f"Details: {e}")

# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95, # high, 0-1
  "top_k": 1, # low, 0-50
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

issues = {
    "DEI_L": "DEI in leadership",
    "DEI_H": "DEI in hiring",
    "QUEER": "LGBTQ support",
    "BIPOC": "BIPOC support",
    "PAY": "Fair wages",
    "ENV": "Environmental impact",
    "CHARITY": "Charitable causes",
    "POLI": "Progressive political engagement"
}

function_declarations = []
for issue_id in issues.keys():
    function_declarations.append(genai.protos.FunctionDeclaration(
      name=f"{issue_id}_INDEX",
      description=(
          "Given the article(s) in the prompt, indicate how strongly the article(s) relate to "
          f"{issues[issue_id]} in regard to the company defined in the prompt as COMPANY NAME. "
          "This weight should be a value from 0-100, with 0 meaning it doesn't mention that issue at all, "
          "and 100 means that issue as it relates to the company is the only thing the article(s) talk about. "
          "Then, assign a score for that company from 1-100 in that category given the content of the article, "
          "where 50 means a net-neutral impact. If the significance weight is 0, don't include the score."
      ),
      parameters=content.Schema(
        type=content.Type.OBJECT,
        enum=[],
        required=["significance"],
        properties={
          "significance": content.Schema(
            type=content.Type.NUMBER,
          ),
          "score": content.Schema(
            type=content.Type.NUMBER,
          ),
        },
      ),
    ))

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  tools=[
    genai.protos.Tool(
      function_declarations=function_declarations
    ),
  ],
  tool_config={'function_calling_config': 'ANY'},
)

def ask_about_article(input_text: str):
    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(input_text)
    except Exception as e:
        safe_print_error("Chat session failed", e)
        return {}
    output = {}
    try:
        for part in response.parts:
            if fn := part.function_call:
                output[fn.name.replace("_INDEX", "")] = [
                    fn.args.get("significance", 0),
                    fn.args.get("score", 0.0)
                ]
    except Exception as e:
        safe_print_error("Error processing chat response", e)
    return output

def extract_text_from_html(html_string):
    try:
        soup = BeautifulSoup(html_string, "html.parser")
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        safe_print_error("Error parsing HTML", e)
        return ""

def load_api_keys():
    try:
        with open("keys.json", "r") as f:
            return json.load(f)
    except Exception as e:
        safe_print_error("Failed loading API keys", e)
        return {}

def data_fmp(symbol: str) -> dict:
    api_keys = load_api_keys()
    if "financialmodelingprep" not in api_keys:
        safe_print_error("Financial Modeling Prep API key not found")
        return {}
    key = api_keys["financialmodelingprep"]
    url = f"https://financialmodelingprep.com/stable/esg-disclosures?symbol={symbol}&apikey={key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        if not json_data:
            safe_print_error("No data received from FMP")
            return {}
        data = json_data[0]
        output = {
            "ENV": [{"score": data.get("environmentalScore", 0) / 100, "weight": 0.8}],
            "PAY": [{"score": data.get("socialScore", 0) / 100, "weight": 0.5}]
        }
        return output
    except Exception as e:
        safe_print_error("Error fetching data from Financial Modeling Prep", e)
        return {}

def data_google(company_name: str):
    api_keys = load_api_keys()
    if "google" not in api_keys:
        safe_print_error("Google API key not found")
        return {}
    key = api_keys["google"]
    base_url = "https://www.googleapis.com/customsearch/v1?key={key}&cx=c1bd8c831439c48db&q={query}"
    responses = {}
    for symbol, description in issues.items():
        query = quote(f"{company_name} {description}")
        final_url = base_url.format(key=key, query=query)
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
                    article_response.raise_for_status()
                    text_response = extract_text_from_html(article_response.text)
                    if text_response:
                        url_list.append(text_response)
                except Exception as e:
                    tb()
                    safe_print_error("Error fetching article content", e)
        except Exception as e:
            safe_print_error("Google API error", e)
        responses[symbol] = url_list
    return responses

def aggregate_metrics(metrics_list: list[dict]) -> dict:
    total_weight = 0.0
    weighted_sum = 0.0
    for entry in metrics_list:
        for value in entry.values():
            try:
                # Accept both list or string input
                if isinstance(value, list) and len(value) >= 2:
                    weight = float(value[0])
                    score = float(value[1])
                elif isinstance(value, str):
                    weight_str, score_str = value.split(",")
                    weight = float(weight_str.strip())
                    score = float(score_str.strip())
                else:
                    continue
            except Exception as e:
                safe_print_error("Error parsing metric value", e)
                continue
            total_weight += weight
            weighted_sum += weight * score
    aggregated_score = weighted_sum / total_weight if total_weight else 0.0
    return {"confidence": total_weight, "score": aggregated_score, "timestamp": time.time()}

def analyze_companies(companies: list[str]):
  all_company_data = {}
  for company in companies:
      print(f"Analyzing {company}...")
      try:
          # Configure keys for GEMINI
          api_keys = load_api_keys()
          if "gemini" not in api_keys:
              safe_print_error("Gemini API key not found")
          else:
              genai.configure(api_key=api_keys["gemini"])
      except Exception as e:
          safe_print_error("Failed to configure Gemini API", e)

      google_data = data_google(company)
      fmp_data = data_fmp("AAPL")
      # print("Google data:", google_data)
      # print("FMP data:", fmp_data)
      all_data = {**google_data, **fmp_data}
      # print("Combined data:", all_data)

      chat_data = []
      for issue_id in issues.keys():
          issue_data = all_data.get(issue_id, [])
          formatted_articles = [f"ARTICLE {i+1}: {article[:20000]}" for i, article in enumerate(issue_data)]
          # Use formatted_articles if needed in the prompt. Otherwise using raw issue_data.
          prompt = f"COMPANY NAME: {company}\nARTICLE(S): {issue_data}"
          chat_data.append(ask_about_article(prompt))

      final_scores = aggregate_metrics(chat_data)
      all_company_data[company] = final_scores

  return all_company_data

if __name__ == "__main__":
  companies = [
      "Apple",
      "Google",
      "Microsoft",
      "Meta"
  ]

  final_data = analyze_companies(companies)
  print(final_data)