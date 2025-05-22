import os
from dotenv import load_dotenv
from google.genai import types

load_dotenv() # Load environment variables from .env file

# API Keys
FMP_API_KEY = os.getenv("FINANCIALMODELINGPREP_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VERTEXAI_PROJECT_NAME = os.getenv("VERTEXAI_PROJECT_NAME")

# Gemini Model
GEMINI_MODEL_ID = "gemini-1.5-flash"

# Issues definition
ISSUES = {
    "DEI_L": "DEI in leadership",
    "DEI_H": "DEI in hiring",
    "QUEER": "LGBTQ support",
    "BIPOC": "BIPOC support",
    "PAY": "Fair wages",
    "ENV": "Environmental impact",
    "CHARITY": "Charitable donations",
    "POLI": "Progressive political engagement"
}

# --- Gemini Tool Definitions ---

# Tool for scoring based on article content
ISSUES_FUNCS: list[types.FunctionDeclaration] = []
for issue_id, issue_desc in ISSUES.items():
    ISSUES_FUNCS.append(types.FunctionDeclaration(
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
          "weight": types.Schema(type='NUMBER'),
          "score": types.Schema(type='NUMBER'),
        },
      ),
    ))

ISSUES_SIGNIFICANCE_TOOL = types.Tool(function_declarations=ISSUES_FUNCS)

# Tool for Gemini's own research and scoring
RESEARCH_SCORING_TOOL_FUNCS: list[types.FunctionDeclaration] = []
for function in ISSUES_FUNCS:
    modified_function = function.model_copy()
    this_issue_id = modified_function.name.replace("_INDEX", "")
    modified_function.description = f"""\
Regarding "{ISSUES[this_issue_id]}", research and score the company defined in the prompt, from 0-100, \
where 50 means a net-neutral impact, 100 means \
that the company is a world leader in the category, \
and 0 means they're doing extensive, lasting damage. \
Also assign a "weight" from 0-100 based on your confidence in this score. \
If you found 10+ sources about the company regarding that issue, set the weight to 100.
If you couldn't find any information about the company, set the score and weight to 0. \
"""
    RESEARCH_SCORING_TOOL_FUNCS.append(modified_function)

RESEARCH_SCORING_TOOL = types.Tool(function_declarations=RESEARCH_SCORING_TOOL_FUNCS)

# Tool for getting competitor information
RESEARCH_COMPETITION_INFO_FUNCS = [
    types.FunctionDeclaration(
        name="list_competition",
        description=""" \
List 1-20 of the specified company's most valuable products, and/or services.
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
                                        "alt_product_names": types.Schema(
                                            type="ARRAY",
                                            description='Broad list of alternate names commonly used to refer to the product. For example, alternate names for "Google" would include "Google Search", "GoogleSearch", "Search", "Search Engine", "Internet Explorer", "Internet Search", "Search Bar".',
                                            items=types.Schema(type="STRING"),
                                        ),
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
RESEARCH_COMPETITION_INFO_TOOL = types.Tool(function_declarations=RESEARCH_COMPETITION_INFO_FUNCS, google_search=types.GoogleSearch())

# Grounding tool (for general search)
GROUNDING_TOOL = types.Tool(google_search=types.GoogleSearch())

# API URLs
SEARCH_ENGINE_ID = "c1bd8c831439c48db"
BASE_GOOGAPI_URL = "https://www.googleapis.com/customsearch/v1?key={key}&cx=" + SEARCH_ENGINE_ID + "&q={query}"
FMP_ESG_URL = "https://financialmodelingprep.com/stable/esg-disclosures?symbol={symbol}&apikey={key}"

# Other constants
DEFAULT_RETRY_DELAY = 30 # seconds
MAX_RETRIES = 5
RATE_LIMIT_COOLDOWN_BASE = 60 # seconds
RATE_LIMIT_COOLDOWN_MULTIPLIER = 300 # seconds
GOOGLE_SEARCH_DAILY_LIMIT_RESET_HOUR = 4 # 4 AM local time
REQUEST_TIMEOUT = 30 # seconds
ARTICLE_REQUEST_TIMEOUT = 100 # seconds
GOOGLE_SEARCH_REQUEST_COOLDOWN = 1 # second