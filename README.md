# liveethics Data Aggregation Algorithm

This is a Python project that analyzes companies (eg. Apple, Google, Microsoft, Meta) by evaluating how various issues — such as environmental impact, fair wages, and diversity & inclusion — are mentioned in related articles. The project leverages multiple APIs and AI tools including Google Custom Search, Financial Modeling Prep, and Google's Generative AI (Gemini). All APIs are completely free at the time of writing and only require you to make an account.

## Features

- **Issue Analysis:** Assess companies on topics such as DEI, LGBTQ support, environmental impact, fair wages, and more.
- **Data Aggregation:** Fetch article content via a **filtered Google Custom Search** (to include only credible, unbiased sources) and ESG scores via Financial Modeling Prep.
- **AI Scoring:** Use Google's Generative AI to interpret article content and yield numerical scores and weights.
- **Simple Extensibility:** Configure new issues or companies as needed.

## Setup

1. **Install Package**

   Install the package using:
   
   ```
   pip install git+https://github.com/KaiSereni/liveethics#egg=liveethicsdata
   ```
2. **API Keys**

Create or update the keys.json file with your API keys. The file should have keys for:

- [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/company-esg-risk-ratings-api)
- [Google search API](https://developers.google.com/custom-search/v1/overview) *note: the free version limits you to 100 queries per day
- [Google Vertex AI](https://console.cloud.google.com/vertex-ai/studio/chat)

The file must follow this JSON structure:
```json
{
    "financialmodelingprep": "YOUR_FMP_API_KEY",
    "google": "YOUR_GOOGLE_API_KEY",
    "vertexai_project_name": "YOUR_VERTEXAI_PROJECT_NAME"
}
```
3. **Vertex AI Auth**

You'll authenticate with Google Vertex AI, and then login before running your script.
```
gcloud auth login
gcloud config set project [PROJECT_NAME]
```
4. **Call the function**

Here's an example:
```py
from liveethicsdata.analyze import analyze_companies

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

```

### Can you code it better? [Branch us](https://github.com/KaiSereni/liveethics/branches) on GitHub!
