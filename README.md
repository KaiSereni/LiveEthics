# OpenBoycott

OpenBoycott is a Python project that analyzes companies (eg. Apple, Google, Microsoft, Meta) by evaluating how various issues — such as environmental impact, fair wages, and diversity & inclusion — are mentioned in related articles. The project leverages multiple APIs and AI tools including Google Custom Search, Financial Modeling Prep, and Google's Generative AI (Gemini). All APIs are completely free and only require you to make an account.

## Features

- **Issue Analysis:** Assess companies on topics such as DEI, LGBTQ support, environmental impact, fair wages, and more.
- **Data Aggregation:** Fetch article content via a **filtered Google Custom Search** (to include only credible, unbiased sources) and ESG scores via Financial Modeling Prep.
- **AI Scoring:** Use Google's Generative AI to interpret article content and yield numerical scores and weights.
- **Simple Extensibility:** Configure new issues or companies as needed.

## Setup

1. **Install Dependencies**

   Install the required packages using:
   
   ```sh
   pip install -r requirements.txt
   ```
2. **API Keys**

Create or update the keys.json file with your API keys. The file should have keys for:

- [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/company-esg-risk-ratings-api)
- [Google search API](https://developers.google.com/custom-search/v1/overview) *note: the free version limits you to 100 queries per day
- [Google Gemini API](https://aistudio.google.com/prompts)

The file must follow this JSON structure:
```json
{
    "financialmodelingprep": "YOUR_FMP_API_KEY",
    "google": "YOUR_GOOGLE_API_KEY",
    "gemini": "YOUR_GEMINI_API_KEY"
}
```
## How It Works
1. **Model Configuration**
    - The project sets up a GenerativeModel using Google's Gemeni AI. 
    - It configures function declarations for various issues based on a predetermined list.

2. **Data Collection**
    - data_google: Searches for articles related to each issue using Google Custom Search [test.py](test.py).
    - data_fmp: Fetches ESG scores from Financial Modeling Prep using the provided symbol (test.py#L44).
    - Processing:
    The ask_about_article function sends the article content to the AI model to get scores and significance values. These values are then aggregated to generate a final score per company (test.py#L85).

3. **Execution**
    - Running the test.py script will perform analysis for each company, combining data from both APIs, and output the scores to the terminal.

4. **Running the Project**
    - To execute the analysis, run `python test.py`.
    Watch the console output for analysis progress and final scores.

### For additional details or modifications, feel free to update this README as needed.