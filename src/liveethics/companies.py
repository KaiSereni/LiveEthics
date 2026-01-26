from dataclasses import dataclass
from google import genai
from google.genai import types as gemtypes
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RatingCategory:
    category_id: str
    category_name: str
    ai_description: str

RATING_CATEGORIES: list[RatingCategory] = [
    RatingCategory(
        category_id='progressive_lobbying',
        category_name='Progressive Lobbying',
        ai_description="The approximate ratio of dollars spent on lobbying or support for progressive political candidates vs conservative candidates."
    ),
    RatingCategory(
        category_id='leadership_dei',
        category_name='DEI in Leadership',
        ai_description="The amount of diverse, equitable, and inclusive representation specifically within the company's executive, high-level leadership and board, where a higher score means the members' groups represent the breakdown of the company's country of origin. 'Group' can mean ethnicity, orientation, sexuality, etc."
    ),
    RatingCategory(
        category_id='employment_dei',
        category_name='DEI in Employment',
        ai_description="The amount of diverse, equitable, and inclusive representation in the company's hiring practices, a higher score means the employees' groups represent the breakdown of the company's country of origin. 'Group' can mean ethnicity, orientation, sexuality, etc."
    ),
    RatingCategory(
        category_id='environmental_impact',
        category_name='Environmental Impact',
        ai_description="Companies that damage the environment will have a low score, companies that have a positive impact on the environment will have a very high score. Environmental damage includes high water usage, carbon emissions, deforestation, etc. Positive environmental impacts include usage of or investment in clean energy (solar, wind, nuclear, etc.), regenerative practices, supporting green causes, etc."
    ),
    RatingCategory(
        category_id='lgbtq_support',
        category_name='LGBTQ Support',
        ai_description="A company with high score will show unwaivering support for the LGBTQ communnity through both their words and their actions."
    ),
    RatingCategory(
        category_id='bipoc_support',
        category_name='BIPOC Support',
        ai_description="A company with a high score will show unwaivering support for the BIPOC communnity through both their words and their actions."
    ),
    RatingCategory(
        category_id='work_conditions',
        category_name='Work Conditions',
        ai_description="A company with unionized, well-paid workers will have a high score. A company with a high rate of workplace injuries, low wages, or a high turnover rate will have a low score."
    ),
    RatingCategory(
        category_id='corporate_ethics',
        category_name='Corporate Ethics',
        ai_description="This category is a catch-all for shady business practices, lawsuits, monopolies, price gouging, etc."
    ),
]

def _generate_gemini_research_prompt(company_name: str, categories: list[RatingCategory] = RATING_CATEGORIES) -> str:
    assert len(categories) > 1
    prompt = f"""\
You are a corporate ethics researcher. Your goal is to search the web and write a detailed report that evaluates the \
ethical and moral character, in both words and actions, of a company, such that it can be "scored" by a reviewer. \
You will be given a list of categories, and you will see if you can find information regarding the company's ethics in that category.
The company you will be evaluating is "{company_name}", and you will be considering the following {len(categories)} categories: \
"""
    for cat in categories:
        prompt += "\n"
        prompt += f'{cat.category_name}: {cat.ai_description}'

    return prompt

def _generate_gemini_scructured_prompt_and_schema(company_name: str, research_report: str, categories: list[RatingCategory] = RATING_CATEGORIES) -> tuple[str, genai.types.Schema]:
    assert len(categories) > 1
    prompt = f"""\
You are a corporate ethics evaluator. You will be scoring a company regarding their ethical and moral character, \
in both words and actions. You will be provided with a list of categories, as well as a research report on that \
company's ethics. For each category on which the report has found substantial information, you will give the company \
a score from 0 to 100. Your output will have a list of objects that each contain a "category_id" and a "score" property. \
Only include objects for the categories for which there is substantial information on the report, enough to reach a conclusion \
as to what their score should be. A 0 on the scale would be as bad as a company could possibly get, and a 100 would be \
as good as possible.

The company you will be evaluating is {company_name}, and the category names, IDs, and descriptions are as follows: \
"""
    for cat in categories:
        prompt += "\n"
        prompt += f"{cat.category_name} - {cat.category_id} - {cat.ai_description}"
    
    prompt += f"""\
\n\nAnd the following is the research report for you to evaluate:
---
{research_report}
---"""
    
    schema = gemtypes.Schema(
        type = genai.types.Type.OBJECT,
        required = ["ratings"],
        properties = {
            "ratings": genai.types.Schema(
                type = genai.types.Type.ARRAY,
                items = genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["category_id", "rating"],
                    properties = {
                        "category_id": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            enum = [cat.category_id for cat in categories]
                        ),
                        "rating": genai.types.Schema(
                            type = genai.types.Type.INTEGER,
                        ),
                    },
                ),
            ),
        },
    )

    return prompt, schema

@dataclass
class Rating:
    score: float
    category_id: str
    category_name: str

    def __str__(self):
        if self.score is not None:
            return f"In {self.category_name}, the rating is {round(self.score * 100)}%"
        return f"This company is unrated for {self.category_name}."

@dataclass
class Company:
    name: str
    ticker: str

    def rate(self) -> list[Rating]:
        research_prompt = _generate_gemini_research_prompt(self.name)
        gemini = _Gem()
        research_report = gemini.call_gemini_research(research_prompt, grounding=True)

        scoring_prompt, schema = _generate_gemini_scructured_prompt_and_schema(self.name, research_report)
        ratings_data = gemini.call_gemini_structured(scoring_prompt, schema)

        ratings = []
        for rating_dict in ratings_data["ratings"]:
            category = next(c for c in RATING_CATEGORIES if c.category_id == rating_dict["category_id"])
            ratings.append(Rating(
                score=rating_dict["rating"] / 100.0,
                category_id=rating_dict["category_id"],
                category_name=category.category_name
            ))

        return ratings


@dataclass
class EvaluatedCompany:
    company: Company
    ratings: list[Rating]

    def __init__(self, company: Company):
        self.company = company
        self.ratings = company.rate()
        self.ratings.sort(key=lambda x: x.category_name)

    def __str__(self):
        return '\n'.join([str(r) for r in self.ratings])

class _Gem:
    client: genai.Client

    def __init__(self):
        self.chat_history = []
        try:
            assert os.environ.get("GEMINI_KEY")
        except AssertionError:
            raise AssertionError("Remember to add your Gemini API key to your .env file in a variable called `GEMINI_KEY`. Read the `README` for more information.")
        self.client = genai.Client(api_key=os.environ.get("GEMINI_KEY"))

    def call_gemini_structured(self, prompt: str, out_schema: gemtypes.Schema) -> dict:
        config = gemtypes.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=out_schema
        )

        response = self.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=self.chat_history + [prompt],
            config=config
        )
        
        return response.parsed # type: ignore
    
    def call_gemini_research(self, prompt: str, grounding: bool = False) -> str:
        config = gemtypes.GenerateContentConfig(
            tools=[gemtypes.Tool(google_search=gemtypes.GoogleSearch())] if grounding else None,
        )

        response = self.client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=self.chat_history + [prompt],
            config=config
        )

        return response.text # type: ignore