# Mock data functions for testing purposes

def get_test_fmp_data() -> dict:
    """Returns mock FMP ESG data."""
    print("Using mock FMP data.")
    return {
        "ENV": [100.0, 75.0], # Weight, Score
        "PAY": [50.0, 65.0]   # Weight, Score
    }

def get_test_google_data_structured(company_name: str) -> dict:
    """Returns mock structured Google search/analysis data."""
    print(f"Using mock Google data for {company_name}.")
    if company_name:
        # This structure matches the expected output of data_google
        return {
            "datasets": [ # List of dicts, as if from multiple ask_about_article calls
                 {
                    "DEI_L": [50.0, 20.0],
                    "DEI_H": [60.0, 30.0],
                 },
                 {
                    "QUEER": [70.0, 40.0],
                    "BIPOC": [80.0, 50.0],
                    "PAY": [90.0, 60.0]
                 }
            ],
            "sources": [
                f"http://mocklink.com/{company_name}/source1",
                f"http://mocklink.com/{company_name}/source2"
                ]
        }
    return {"datasets": [], "sources": []}


def get_test_gemini_response(company_name: str) -> dict:
    """Returns mock Gemini grounded research response."""
    print(f"Using mock Gemini grounded response for {company_name}.")
    if company_name:
        return {
            "DEI_L": [50.0, 75.0],
            "DEI_H": [50.0, 80.0],
            "QUEER": [50.0, 70.0],
            "BIPOC": [50.0, 65.0],
            "PAY": [50.0, 60.0],
            "ENV": [50.0, 85.0]
        }
    return {}

def get_test_competitors(company_name: str) -> list:
    """Returns mock competitor data (list of product dicts)."""
    print(f"Using mock competitor data for {company_name}.")
    # Mock structure similar to what Gemini function call might return
    mock_product_list = [
         {
             "product_name": f"{company_name} Product A",
             "competitor_products": [
                 {"product_name": "Competitor Product X", "parent_company": "Competitor Inc."},
                 {"product_name": "Alternative Product Y", "parent_company": "Another Competitor LLC"}
             ],
             "availability": {"online": True, "in_person": False}
         },
         {
             "product_name": f"{company_name} Service B",
              "competitor_products": [
                 {"product_name": "Similar Service Z", "parent_company": "Competitor Inc."}
             ],
             "availability": {"online": True, "in_person": True}
         }
    ]
    test_competitors = {
        "Apple": mock_product_list,
        "Google": mock_product_list,
        "Meta": mock_product_list,
        "TestCompany": [] # Example of company with no competitors found
    }
    # Return specific mock data or a default structure
    return test_competitors.get(company_name, mock_product_list[:1]) # Default to one product


def get_test_alt_names(company_name: str) -> list[str]:
    """Returns mock alternative names."""
    print(f"Using mock alternative names for {company_name}.")
    test_names = {
        "Google": ["Alphabet", "GOOGL", "Goggle"],
        "Meta": ["Facebook", "META", "Face Book Inc"],
        "Apple": ["AAPL", "Apple Inc."]
    }
    return test_names.get(company_name, [f"{company_name} MockName1", f"{company_name} MockName2"])
