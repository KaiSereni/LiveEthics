# This file is now primarily for demonstrating usage or running as a script.
# The core logic has been moved to other modules within the package.

import json
import os
from .analysis import analyze_companies

# --- Main execution block ---
if __name__ == "__main__":
    # Configuration
    TEST_MODE = False  # Set to False for live API calls
    OUTPUT_FILENAME = "output.json"

    if TEST_MODE:
        print("[TEST MODE ENABLED] Using mock data for API calls.")
        # In test mode, API keys are not strictly needed but dotenv still loads them
    else:
        print("[LIVE MODE] Using live API calls.")
        # Ensure necessary environment variables are set in .env
        required_vars = ["GOOGLE_API_KEY", "VERTEXAI_PROJECT_NAME"] # FMP is optional
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please ensure they are set in the .env file.")
            exit(1) # Exit if required keys are missing in live mode


    # List of companies to analyze
    companies_to_analyze = [
        "Google",
        "Apple",
        # "Tesla",
        # "Temu",
        # "Meta"
    ]

    # Define callback functions (optional)
    def my_add_data_func(data: dict):
        """Custom function to handle data for each company as it's processed."""
        company_key = list(data.keys())[0]
        print(f"Received data for {data[company_key]['full_name']}. Metrics count: {len(data[company_key]['metrics'])}")
        # Could potentially write to DB or update UI here
        pass

    def my_skip_company_func(company_std_name: str) -> bool:
        """Custom function to decide whether to skip a company."""
        if company_std_name == "tesla":
             print("Skipping Tesla based on custom rule.")
             return True
        return False

    # --- Run Analysis ---
    print(f"Starting analysis for: {', '.join(companies_to_analyze)}")
    final_data = analyze_companies(
        companies=companies_to_analyze,
        test_mode=TEST_MODE,
        add_data=my_add_data_func,
        skip_company=my_skip_company_func
        # model_id can be overridden here if needed, e.g., model_id="gemini-pro"
    )

    # --- Handle Output ---
    if not TEST_MODE:
        # Load previous data if exists
        previous_data = {}
        try:
            if os.path.exists(OUTPUT_FILENAME):
                with open(OUTPUT_FILENAME, "r") as f:
                    previous_data = json.load(f)
                print(f"Loaded previous data from {OUTPUT_FILENAME}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {OUTPUT_FILENAME}. Starting fresh.")
        except Exception as e:
            print(f"Error loading {OUTPUT_FILENAME}: {e}. Starting fresh.")

        # Merge new data into previous data
        if final_data: # Only update if analysis returned data
            for company_key, company_data in final_data.items():
                previous_data[company_key] = company_data # Overwrite/add new data
            print(f"Merged analysis results for {len(final_data)} companies.")

            # Write updated data back to file
            try:
                with open(OUTPUT_FILENAME, "w") as f:
                    json.dump(previous_data, f, indent=2)
                print(f"Successfully saved updated data to {OUTPUT_FILENAME}")
            except Exception as e:
                print(f"Error saving data to {OUTPUT_FILENAME}: {e}")
        else:
             print("No new data generated from analysis. Output file not updated.")

    else:
        # In test mode, just print the final aggregated data
        print("\n--- [TEST MODE] Final Aggregated Data ---")
        print(json.dumps(final_data, indent=2))

    print("\nScript finished.")

