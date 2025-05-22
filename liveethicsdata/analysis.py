import time
from .config import GEMINI_MODEL_ID
from .utils import string_standard_formatting, sum_weights, empty_function_add_data, empty_function_skip_company
from .data_sources import data_fmp, data_google
from .gemini_utils import get_gemini_client, data_grounded_gemini, ask_competitors, ask_alt_names
from .test_utils import ( # Import test functions if needed for test_mode
    get_test_fmp_data, get_test_google_data_structured,
    get_test_gemini_response, get_test_competitors, get_test_alt_names
)

def aggregate_metrics(metrics_list: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """
    Aggregates metrics from multiple sources using weighted averages.

    Args:
        metrics_list: A list of dictionaries, where each dictionary maps
                      issue IDs to [weight, score] lists.

    Returns:
        A dictionary mapping issue IDs to aggregated [total_weight, final_score] lists.
    """
    aggregated_metrics = {}
    combined_metrics: dict[str, list[list[float]]] = {} # {issue_id: [[w1, s1], [w2, s2], ...]}

    for metrics_source in metrics_list:
        if not isinstance(metrics_source, dict): continue # Skip if not a dict

        for issue_id, data in metrics_source.items():
            # Validate data structure and types
            if (not isinstance(data, list) or len(data) != 2 or
                    not all(isinstance(x, (int, float)) for x in data)):
                print(f"Warning: Skipping invalid metric data for {issue_id}: {data}")
                continue

            weight, score = float(data[0]), float(data[1])

            # Ensure weight is non-negative
            if weight < 0:
                 print(f"Warning: Skipping metric with negative weight for {issue_id}: {weight}")
                 continue

            if issue_id not in combined_metrics:
                combined_metrics[issue_id] = []

            combined_metrics[issue_id].append([weight, score])

    # Calculate weighted averages
    for issue_id, data_points in combined_metrics.items():
        if not data_points: continue

        total_weight = sum(point[0] for point in data_points)

        # Avoid division by zero or negative total weight
        if total_weight <= 0:
            # If total weight is zero, result is undefined or could be default (e.g., [0, 0])
            # Let's default to [0, 0] to indicate no meaningful data
            aggregated_metrics[issue_id] = [0.0, 0.0]
            continue

        weighted_sum = sum(point[0] * point[1] for point in data_points)
        final_score = weighted_sum / total_weight

        # Round results for cleaner output
        aggregated_metrics[issue_id] = [
            round(total_weight, 3),
            round(final_score, 3)
        ]

    return aggregated_metrics

# Main analysis function
def analyze_companies(
        companies: list[str],
        test_mode=False,
        add_data=empty_function_add_data,
        skip_company=empty_function_skip_company,
        model_id=GEMINI_MODEL_ID # Use model_id from config by default
    ) -> dict[str, dict]:
    """
    Analyzes a list of companies by fetching data from various sources,
    aggregating scores, and collecting metadata.

    Args:
        companies: A list of company names (or symbols for FMP).
        test_mode: If True, use mock data functions instead of live API calls.
        add_data: Callback function to process data for each company incrementally.
        skip_company: Callback function to check if a company should be skipped.
        model_id: The Gemini model ID to use for analysis.

    Returns:
        A dictionary where keys are standardized company names and values are
        dictionaries containing 'metrics', 'full_name', 'competitors',
        'alt_names', 'sources', and 'date'.
    """

    gemini_client = get_gemini_client() # Initialize Gemini client

    all_company_data = {}
    for company_input_name in companies:
        # Standardize name for internal use and output key
        company_std_name = string_standard_formatting(company_input_name)
        print(f"\n--- Analyzing {company_input_name} ({company_std_name}) ---")

        if skip_company(company_std_name):
            print(f"Skipping {company_input_name} based on skip_company function.")
            continue

        # --- Data Fetching ---
        # Initialize data containers for each company
        google_datasets = []
        google_sources = []
        fmp_data = {}
        gemini_grounded_data = {}
        competitors = []
        alt_names = []

        if test_mode:
            print("[TEST MODE] Using mock data.")
            # Use test utility functions
            google_results = get_test_google_data_structured(company_input_name) # Needs updated test func
            google_datasets = google_results.get('datasets', []) # Extract datasets
            google_sources = google_results.get('sources', []) # Extract sources
            fmp_data = get_test_fmp_data()
            gemini_grounded_data = get_test_gemini_response(company_input_name)
            competitors = get_test_competitors(company_input_name) # Assumes test func returns list directly
            alt_names = get_test_alt_names(company_input_name) # Needs new test func
            print(f"[TEST MODE] Google Search yielded {len(google_sources)} sources and {len(google_datasets)} analysis datasets.")
            print(f"[TEST MODE] FMP data total weight: {sum_weights(fmp_data)}")
            print(f"[TEST MODE] Gemini grounded data total weight: {sum_weights(gemini_grounded_data)}")

        else:
            # Live data fetching
            # Google Search + Gemini Analysis (requires client)
            if gemini_client:
                 google_raw_results = data_google(company_input_name, gemini_client)
                 # aggregate_metrics expects list of dicts like {issue: [w, s]}
                 # data_google returns {'datasets': [ {issue: [w,s]}, ... ], 'sources': [...]}
                 # We need to pass google_raw_results['datasets'] to aggregate_metrics later
                 google_datasets = google_raw_results.get('datasets', [])
                 google_sources = google_raw_results.get('sources', [])
                 print(f"Google Search yielded {len(google_sources)} sources and {len(google_datasets)} analysis datasets.")
            else:
                 # google_datasets and google_sources already initialized to []
                 print("Skipping Google Search data due to missing Gemini client.")


            # FMP Data (uses company name as symbol, might need adjustment)
            # Consider adding logic to find symbol if needed, or pass symbol separately
            fmp_data = data_fmp(company_input_name)
            print(f"FMP data total weight: {sum_weights(fmp_data)}")

            # Gemini Grounded Research (requires client)
            if gemini_client:
                gemini_grounded_data = data_grounded_gemini(company_input_name, gemini_client, model_id)
                print(f"Gemini grounded data total weight: {sum_weights(gemini_grounded_data)}")
            else:
                 # gemini_grounded_data already initialized to {}
                 print("Skipping Gemini grounded data due to missing Gemini client.")


            # Metadata (requires client)
            if gemini_client:
                competitors_raw = ask_competitors(company_input_name, gemini_client, model_id)
                # ask_competitors now returns the list of products directly, or []
                competitors = competitors_raw if isinstance(competitors_raw, list) else []

                alt_names = ask_alt_names(company_input_name, gemini_client, model_id)
            else:
                 # competitors and alt_names already initialized to []
                 print("Skipping Competitor/Alt Name fetching due to missing Gemini client.")


        # --- Aggregation ---
        # Combine all data sources for aggregation
        # Note: google_datasets is already a list of dicts
        # Ensure all components are lists or dicts before concatenation/aggregation
        all_metrics_sources = google_datasets + ([fmp_data] if fmp_data else []) + ([gemini_grounded_data] if gemini_grounded_data else [])
        final_metrics = aggregate_metrics(all_metrics_sources)
        print(f"Aggregated metrics total weight: {sum_weights(final_metrics)}")


        # --- Store Results ---
        if final_metrics: # Only store if we have some aggregated data
            output_data = {
                "metrics": final_metrics,
                "full_name": company_input_name, # Store original input name
                "competitors": competitors, # Store competitor data structure
                "alt_names": alt_names,
                "sources": google_sources, # Store links from Google search
                "date": int(time.time()) # Timestamp of analysis completion
            }
            all_company_data[company_std_name] = output_data
            # Call incremental add function
            add_data({company_std_name: output_data})
        else:
            print(f"No final metrics generated for {company_input_name}. Skipping storage.")

    print("\n--- Analysis Complete ---")
    return all_company_data