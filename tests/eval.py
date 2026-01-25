from src.liveethics.companies import Company, EvaluatedCompany

test_company = Company("Google", 'GOOG')
test_evaluation = EvaluatedCompany(test_company)

for rating in test_evaluation.ratings:
    print(f"{rating.category_name} - {rating.score*100}%")