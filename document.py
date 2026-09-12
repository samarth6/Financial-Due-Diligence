from edgar import set_identity, Company

set_identity("Vanshika Mittal mitvanshika@gmail.com")

company = Company("AAPL")
filings = company.get_filings(form="10-K")
latest = filings.latest()

tenk = latest.obj()
financials = tenk.financials

income_df = financials.income_statement().to_dataframe()
balance_df = financials.balance_sheet().to_dataframe()
cashflow_df= financials.cash_flow_statement().to_dataframe()

income_df.to_csv("apple_income_statement.csv")
balance_df.to_csv("apple_balance_sheet.csv")
cashflow_df.to_csv("apple_cash_flow.csv")

print(income_df.head())

