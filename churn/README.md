This notebook contains a complete workflow for analyzing customer churn using data sourced from multiple internal systems. The project integrates SQL-based data extraction, data cleaning, feature preparation, and exploratory analysis to understand customer behavior patterns related to churn.

Key components of the notebook include:

* Database Integration: Connecting to a MySQL database to retrieve customer information and transactional history. The notebook executes SQL queries, fetches records, and converts them into structured DataFrames for further analysis.
* Data Aggregation from Multiple Sources: Loading additional CSV datasets (such as return customer data, delivery codes, product information, etc.) and merging them into a unified analytical table. Columns are standardized, renamed, and validated.
* Feature Engineering: Cleaning raw fields, handling missing values, merging relational attributes, and constructing churn-related behavioral metrics.
* Exploratory Data Analysis (EDA): Generating insights from customer patterns, return behavior, and other business attributes to identify churn indicators.

The notebook demonstrates practical end-to-end analytics work, starting from raw database extraction to a refined dataset ready for modeling, while keeping all sensitive details and internal assets abstracted.
