This notebook showcases an end-to-end analytics workflow developed to support a gamification system for the CSO (Customer Service/Outbound) team. The project integrates data from multiple sources, cleans and transforms the records, applies custom operational rules, and calculates daily performance metrics used for internal dashboards and leaderboards.

# Key Components
1. Multi-Source Data Integration
- The notebook retrieves data from several systems, including:

A SQL database containing CSO activity logs

Google Sheets storing lead information and employee data

Auxiliary datasets for daily reporting

These sources are combined into a unified analytical table.

2. Data Cleaning & Preprocessing

The workflow includes:

Timestamp cleaning and standardization

Null and invalid data handling

Normalization of employee names

Restructuring fields for analysis

3. Custom Working-Day Logic

A unique business rule defines the “operational day” starting at 12:31 PM, rather than midnight.
The notebook constructs:

A custom date column

Hourly labels for performance grouping

This ensures metrics align with the team’s shift structure.

4. KPI Calculation & Aggregation

The script computes daily and hourly performance metrics, such as:

Lead counts

Conversion indicators

Product-level contributions

Per-staff summaries

The output is optimized for use in progress tracking and monitoring dashboards.

5. Gamification Scoring System

The notebook generates point-based scoring to support internal gamification, based on:

Product targets

Individual completion rates

Daily achievement summaries

This enables ranking and leaderboard generation.
