# add_units.py
import pandas as pd

def add_units_to_happiness_df(happiness_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds units to column names in happiness_df where applicable.
    Returns a new DataFrame with updated column names.
    """

    # Mapping of columns to their units (based on Kaggle dataset descriptions)
    units_mapping = {
        "Happiness": "(score 0-10)",
        "Median Salary": "(USD/year)",
        "Average Salary": "(USD/year)",
        "Lowest Salary": "(USD/year)",
        "Highest Salary": "(USD/year)",
        "IQAir_AQI": "(AQI index)",
        "Infant Survival Rate": "(per 1000 births)",
        "Alcohol Consumption Rate": "(liters per capita per year)",
        "Life Expectancy": "(years)",
        "Netflix (USD/month)": "(USD/month)",
        "Women Safety Index": "(score 0-1)",
        "Average Summer Temperature": "(°C)",
        "Average Winter Temperature": "(°C)",
        "Population": "(people)",
        "Energy Per Capita": "(kWh per capita)",
        "% of Power from Fossil Fuels": "(%)",
        "% of Power from Nuclear": "(%)",
        "% of Power from Renewables": "(%)",
        "% of Agricultural Land": "(%)",
        "% of Forest Land": "(%)",
        "Territory": "(sq km)",
        "Average Rainfall": "(mm/year)",
        "Corruption": "(std score)",
        "Wheat Production": "(tonnes)",
        "Rye Production": "(tonnes)",
        "Potatoes Production": "(tonnes)",
        "Meat, Chicken Production": "(tonnes)",
        "Avocados Production": "(tonnes)",
        "Petrol Price": "(USD/liter)",
        "Daily Oil Consumption": "(barrels/day)",
        "CO2 Emissions": "(tonnes per capita)",
        "Price per Square Meter to Buy Apartment in City Centre": "(USD/sqm)",
        "Fitness Club, Monthly Fee for 1 Adult": "(USD/month)",
        "Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)": "(USD/month)",
        "Coke/Pepsi (0.33 liter bottle, in restaurants)": "(USD/bottle)",
        "Meal for 2 People, Mid-range Restaurant, Three-course": "(USD/meal)",
        "GDP Per Capita": "(USD/year)",
        "Military Expenditure": "(USD/year)",
        "Yearly Homicide Rate (% per 100,000 people)": "(per 100,000 people)",
        "Average Age": "(years)",
        "Inflation Rate": "(year, %)"
    }

    # Update column names
    updated_columns = {}
    for col in happiness_df.columns:
        if col in units_mapping:
            updated_columns[col] = f"{col} {units_mapping[col]}"
        else:
            updated_columns[col] = col

    happiness_df = happiness_df.rename(columns=updated_columns)
    return happiness_df
