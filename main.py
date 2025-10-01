# Project: what makes a land happy?

# set up the environment
import kagglehub
import pandas as pd
import pandasql
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
from thefuzz import process
from download import download_datasets
from string_helpers import normalize_country, apply_alias, fuzzy_merge

# checks for csvs and re-download data if csvs are not available
data = download_datasets()
happiness_df = data['happiness_df'][['Country name', 'Country_clean', 'Regional indicator', 'Ladder score']].rename(columns={"Country name": "Country", "Regional indicator" : "Region", "Ladder score" : "Happiness"})

# isolate the country name + column that I need from supporting datasets
# rename the column
# then clean each one
# then merge each one into happiness_df

# 1) Average Wage - simple [but mind not full matches, also count them]
average_wage_isolated_df = data['average_wage_df'][['country_name', 'median_salary', 'average_salary', 'lowest_salary', 'highest_salary']] 
# Note for above: DOUBLE SQUARE BRACKETS!!! 1st brackets are the INDEX selector, 2nd brackets are the LIST of columns we pass
average_wage_isolated_df = average_wage_isolated_df.rename(columns={'country_name': 'Country', 'median_salary' : 'Median Salary', 'average_salary' : 'Average Salary', 'lowest_salary' : 'Lowest Salary', 'highest_salary' : 'Highest Salary'})
average_wage_isolated_df = average_wage_isolated_df.dropna().drop_duplicates()
average_wage_isolated_df['Country'] = average_wage_isolated_df['Country'].str.strip()
average_wage_isolated_df['Median Salary'] = pd.to_numeric(average_wage_isolated_df['Median Salary'], errors='coerce')
average_wage_isolated_df['Average Salary'] = pd.to_numeric(average_wage_isolated_df['Average Salary'], errors='coerce')
average_wage_isolated_df['Lowest Salary'] = pd.to_numeric(average_wage_isolated_df['Lowest Salary'], errors='coerce')
average_wage_isolated_df['Highest Salary'] = pd.to_numeric(average_wage_isolated_df['Highest Salary'], errors='coerce')
average_wage_isolated_df["Country_clean"] = average_wage_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    average_wage_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Median Salary', 'Average Salary', 'Lowest Salary', 'Highest Salary'],
    threshold=85
)

# 2) IQ Air - extract country from city, aggregate cities per country, compute mean AQI, and merge
iq_air_df = data['iq_air_df'].copy()
iq_air_df.columns = [c.strip() for c in iq_air_df.columns]

# detect likely column names
city_col = next((c for c in iq_air_df.columns if 'city' in c.lower()), None)
# prefer a 4-digit year column like "2021" then fallback to first numeric column
value_col = next((c for c in iq_air_df.columns if re.fullmatch(r'\d{4}', c)), None)
if value_col is None:
    value_col = next((c for c in iq_air_df.columns if '2021' in c or '2022' in c), None)
if value_col is None:
    numeric_cols = iq_air_df.select_dtypes(include='number').columns.tolist()
    value_col = numeric_cols[0] if numeric_cols else None

if city_col is None or value_col is None:
    raise ValueError("Could not detect 'city' column or a numeric AQI column in the IQAir dataset.")

# helper: extract country from city text
def extract_country_from_city(city_text: str) -> str:
    if pd.isna(city_text):
        return None
    s = str(city_text).strip()
    # common formats:
    # "City, Country", "City, State, Country", "City (Country)", "City - Country"
    # 1) last comma-separated token
    m = re.search(r',\s*([^,]+)\s*$', s)
    if m:
        return m.group(1).strip()
    # 2) parentheses at the end
    m = re.search(r'\(([^)]+)\)\s*$', s)
    if m:
        return m.group(1).strip()
    # 3) dash / pipe separators
    m = re.search(r'[-|]\s*([^-\|]+)\s*$', s)
    if m:
        return m.group(1).strip()
    # 4) fallback: last word (may be wrong for multi-word country names but better than nothing)
    parts = s.split()
    return parts[-1].strip() if parts else None

# apply extraction and cleaning
iq_air_df['Country_extracted'] = iq_air_df[city_col].astype(str).apply(extract_country_from_city)
iq_air_df['Country_clean'] = iq_air_df['Country_extracted'].map(normalize_country).map(apply_alias)

# numeric conversion for the AQI/value column
iq_air_df[value_col] = pd.to_numeric(iq_air_df[value_col], errors='coerce')

# aggregate per country: mean (and median) of the numeric value and count of cities
iq_air_country = (
    iq_air_df
    .dropna(subset=['Country_clean'])
    .groupby('Country_clean', as_index=False)
    .agg(
        IQAir_AQI_Mean = (value_col, 'mean'),
        IQAir_AQI_Median = (value_col, 'median')
    )
)

# optional: round the AQI numbers
iq_air_country['IQAir_AQI_Mean'] = iq_air_country['IQAir_AQI_Mean'].round(2)
iq_air_country['IQAir_AQI_Median'] = iq_air_country['IQAir_AQI_Median'].round(2)

# fuzzy-merge into happiness_df (matches pipeline style)
happiness_df = fuzzy_merge(
    happiness_df,
    iq_air_country,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['IQAir_AQI_Mean', 'IQAir_AQI_Median'],
    threshold=85
)

# 3) Lifespan - need to only grab rows with the latest year
life_expectancy_isolated_df = data['life_expectancy_df'][['Country', 'Year', 'infant deaths', 'Alcohol', 'Life expectancy']]
life_expectancy_isolated_df = life_expectancy_isolated_df.rename(columns={'infant deaths': "Infant Death Rate", 'Alcohol': 'Alcohol Consumption Rate', 'Life expectancy' : 'Life Expectancy'})
life_expectancy_isolated_df = life_expectancy_isolated_df.drop_duplicates()
life_expectancy_isolated_df['Country'] = life_expectancy_isolated_df['Country'].str.strip()
life_expectancy_isolated_df['Infant Death Rate'] = pd.to_numeric(life_expectancy_isolated_df['Infant Death Rate'], errors='coerce')
life_expectancy_isolated_df['Infant Survival Rate'] = 1000 - life_expectancy_isolated_df['Infant Death Rate'] 
# Note for above: PD applies things ELEMENT-WIDE!!! so: every row in gets transformed into "survival rate" as 1000 - value.
life_expectancy_isolated_df['Alcohol Consumption Rate'] = pd.to_numeric(life_expectancy_isolated_df['Alcohol Consumption Rate'], errors='coerce')
life_expectancy_isolated_df['Life Expectancy'] = pd.to_numeric(life_expectancy_isolated_df['Life Expectancy'], errors='coerce')
life_expectancy_isolated_df["Country_clean"] = life_expectancy_isolated_df["Country"].map(normalize_country).map(apply_alias)
life_expectancy_isolated_df = life_expectancy_isolated_df.sort_values(
    ["Country_clean", "Year"]
)
life_expectancy_latest = (
    life_expectancy_isolated_df
    .groupby("Country_clean")
    .agg({
        "Year": "max",
        "Infant Survival Rate": "last",
        "Alcohol Consumption Rate": "last",
        "Life Expectancy": "last"
    })
    .reset_index()
)
happiness_df = fuzzy_merge(
    happiness_df,
    life_expectancy_latest,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Infant Survival Rate', 'Alcohol Consumption Rate', 'Life Expectancy'],
    threshold=85
)

# 4) Netflix - simple
netflix_isolated_df = data['netflix_data_df'][['Country', 'Cost Per Month - Standard ($)']]
netflix_isolated_df = netflix_isolated_df.rename(columns={'Cost Per Month - Standard ($)': 'Netflix (USD/month)'})
netflix_isolated_df = netflix_isolated_df.dropna().drop_duplicates()
netflix_isolated_df['Country'] = netflix_isolated_df['Country'].str.strip()
netflix_isolated_df['Netflix (USD/month)'] = pd.to_numeric(netflix_isolated_df['Netflix (USD/month)'], errors='coerce')
netflix_isolated_df["Country_clean"] = netflix_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    netflix_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Netflix (USD/month)'],
    threshold=85
)

# 5) Women's Safety - simple
women_safety_isolated_df = data['women_safety_df'][['country', 'MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023']]
women_safety_isolated_df = women_safety_isolated_df.rename(columns={'country' : 'Country', 'MostDangerousCountriesForWomen_WomenPeaceAndSecurityIndex_Score_2023': 'Women Safety Index'})
women_safety_isolated_df = women_safety_isolated_df.dropna().drop_duplicates()
women_safety_isolated_df['Country'] = women_safety_isolated_df['Country'].str.strip()
women_safety_isolated_df['Women Safety Index'] = pd.to_numeric(women_safety_isolated_df['Women Safety Index'], errors='coerce')
women_safety_isolated_df["Country_clean"] = women_safety_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    women_safety_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Women Safety Index'],
    threshold=85
)

# 6) Temperature - need to only grab rows with the latest year
temperature_isolated_df = data['temperature_df'][['Entity', 'Day', 'year', 'Average surface temperature']]
temperature_isolated_df = temperature_isolated_df.rename(
    columns={'Entity': 'Country', 'Average surface temperature': 'Average Temperature'}
)
temperature_isolated_df = temperature_isolated_df[pd.to_datetime(temperature_isolated_df['Day']).dt.month.isin([1, 12])]
temperature_isolated_df = temperature_isolated_df[temperature_isolated_df["year"] == 2024]
# Need to turn this:
# Argentine 2024-01-15 2024 +20
# Argentine 2024-12-15 2024 +10
# Into this:
# Argentine +20 (column 'average summer temp') +15 (column 'average winter temp') <--- sort the higher one into summer, the lower into winter
jan_df = temperature_isolated_df[pd.to_datetime(temperature_isolated_df['Day']).dt.month == 1] # Separate January and December
dec_df = temperature_isolated_df[pd.to_datetime(temperature_isolated_df['Day']).dt.month == 12] 
merged_df = jan_df.merge(dec_df, on=['Country', 'year'], suffixes=('_Jan', '_Dec')) # Merge January and December rows by Country and Year
merged_df['Average Summer Temperature'] = merged_df[['Average Temperature_Jan', 'Average Temperature_Dec']].max(axis=1) # Create Summer and Winter columns
merged_df['Average Winter Temperature'] = merged_df[['Average Temperature_Jan', 'Average Temperature_Dec']].min(axis=1)
temperature_isolated_df = merged_df[['Country', 'Average Summer Temperature', 'Average Winter Temperature']] # Keep only the relevant columns
temperature_isolated_df.loc[:, 'Country'] = temperature_isolated_df['Country'].str.strip()
temperature_isolated_df.loc[:, 'Average Summer Temperature'] = pd.to_numeric(temperature_isolated_df['Average Summer Temperature'], errors='coerce')
temperature_isolated_df.loc[:, 'Average Winter Temperature'] = pd.to_numeric(temperature_isolated_df['Average Winter Temperature'], errors='coerce')
temperature_isolated_df = temperature_isolated_df.copy()
# Not on the above: some DFs are not independent!!! they don't have data, just pointers to the original. copy and use .loc to not alter the original and make them independent
temperature_isolated_df["Country_clean"] = temperature_isolated_df["Country"].map(normalize_country).map(apply_alias)
# Note on the above: df.loc[<row_selector>, <column_selector>]!!! # select or assign values for specific rows and columns, ":"" selects all rows or columns
happiness_df = fuzzy_merge(
    happiness_df,
    temperature_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Average Winter Temperature', 'Average Summer Temperature'],
    threshold=85
)

# 7) Population - simple
population_isolated_df = data['population_df'][['Country/Territory', '2022 Population']]
population_isolated_df = population_isolated_df.rename(columns={'Country/Territory' : 'Country', '2022 Population': 'Population'})
population_isolated_df = population_isolated_df.dropna().drop_duplicates()
population_isolated_df['Country'] = population_isolated_df['Country'].str.strip()
population_isolated_df['Population'] = pd.to_numeric(population_isolated_df['Population'], errors='coerce')
population_isolated_df["Country_clean"] = population_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    population_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Population'],
    threshold=85
)

# 8) Energy Consumption - need to only grab rows with the latest year
energy_isolated_df = data['energy_consumption_df'][['country', 'year', 'energy_per_capita', 'fossil_share_elec', 'nuclear_share_elec', 'renewables_share_elec']]
energy_isolated_df = energy_isolated_df.rename(columns={'country' : 'Country',  'energy_per_capita' : 'Energy Per Capita', 'fossil_share_elec' : '% of Power from Fossil Fuels', 'nuclear_share_elec' : '% of Power from Nuclear', 'renewables_share_elec' : '% of Power from Renewables'})
energy_isolated_df = energy_isolated_df[~energy_isolated_df['Country'].str.contains(r"\(|region|aggregate|ember", case=False, na=False)] # drop obvious aggregates 
energy_isolated_df['Country'] = energy_isolated_df['Country'].str.strip()
energy_isolated_df['Energy Per Capita'] = pd.to_numeric(energy_isolated_df['Energy Per Capita'], errors='coerce')
energy_isolated_df['% of Power from Fossil Fuels'] = pd.to_numeric(energy_isolated_df['% of Power from Fossil Fuels'], errors='coerce')
energy_isolated_df['% of Power from Nuclear'] = pd.to_numeric(energy_isolated_df['% of Power from Nuclear'], errors='coerce')
energy_isolated_df['% of Power from Renewables'] = pd.to_numeric(energy_isolated_df['% of Power from Renewables'], errors='coerce')
energy_isolated_df['year'] = pd.to_numeric(energy_isolated_df['year'], errors='coerce')
energy_isolated_df["Country_clean"] = energy_isolated_df["Country"].map(normalize_country).map(apply_alias)
energy_isolated_df = energy_isolated_df.sort_values(
    ["Country_clean", "year"]
)
energy_latest = (
    energy_isolated_df
    .groupby("Country_clean")
    .agg({
        "year": "max",
        "Energy Per Capita": "last",
        "% of Power from Fossil Fuels": "last",
        "% of Power from Nuclear": "last",
        "% of Power from Renewables": "last"
    })
    .reset_index()
)
happiness_df = fuzzy_merge(
    happiness_df,
    energy_latest,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Energy Per Capita', '% of Power from Fossil Fuels', '% of Power from Nuclear', '% of Power from Renewables'],
    threshold=85
)

# 9) World Bank Development - need to only grab rows with the latest year
wbdi_isolated_df = data['world_bank_development_df'][['country', 'date', 'agricultural_land%', 'forest_land%', 'land_area', 'avg_precipitation', 'control_of_corruption_std']]
wbdi_isolated_df = wbdi_isolated_df.rename(columns={'country' : 'Country', 'agricultural_land%' : '% of Agricultural Land', 'forest_land%' : '% of Forest Land', 'land_area' : 'Territory', 'avg_precipitation' : 'Average Rainfall', 'control_of_corruption_std' : "Corruption"})
wbdi_isolated_df['Country'] = wbdi_isolated_df['Country'].str.strip()
wbdi_isolated_df['% of Agricultural Land'] = pd.to_numeric(wbdi_isolated_df['% of Agricultural Land'], errors='coerce')
wbdi_isolated_df['% of Forest Land'] = pd.to_numeric(wbdi_isolated_df['% of Forest Land'], errors='coerce')
wbdi_isolated_df['Territory'] = pd.to_numeric(wbdi_isolated_df['Territory'], errors='coerce')
wbdi_isolated_df['Average Rainfall'] = pd.to_numeric(wbdi_isolated_df['Average Rainfall'], errors='coerce')
wbdi_isolated_df['Corruption'] = pd.to_numeric(wbdi_isolated_df['Corruption'], errors='coerce')
wbdi_isolated_df["Country_clean"] = wbdi_isolated_df["Country"].map(normalize_country).map(apply_alias)
wbdi_isolated_df['date'] = pd.to_datetime(wbdi_isolated_df['date'], errors='coerce')
wbdi_isolated_df["Country_clean"] = wbdi_isolated_df["Country"].map(normalize_country).map(apply_alias)
# Copy Average Rainfall from 2020 to 2021
rainfall_2020 = (
    wbdi_isolated_df.loc[wbdi_isolated_df['date'].dt.year == 2020]
    .groupby('Country_clean')['Average Rainfall']
    .first()   # pick the first row per Country_clean
)
mask_2021 = (wbdi_isolated_df['date'].dt.year == 2021) & (wbdi_isolated_df['Average Rainfall'].isna())
wbdi_isolated_df.loc[mask_2021, 'Average Rainfall'] = wbdi_isolated_df.loc[mask_2021, 'Country_clean'].map(rainfall_2020)
wbdi_isolated_df = wbdi_isolated_df[wbdi_isolated_df['date'].dt.year == 2021]
happiness_df = fuzzy_merge(
    happiness_df,
    wbdi_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['% of Agricultural Land', '% of Forest Land', 'Territory', 'Average Rainfall', 'Corruption'],
    threshold=85
)

# 10) Food Production - need to only grab rows with the latest year
food_production_isolated_df = data['food_production_df'][['Entity', 'Year', 'Wheat Production (tonnes)', 'Rye  Production (tonnes)', 'Potatoes  Production (tonnes)', 'Meat, chicken  Production (tonnes)', 'Avocados Production (tonnes)']]
food_production_isolated_df = food_production_isolated_df.rename(columns={'Entity' : 'Country', 'Rye  Production (tonnes)' : 'Rye Production (tonnes)', 'Meat, chicken  Production (tonnes)' : 'Meat, chicken Production (tonnes)', 'Potatoes  Production (tonnes)' : 'Potatoes Production (tonnes)'})
food_production_isolated_df = food_production_isolated_df.dropna().drop_duplicates()
food_production_isolated_df['Country'] = food_production_isolated_df['Country'].str.strip()
food_production_isolated_df['Wheat Production (tonnes)'] = pd.to_numeric(food_production_isolated_df['Wheat Production (tonnes)'], errors='coerce')
food_production_isolated_df['Rye Production (tonnes)'] = pd.to_numeric(food_production_isolated_df['Rye Production (tonnes)'], errors='coerce')
food_production_isolated_df['Potatoes Production (tonnes)'] = pd.to_numeric(food_production_isolated_df['Potatoes Production (tonnes)'], errors='coerce')
food_production_isolated_df['Meat, chicken Production (tonnes)'] = pd.to_numeric(food_production_isolated_df['Meat, chicken Production (tonnes)'], errors='coerce')
food_production_isolated_df['Avocados Production (tonnes)'] = pd.to_numeric(food_production_isolated_df['Avocados Production (tonnes)'], errors='coerce')
food_production_isolated_df["Country_clean"] = food_production_isolated_df["Country"].map(normalize_country).map(apply_alias)
food_production_isolated_df = food_production_isolated_df.sort_values(
    ["Country_clean", "Year"]
)
food_production_latest = (
    food_production_isolated_df
    .groupby("Country_clean")
    .agg({
        "Year": "max",
        "Wheat Production (tonnes)": "last",
        "Rye Production (tonnes)": "last",
        "Potatoes Production (tonnes)" : "last",
        "Meat, chicken Production (tonnes)": "last",
        "Avocados Production (tonnes)": "last",
    })
    .reset_index()
)
happiness_df = fuzzy_merge(
    happiness_df,
    food_production_latest,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Wheat Production (tonnes)', 'Rye Production (tonnes)', 'Potatoes Production (tonnes)', 'Meat, chicken Production (tonnes)', 'Avocados Production (tonnes)'],
    threshold=85
)

# 11) Petrol Price (USD/liter) Petrol Prices - simple
petrol_prices_isolated_df = data['petrol_prices_df'][['Country', 'Daily Oil Consumption (Barrels)', 'Price Per Liter (USD)']]
petrol_prices_isolated_df = petrol_prices_isolated_df.rename(columns={'Price Per Liter (USD)' : 'Petrol Price (USD/liter)'})
petrol_prices_isolated_df = petrol_prices_isolated_df.dropna().drop_duplicates()
petrol_prices_isolated_df['Country'] = petrol_prices_isolated_df['Country'].str.strip()
petrol_prices_isolated_df['Daily Oil Consumption (Barrels)'] = pd.to_numeric(petrol_prices_isolated_df['Daily Oil Consumption (Barrels)'], errors='coerce')
petrol_prices_isolated_df['Petrol Price (USD/liter)'] = pd.to_numeric(petrol_prices_isolated_df['Petrol Price (USD/liter)'], errors='coerce')
petrol_prices_isolated_df["Country_clean"] = petrol_prices_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    petrol_prices_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Petrol Price (USD/liter)', 'Daily Oil Consumption (Barrels)'],
    threshold=85
)

# 12) CO2 Emissions - simple
co2_emissions_isolated_df = data['co2_emissions_df'][['Country Name', '2019']]
co2_emissions_isolated_df = co2_emissions_isolated_df.rename(columns={'Country Name': 'Country', '2019' : 'CO2 Emissions (ton per capita)'})
co2_emissions_isolated_df = co2_emissions_isolated_df.dropna().drop_duplicates()
co2_emissions_isolated_df['Country'] = co2_emissions_isolated_df['Country'].str.strip()
co2_emissions_isolated_df['CO2 Emissions (ton per capita)'] = pd.to_numeric(co2_emissions_isolated_df['CO2 Emissions (ton per capita)'], errors='coerce')
co2_emissions_isolated_df["Country_clean"] = co2_emissions_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    co2_emissions_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['CO2 Emissions (ton per capita)'],
    threshold=85
)

# use pandas to find which metrics correlate to happiness and which don't
numeric_cols = happiness_df.select_dtypes(include='number') # Focus on numeric columns 
corr_matrix = numeric_cols.corr() # Correlation matrix
corr_with_happiness = corr_matrix['Happiness'].drop('Happiness') # Remove the target
colors = corr_with_happiness.apply(lambda x: 'blue' if x>0 else 'red').sort_values(ascending=False) # Differentiate positive and negative correlation
corr_sorted = corr_with_happiness.abs().sort_values(ascending=False) # Take absolute value
plt.figure(figsize=(8,6))
plt.barh(corr_sorted.index, corr_sorted.values, color=[colors[i] for i in corr_sorted.index])
plt.xlabel("Strength of Correlation with Happiness (absolute)")
plt.title("Which metrics affect Happiness most (color shows + / -)")
plt.subplots_adjust(left=0.35)
plt.show()

# exporting to json
def export_as_json():
    os.makedirs("exports", exist_ok=True)
    json_path_main = os.path.join("exports", "happiness_df.json")
    if not os.path.exists(json_path_main):
        happiness_df.to_json(json_path_main, orient="records", indent=2)
        print(f"✅ Exported to {json_path_main}")
    else:
        print(f"⚠️ File already exists at {json_path_main}, skipping export")
    json_path_corr = os.path.join("exports", "happiness_corr.json")
    if not os.path.exists(json_path_corr):
        # compute signed correlations and export both signed and absolute weights
        numeric_cols = happiness_df.select_dtypes(include='number')
        corr_with_happiness = numeric_cols.corr()['Happiness'].drop('Happiness')

        # DataFrame with signed weight and absolute magnitude
        corr_df = pd.DataFrame({
            "metric": corr_with_happiness.index,
            "weight_signed": corr_with_happiness.values,           # signed correlation (-1..1)
            "weight_magnitude": corr_with_happiness.abs().values  # magnitude for display / scaling
        })
        corr_df.to_json(json_path_corr, orient="records", indent=2)
        print(f"✅ Exported to {json_path_corr}")
    else:
        print(f"⚠️ File already exists at {json_path_corr}, skipping export")

export_as_json()

# count nulls
null_counts = happiness_df.isna().sum().sort_values(ascending=False)

print("Null count per column:")
print(null_counts)