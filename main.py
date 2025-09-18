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


def normalize_country(name: str) -> str:
    if pd.isna(name):
        return None
    name = name.lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z\s]", "", name)   # remove punctuation
    name = re.sub(r"\s+", " ", name)       # collapse spaces
    return name.strip()

country_aliases = {
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "south korea": "korea republic of",
    "north korea": "korea democratic peoples republic of",
    "russia": "russian federation",
    "venezuela": "venezuela bolivarian republic of",
    "czechia": "czech republic",
    "ivory coast": "cote divoire",
    "antigua": "antigua and barbuda",
}

def apply_alias(name):
    return country_aliases.get(name, name)

def fuzzy_merge(df_left,
                df_right,
                left_on='Country_clean',
                right_on='Country_clean',
                right_cols=None,
                threshold=90,
                return_matches=False):
    """
    Fuzzy-merge df_right into df_left.
    - right_cols: list of columns from df_right to bring in (excluding right_on). If None, bring all except right_on.
    - threshold: minimum similarity score to accept a match (0-100).
    - return_matches: if True, also returns the mapping dict (left_value -> matched_right_value or None).
    """
    if left_on not in df_left.columns:
        raise KeyError(f"left_on '{left_on}' not found in left dataframe")
    if right_on not in df_right.columns:
        raise KeyError(f"right_on '{right_on}' not found in right dataframe")

    # prepare right columns to merge
    if right_cols is None:
        right_cols = [c for c in df_right.columns if c != right_on]
    else:
        right_cols = [c for c in right_cols if c != right_on]

    # unique right keys to match against
    right_keys = df_right[right_on].dropna().astype(str).unique().tolist()

    # build mapping from left values -> best right key (or None)
    mapping = {}
    left_vals = df_left[left_on].dropna().astype(str).unique().tolist()
    for val in left_vals:
        match = process.extractOne(val, right_keys)
        if match is None:
            mapping[val] = None
        else:
            best, score = match
            mapping[val] = best if score >= threshold else None

    # apply mapping to create helper key column
    left = df_left.copy()
    left['_fuzzy_key'] = left[left_on].astype(str).map(mapping)

    # prepare right subset (one row per right_on)
    right_subset = df_right[[right_on] + right_cols].copy()
    right_subset = right_subset.drop_duplicates(subset=right_on)

    # merge on helper key -> right_on
    merged = pd.merge(left, right_subset, left_on='_fuzzy_key', right_on=right_on, how='left', suffixes=('', '_r'))

    # For any columns that exist in both df_left and df_right, prefer df_left values and fill missing with right
    for col in right_cols:
        right_col_r = col + '_r'
        if col in df_left.columns:
            # If left has column, fillna from right and drop helper
            if right_col_r in merged.columns:
                merged[col] = merged[col].fillna(merged[right_col_r])
                merged.drop(columns=[right_col_r], inplace=True, errors='ignore')
        else:
            # left doesn't have column: rename the imported right_col_r to col (or keep col if no suffix)
            if col in merged.columns and right_col_r in merged.columns:
                # unlikely but handle by filling then dropping suffix
                merged[col] = merged[col].fillna(merged[right_col_r])
                merged.drop(columns=[right_col_r], inplace=True, errors='ignore')
            elif right_col_r in merged.columns:
                merged.rename(columns={right_col_r: col}, inplace=True)

    # cleanup helper and duplicate country columns
    merged.drop(
        columns=['_fuzzy_key'] + [c for c in merged.columns if c.startswith('Country_clean_r')],
        inplace=True,
        errors='ignore'
    )

    if return_matches:
        return merged, mapping
    return merged

# download happiness + lots of other tables | store the csvs to not re-download data 
def download_datasets():
    data_dir = "datasets"
    os.makedirs(data_dir, exist_ok=True)
    downloaded_datasets = {}

    # World Happiness
    happiness_download_path = kagglehub.dataset_download("jainaru/world-happiness-report-2024-yearly-updated")
    happiness_local_path = os.path.join(data_dir, "World-happiness-report-2024.csv")
    if not os.path.exists(happiness_local_path):
        happiness_df = pd.read_csv(f"{happiness_download_path}/World-happiness-report-2024.csv")
        happiness_df.to_csv(happiness_local_path, index=False)
        happiness_df["Country_clean"] = happiness_df["Country name"].map(normalize_country).map(apply_alias)
        downloaded_datasets['happiness_df'] = happiness_df
    else:
        happiness_df = pd.read_csv(f"{happiness_local_path}")
        happiness_df["Country_clean"] = happiness_df["Country name"].map(normalize_country).map(apply_alias)
        downloaded_datasets['happiness_df'] = happiness_df

    # Average Wage
    average_wage_download_path = kagglehub.dataset_download("kabhishm/countries-by-average-wage")
    average_wage_local_path = os.path.join(data_dir, "avg_wage.csv")
    if not os.path.exists(average_wage_local_path):
        average_wage_df = pd.read_csv(f"{average_wage_download_path}/avg_wage.csv")
        average_wage_df.to_csv(average_wage_local_path, index=False)
        downloaded_datasets['average_wage_df'] = average_wage_df
    else:
        average_wage_df = pd.read_csv(f"{average_wage_local_path}")
        downloaded_datasets['average_wage_df'] = average_wage_df

    # IQ Air
    iq_air_download_path = kagglehub.dataset_download("ramjasmaurya/most-polluted-cities-and-countries-iqair-index")
    iq_air_local_path = os.path.join(data_dir, "AIR QUALITY INDEX (by cities) - IQAir.csv")
    if not os.path.exists(iq_air_local_path):
        iq_air_df = pd.read_csv(f"{iq_air_download_path}/AIR QUALITY INDEX (by cities) - IQAir.csv")
        iq_air_df.to_csv(iq_air_local_path, index=False)
        downloaded_datasets['iq_air_df'] = iq_air_df
    else:
        iq_air_df = pd.read_csv(f"{iq_air_local_path}")
        downloaded_datasets['iq_air_df'] = iq_air_df

    # Life Expectancy
    life_expectancy_download_path = kagglehub.dataset_download("amirhosseinmirzaie/countries-life-expectancy")
    life_expectancy_local_path = os.path.join(data_dir, "life_expectancy.csv")
    if not os.path.exists(life_expectancy_local_path):
        life_expectancy_df = pd.read_csv(f"{life_expectancy_download_path}/life_expectancy.csv")
        life_expectancy_df.to_csv(life_expectancy_local_path, index=False)
        downloaded_datasets['life_expectancy_df'] = life_expectancy_df
    else:
        life_expectancy_df = pd.read_csv(f"{life_expectancy_local_path}")
        downloaded_datasets['life_expectancy_df'] = life_expectancy_df

    # Netflix
    netflix_download_path = kagglehub.dataset_download("prasertk/netflix-subscription-price-in-different-countries")
    netflix_local_path = os.path.join(data_dir, "Netflix subscription fee Dec-2021.csv")
    if not os.path.exists(netflix_local_path):
        netflix_data_df = pd.read_csv(f"{netflix_download_path}/Netflix subscription fee Dec-2021.csv")
        netflix_data_df.to_csv(netflix_local_path, index=False)
        downloaded_datasets['netflix_data_df'] = netflix_data_df
    else:
        netflix_data_df = pd.read_csv(f"{netflix_local_path}")
        downloaded_datasets['netflix_data_df'] = netflix_data_df

    # Women Safety
    women_safety_download_path = kagglehub.dataset_download("arpitsinghaiml/most-dangerous-countries-for-women-2024")
    women_safety_local_path = os.path.join(data_dir, "most-dangerous-countries-for-women-2024.csv")
    if not os.path.exists(women_safety_local_path):
        women_safety_df = pd.read_csv(f"{women_safety_download_path}/most-dangerous-countries-for-women-2024.csv")
        women_safety_df.to_csv(women_safety_local_path, index=False)
        downloaded_datasets['women_safety_df'] = women_safety_df
    else:
        women_safety_df = pd.read_csv(f"{women_safety_local_path}")
        downloaded_datasets['women_safety_df'] = women_safety_df

    # Temperature
    temperature_download_path = kagglehub.dataset_download("samithsachidanandan/average-monthly-surface-temperature-1940-2024")
    temperature_local_path = os.path.join(data_dir, "average-monthly-surface-temperature.csv")
    if not os.path.exists(temperature_local_path):
        temperature_df = pd.read_csv(f"{temperature_download_path}/average-monthly-surface-temperature.csv")
        temperature_df.to_csv(temperature_local_path, index=False)
        downloaded_datasets['temperature_df'] = temperature_df
    else:
        temperature_df = pd.read_csv(f"{temperature_local_path}")
        downloaded_datasets['temperature_df'] = temperature_df

    # Population
    population_download_path = kagglehub.dataset_download("iamsouravbanerjee/world-population-dataset")
    population_local_path = os.path.join(data_dir, "world_population.csv")
    if not os.path.exists(population_local_path):
        population_df = pd.read_csv(f"{population_download_path}/world_population.csv")
        population_df.to_csv(population_local_path, index=False)
        downloaded_datasets['population_df'] = population_df
    else:
        population_df = pd.read_csv(f"{population_local_path}")
        downloaded_datasets['population_df'] = population_df

    # Energy Consumption
    energy_consumption_download_path = kagglehub.dataset_download("pralabhpoudel/world-energy-consumption")
    energy_consumption_local_path = os.path.join(data_dir, "World Energy Consumption.csv")
    if not os.path.exists(energy_consumption_local_path):
        energy_consumption_df = pd.read_csv(f"{energy_consumption_download_path}/World Energy Consumption.csv")
        energy_consumption_df.to_csv(energy_consumption_local_path, index=False)
        downloaded_datasets['energy_consumption_df'] = energy_consumption_df
    else:
        energy_consumption_df = pd.read_csv(f"{energy_consumption_local_path}")
        downloaded_datasets['energy_consumption_df'] = energy_consumption_df

    # World Bank Development
    world_bank_download_path = kagglehub.dataset_download("nicolasgonzalezmunoz/world-bank-world-development-indicators")
    world_bank_local_path = os.path.join(data_dir, "world_bank_development_indicators.csv")
    if not os.path.exists(world_bank_local_path):
        world_bank_development_df = pd.read_csv(f"{world_bank_download_path}/world_bank_development_indicators.csv")
        world_bank_development_df.to_csv(world_bank_local_path, index=False)
        downloaded_datasets['world_bank_development_df'] = world_bank_development_df
    else:
        world_bank_development_df = pd.read_csv(f"{world_bank_local_path}")
        downloaded_datasets['world_bank_development_df'] = world_bank_development_df

    # Food Production
    food_production_download_path = kagglehub.dataset_download("rafsunahmad/world-food-production")
    food_production_local_path = os.path.join(data_dir, "world food production.csv")
    if not os.path.exists(food_production_local_path):
        food_production_df = pd.read_csv(f"{food_production_download_path}/world food production.csv")
        food_production_df.to_csv(food_production_local_path, index=False)
        downloaded_datasets['food_production_df'] = food_production_df
    else:
        food_production_df = pd.read_csv(f"{food_production_local_path}")
        downloaded_datasets['food_production_df'] = food_production_df

    # Petrol Prices
    petrol_prices_download_path = kagglehub.dataset_download("zusmani/petrolgas-prices-worldwide")
    petrol_prices_local_path = os.path.join(data_dir, "Petrol Dataset June 20 2022.csv")
    if not os.path.exists(petrol_prices_local_path):
        petrol_prices_df = pd.read_csv(f"{petrol_prices_download_path}/Petrol Dataset June 20 2022.csv", encoding='latin1')
        petrol_prices_df.to_csv(petrol_prices_local_path, index=False)
        downloaded_datasets['petrol_prices_df'] = petrol_prices_df
    else:
        petrol_prices_df = pd.read_csv(f"{petrol_prices_local_path}")
        downloaded_datasets['petrol_prices_df'] = petrol_prices_df

    # CO2 Emissions
    co2_emissions_download_path = kagglehub.dataset_download("koustavghosh149/co2-emission-around-the-world")
    co2_emissions_local_path = os.path.join(data_dir, "CO2_emission.csv")
    if not os.path.exists(co2_emissions_local_path):
        co2_emissions_df = pd.read_csv(f"{co2_emissions_download_path}/CO2_emission.csv")
        co2_emissions_df.to_csv(co2_emissions_local_path, index=False)
        downloaded_datasets['co2_emissions_df'] = co2_emissions_df
    else:
        co2_emissions_df = pd.read_csv(f"{co2_emissions_local_path}")
        downloaded_datasets['co2_emissions_df'] = co2_emissions_df

    # Return all dataframes
    return downloaded_datasets

# checks for csvs and re-download data if csvs are not available
data = download_datasets()
happiness_df = data['happiness_df'][['Country name', 'Country_clean', 'Regional indicator', 'Ladder score']].rename(columns={"Country name": "Country", "Regional indicator" : "Region", "Ladder score" : "Happiness"})

# isolate the country name + column that I need from supporting datasets
# rename the column
# then clean each one
# then merge each one into happiness_df

# 1) Average Wage - simple [but mind not full matches, also count them]
average_wage_isolated_df = data['average_wage_df'][['Country', '2020']] 
# Note for above: DOUBLE SQUARE BRACKETS!!! 1st brackets are the INDEX selector, 2nd brackets are the LIST of columns we pass
average_wage_isolated_df = average_wage_isolated_df.rename(columns={'2020': 'Average Wage (USD/year)'})
average_wage_isolated_df = average_wage_isolated_df.dropna().drop_duplicates()
average_wage_isolated_df['Country'] = average_wage_isolated_df['Country'].str.strip()
average_wage_isolated_df['Average Wage (USD/year)'] = pd.to_numeric(average_wage_isolated_df['Average Wage (USD/year)'], errors='coerce')
average_wage_isolated_df["Country_clean"] = average_wage_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    average_wage_isolated_df,
    left_on='Country_clean',
    right_on='Country_clean',
    right_cols=['Average Wage (USD/year)'],
    threshold=85
)

# # 2) IQ Air - need to sort cities into countries

# 3) Lifespan - need to only grab rows with the latest year
life_expectancy_isolated_df = data['life_expectancy_df'][['Country', 'Year', 'infant deaths', 'Alcohol', 'Life expectancy']]
life_expectancy_isolated_df = life_expectancy_isolated_df.rename(columns={'infant deaths': "Infant Death Rate", 'Alcohol': 'Alcohol Consumption Rate', 'Life expectancy' : 'Life Expectancy'})
life_expectancy_isolated_df = life_expectancy_isolated_df.dropna().drop_duplicates()
life_expectancy_isolated_df['Country'] = life_expectancy_isolated_df['Country'].str.strip()
life_expectancy_isolated_df['Infant Death Rate'] = pd.to_numeric(life_expectancy_isolated_df['Infant Death Rate'], errors='coerce')
life_expectancy_isolated_df['Infant Survival Rate'] = 1000 - life_expectancy_isolated_df['Infant Death Rate'] 
# Note for above: PD applies things ELEMENT-WIDE!!! so: every row in gets transformed into "survival rate" as 1000 - value.
life_expectancy_isolated_df['Alcohol Consumption Rate'] = pd.to_numeric(life_expectancy_isolated_df['Alcohol Consumption Rate'], errors='coerce')
life_expectancy_isolated_df['Life Expectancy'] = pd.to_numeric(life_expectancy_isolated_df['Life Expectancy'], errors='coerce')
life_expectancy_isolated_df = life_expectancy_isolated_df[life_expectancy_isolated_df["Year"] == 2015]
life_expectancy_isolated_df["Country_clean"] = life_expectancy_isolated_df["Country"].map(normalize_country).map(apply_alias)
happiness_df = fuzzy_merge(
    happiness_df,
    life_expectancy_isolated_df,
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
energy_isolated_df['Country'] = energy_isolated_df['Country'].str.strip()
energy_isolated_df['Energy Per Capita'] = pd.to_numeric(energy_isolated_df['Energy Per Capita'], errors='coerce')
energy_isolated_df['% of Power from Fossil Fuels'] = pd.to_numeric(energy_isolated_df['% of Power from Fossil Fuels'], errors='coerce')
energy_isolated_df['% of Power from Nuclear'] = pd.to_numeric(energy_isolated_df['% of Power from Nuclear'], errors='coerce')
energy_isolated_df['% of Power from Renewables'] = pd.to_numeric(energy_isolated_df['% of Power from Renewables'], errors='coerce')
energy_isolated_df["Country_clean"] = energy_isolated_df["Country"].map(normalize_country).map(apply_alias)
energy_isolated_df = energy_isolated_df[energy_isolated_df["year"] == 2022]
happiness_df = fuzzy_merge(
    happiness_df,
    energy_isolated_df,
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
food_production_isolated_df = food_production_isolated_df[food_production_isolated_df["Year"] == 2021]
happiness_df = fuzzy_merge(
    happiness_df,
    food_production_isolated_df,
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

# add more tables / metrics for more opportunities to find correlations:
# GDP per capita: https://www.kaggle.com/datasets/nitishabharathi/gdp-per-capita-all-countries
# Military spending: https://www.kaggle.com/datasets/nitinsss/military-expenditure-of-countries-19602019
# Homicide rate: https://www.kaggle.com/datasets/bilalwaseer/countries-by-intentional-homicide-rate
# Geography variables: https://www.kaggle.com/datasets/zanderventer/environmental-variables-for-world-countries?select=World_countries_env_vars.csv
# Average age: https://www.kaggle.com/datasets/divyansh22/average-age-of-countries
# Inflation: https://www.kaggle.com/datasets/meeratif/inflation-2022

# post on kaggle
# add to portfolio
# post about it on linked in