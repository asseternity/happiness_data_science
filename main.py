# Project: what makes a land happy?

# set up the environment
import kagglehub
import pandas as pd
import pandasql
import os
import seaborn as sns

# download happiness + lots of other tables | store the csvs to not re-download data 
def download_datasets():
    data_dir = "datasets"
    os.makedirs(data_dir, exist_ok=True)
    downloaded_datasets = []

    # World Happiness
    happiness_download_path = kagglehub.dataset_download("jainaru/world-happiness-report-2024-yearly-updated")
    happiness_local_path = os.path.join(data_dir, "World-happiness-report-2024.csv")
    if not os.path.exists(happiness_local_path):
        happiness_df = pd.read_csv(f"{happiness_download_path}/World-happiness-report-2024.csv")
        happiness_df.to_csv(happiness_local_path, index=False)
        downloaded_datasets.append(happiness_df)
    else:
        happiness_df = pd.read_csv(f"{happiness_local_path}")
        downloaded_datasets.append(happiness_df)

    # Average Wage
    average_wage_download_path = kagglehub.dataset_download("kabhishm/countries-by-average-wage")
    average_wage_local_path = os.path.join(data_dir, "avg_wage.csv")
    if not os.path.exists(average_wage_local_path):
        average_wage_df = pd.read_csv(f"{average_wage_download_path}/avg_wage.csv")
        average_wage_df.to_csv(average_wage_local_path, index=False)
        downloaded_datasets.append(average_wage_df)
    else:
        average_wage_df = pd.read_csv(f"{average_wage_local_path}")
        downloaded_datasets.append(average_wage_df)

    # IQ Air
    iq_air_download_path = kagglehub.dataset_download("ramjasmaurya/most-polluted-cities-and-countries-iqair-index")
    iq_air_local_path = os.path.join(data_dir, "AIR QUALITY INDEX (by cities) - IQAir.csv")
    if not os.path.exists(iq_air_local_path):
        iq_air_df = pd.read_csv(f"{iq_air_download_path}/AIR QUALITY INDEX (by cities) - IQAir.csv")
        iq_air_df.to_csv(iq_air_local_path, index=False)
        downloaded_datasets.append(iq_air_df)
    else:
        iq_air_df = pd.read_csv(f"{iq_air_local_path}")
        downloaded_datasets.append(iq_air_df)

    # Lifespan
    lifespan_download_path = kagglehub.dataset_download("amirhosseinmirzaie/countries-life-expectancy")
    lifespan_local_path = os.path.join(data_dir, "life_expectancy.csv")
    if not os.path.exists(lifespan_local_path):
        lifespan_df = pd.read_csv(f"{lifespan_download_path}/life_expectancy.csv")
        lifespan_df.to_csv(lifespan_local_path, index=False)
        downloaded_datasets.append(lifespan_df)
    else:
        lifespan_df = pd.read_csv(f"{lifespan_local_path}")
        downloaded_datasets.append(lifespan_df)

    # Netflix
    netflix_download_path = kagglehub.dataset_download("prasertk/netflix-subscription-price-in-different-countries")
    netflix_local_path = os.path.join(data_dir, "Netflix subscription fee Dec-2021.csv")
    if not os.path.exists(netflix_local_path):
        netflix_data_df = pd.read_csv(f"{netflix_download_path}/Netflix subscription fee Dec-2021.csv")
        netflix_data_df.to_csv(netflix_local_path, index=False)
        downloaded_datasets.append(netflix_data_df)
    else:
        netflix_data_df = pd.read_csv(f"{netflix_local_path}")
        downloaded_datasets.append(netflix_data_df)

    # Women Safety
    women_safety_download_path = kagglehub.dataset_download("arpitsinghaiml/most-dangerous-countries-for-women-2024")
    women_safety_local_path = os.path.join(data_dir, "most-dangerous-countries-for-women-2024.csv")
    if not os.path.exists(women_safety_local_path):
        women_safety_df = pd.read_csv(f"{women_safety_download_path}/most-dangerous-countries-for-women-2024.csv")
        women_safety_df.to_csv(women_safety_local_path, index=False)
        downloaded_datasets.append(women_safety_df)
    else:
        women_safety_df = pd.read_csv(f"{women_safety_local_path}")
        downloaded_datasets.append(women_safety_df)

    # Temperature
    temperature_download_path = kagglehub.dataset_download("samithsachidanandan/average-monthly-surface-temperature-1940-2024")
    temperature_local_path = os.path.join(data_dir, "average-monthly-surface-temperature.csv")
    if not os.path.exists(temperature_local_path):
        temperature_df = pd.read_csv(f"{temperature_download_path}/average-monthly-surface-temperature.csv")
        temperature_df.to_csv(temperature_local_path, index=False)
        downloaded_datasets.append(temperature_df)
    else:
        temperature_df = pd.read_csv(f"{temperature_local_path}")
        downloaded_datasets.append(temperature_df)

    # Population
    population_download_path = kagglehub.dataset_download("iamsouravbanerjee/world-population-dataset")
    population_local_path = os.path.join(data_dir, "world_population.csv")
    if not os.path.exists(population_local_path):
        population_df = pd.read_csv(f"{population_download_path}/world_population.csv")
        population_df.to_csv(population_local_path, index=False)
        downloaded_datasets.append(population_df)
    else:
        population_df = pd.read_csv(f"{population_local_path}")
        downloaded_datasets.append(population_df)

    # Energy Consumption
    energy_consumption_download_path = kagglehub.dataset_download("pralabhpoudel/world-energy-consumption")
    energy_consumption_local_path = os.path.join(data_dir, "World Energy Consumption.csv")
    if not os.path.exists(energy_consumption_local_path):
        energy_consumption_df = pd.read_csv(f"{energy_consumption_download_path}/World Energy Consumption.csv")
        energy_consumption_df.to_csv(energy_consumption_local_path, index=False)
        downloaded_datasets.append(energy_consumption_df)
    else:
        energy_consumption_df = pd.read_csv(f"{energy_consumption_local_path}")
        downloaded_datasets.append(energy_consumption_df)

    # World Bank Development
    world_bank_download_path = kagglehub.dataset_download("nicolasgonzalezmunoz/world-bank-world-development-indicators")
    world_bank_local_path = os.path.join(data_dir, "world_bank_development_indicators.csv")
    if not os.path.exists(world_bank_local_path):
        world_bank_development_df = pd.read_csv(f"{world_bank_download_path}/world_bank_development_indicators.csv")
        world_bank_development_df.to_csv(world_bank_local_path, index=False)
        downloaded_datasets.append(world_bank_development_df)
    else:
        world_bank_development_df = pd.read_csv(f"{world_bank_local_path}")
        downloaded_datasets.append(world_bank_development_df)

    # Food Production
    food_production_download_path = kagglehub.dataset_download("rafsunahmad/world-food-production")
    food_production_local_path = os.path.join(data_dir, "world food production.csv")
    if not os.path.exists(food_production_local_path):
        food_production_df = pd.read_csv(f"{food_production_download_path}/world food production.csv")
        food_production_df.to_csv(food_production_local_path, index=False)
        downloaded_datasets.append(food_production_df)
    else:
        food_production_df = pd.read_csv(f"{food_production_local_path}")
        downloaded_datasets.append(food_production_df)

    # Petrol Prices
    petrol_prices_download_path = kagglehub.dataset_download("zusmani/petrolgas-prices-worldwide")
    petrol_prices_local_path = os.path.join(data_dir, "Petrol Dataset June 20 2022.csv")
    if not os.path.exists(petrol_prices_local_path):
        petrol_prices_df = pd.read_csv(f"{petrol_prices_download_path}/Petrol Dataset June 20 2022.csv", encoding='latin1')
        petrol_prices_df.to_csv(petrol_prices_local_path, index=False)
        downloaded_datasets.append(petrol_prices_df)
    else:
        petrol_prices_df = pd.read_csv(f"{petrol_prices_local_path}")
        downloaded_datasets.append(petrol_prices_df)

    # CO2 Emissions
    co2_emissions_download_path = kagglehub.dataset_download("koustavghosh149/co2-emission-around-the-world")
    co2_emissions_local_path = os.path.join(data_dir, "CO2_emission.csv")
    if not os.path.exists(co2_emissions_local_path):
        co2_emissions_df = pd.read_csv(f"{co2_emissions_download_path}/CO2_emission.csv")
        co2_emissions_df.to_csv(co2_emissions_local_path, index=False)
        downloaded_datasets.append(co2_emissions_df)
    else:
        co2_emissions_df = pd.read_csv(f"{co2_emissions_local_path}")
        downloaded_datasets.append(co2_emissions_df)

    # Return all dataframes
    return downloaded_datasets

# checks for csvs and re-download data if csvs are not available
data = download_datasets()
happiness_df = data.happiness_df.rename(columns={"Country name": "Country"})

# isolate the country name + column that I need from supporting datasets
# rename the column
# then clean each one
# then merge each one into happiness_df

# 1) Average Wage
average_wage_isolated_df = data.average_wage_df[['Country', '2020']]
average_wage_isolated_df = average_wage_isolated_df.rename(columns={'2020': 'Average_Wage'})
average_wage_isolated_df = average_wage_isolated_df.dropna().drop_duplicates()
average_wage_isolated_df['Country'] = average_wage_isolated_df['Country'].str.strip()
average_wage_isolated_df['Average_Wage'] = pd.to_numeric(average_wage_isolated_df['Average_Wage'], errors='coerce')
happiness_df = pd.merge(happiness_df, average_wage_isolated_df, on="Country", how="left")

# 2) IQ Air

# 3) Lifespan

# 4) Netflix

# 5) Temperature

# 6) Population

# 7) Energy Consumption

# 8) World Bank Development

# 9) Food Production

# 10) Petrol Prices

# 11) CO2 Emissions

# use AI to find which metrics correlate to happiness and which don't


# add more tables / metrics for more opportunities to find correlations
