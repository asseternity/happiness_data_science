import kagglehub
import pandas as pd
import os

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
    average_wage_download_path = kagglehub.dataset_download("zedataweaver/global-salary-data")
    average_wage_local_path = os.path.join(data_dir, "salary_data.csv")
    if not os.path.exists(average_wage_local_path):
        average_wage_df = pd.read_csv(f"{average_wage_download_path}/salary_data.csv")
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