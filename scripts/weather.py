# Imports.
import requests
import polars as pl

pl.Config(tbl_cols = 50, tbl_rows = 60, set_fmt_str_lengths = 150)

# Here is a dictionary of NFL stadium coordinates.
stadium_coordinates = {
    'ARI': '33.5277,-112.2626',
    'ATL': '33.7576,-84.4009',
    'NE': '42.0909,-71.2643',
    'BAL': '39.2779,-76.6227',
    'BUF': '42.7737,-78.7869',
    'CAR': '35.2258,-80.8528',
    'CHI': '41.8623,-87.6166',
    'CIN': '39.0954,-84.5160',
    'CLE': '41.5060,-81.6995',
    'DAL': '32.7477,-97.0927',
    'DEN': '39.7439,-105.0200',
    'DET': '42.3401,-83.0458',
    'GB': '44.5013,-88.0621',
    'HOU': '29.6847,-95.4109',
    'IND': '39.7600,-86.1638',
    'JAX': '30.3239,-81.6373',
    'KC': '39.0489,-94.4840',
    'LV': '36.0907,-115.1839',
    'LA': '33.9535,-118.3390',
    'LAC': '33.9535,-118.3390',
    'MIA': '25.9579,-80.2388',
    'MIN': '44.9738,-93.2580',
    'NE': '42.0909,-71.2643',
    'NO': '29.9509,-90.0813',
    'NYG': '40.8121,-74.0769',
    'NYJ': '40.8121,-74.0769',
    'PHI': '39.9007,-75.1674',
    'PIT': '40.4467,-80.0157',
    'SF': '37.7134,-122.3862',
    'SEA': '47.5951,-122.3316',
    'TB': '27.9759,-82.5033',
    'TEN': '36.1664,-86.7712',
    'WAS': '38.9076,-76.8645',
}

# Let's write a class to retrieve and process weather data to be used for prediction.
class NFLWeather():
    """
    A class to retrieve and process weather data from the National Weather Service (NWS) for use in NFL event prediction.

    Methods:
        get_weather_data(team_abbr, lat_lon): Retrieves hourly weather forecast JSON data from the NWS API using a latitude-longitude coordinate.

        process_weather_json(team_abbr, weather_json): Processes the weather JSON returned by the NWS API and returns a cleaned Polars DataFrame.
    """

    def get_weather_data(team_abbr: str, lat_lon: str):
        """
        Retrieves weather forecast data from the NWS API based on geographical coordinates.

        Args:
            team_abbr (str): Abbreviation of the NFL team, used for labeling or error messages.
            lat_lon (str): A string of the form "lat,lon" representing the location.

        Returns:
            dict or None: Parsed JSON containing hourly forecast data if successful; otherwise, None.
        """
        
        # Here is the URL to access the NWS API
        url = f'https://api.weather.gov/points/{lat_lon}'
        response = requests.get(url)
        if response.status_code != 200:
            print(f'Error querying NWS API for {team_abbr}.')
        
        # Get the hourly forecast URL from the response
        forecast_url = response.json()['properties']['forecastHourly']
        
        # Fetch the forecast data.
        forecast_response = requests.get(forecast_url)
        if forecast_response.status_code != 200:
            print(f"Failed to get forecast data for {team_abbr}")
            return None
        
        forecast_data = forecast_response.json()
        return forecast_data

    def process_weather_json(team_abbr: str, weather_json: dict):
        """
        Converts NWS API forecast JSON into a cleaned Polars DataFrame with selected weather features.

        Args:
            team_abbr (str): Abbreviation of the NFL team, used to tag each weather record.
            weather_json (dict): Raw JSON response from the NWS API's hourly forecast endpoint.

        Returns:
            pl.DataFrame: A Polars DataFrame with columns: 'team', 'date', 'time', 'temperature', 'wind_speed', and 'humidity'.
        """
        
        # Grab the list of hourly forecasts and extract the relevant data
        hourly_forecast = weather_json['properties']['periods']
    
        # Create a Polars DataFrame from the relevant columns
        df = pl.DataFrame({
            'team': team_abbr,
            'start_time': [hour['startTime'] for hour in hourly_forecast],
            'temperature': [hour['temperature'] for hour in hourly_forecast],
            'wind_speed': [hour['windSpeed'] for hour in hourly_forecast],
            'humidity': [hour['relativeHumidity'] for hour in hourly_forecast]
        })

        # Clean them
        df = (
            df
            .with_columns(
                pl.col('wind_speed').str.extract(r'(\d+)').cast(pl.Int64).alias('wind_speed'),
                pl.col('humidity').struct.json_encode().str.extract(r'"value":(\d+)', 1).cast(pl.Int64).alias('humidity'),
                pl.col("start_time").str.extract(r'(\d{4}-\d{2}-\d{2})', 1).alias("date"),
                pl.col("start_time").str.extract(r'T(\d{2}:\d{2})', 1).alias("time")
            )
            .select('team', 'date', 'time', 'temperature', 'wind_speed', 'humidity')
        )
    
        return df
    
# Generate a dataframe containing week-ahead hourly weather forecasts for every NFL stadium.
nfl_weather_forecast_schema = {'team': str, 'date': str, 'time': str, 'temperature': pl.Int64, 'wind_speed': pl.Int64, 'humidity': pl.Int64}
week_ahead_nfl_weather_forecast = pl.DataFrame(schema = nfl_weather_forecast_schema)
    
for team, lat_lon in stadium_coordinates.items():
    weather_data = NFLWeather.get_weather_data(team, lat_lon)
    weather_data_clean_df = NFLWeather.process_weather_json(team, weather_data)
    week_ahead_nfl_weather_forecast = pl.concat([week_ahead_nfl_weather_forecast, weather_data_clean_df])