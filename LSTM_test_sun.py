import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from diskcache import Cache


cache = Cache('cache2023')

##################################
###        Data reading        ###
##################################

@cache.memoize()
def file_reading():
    # read 2023 data
    NUMBAT23_fri = pd.read_excel("Data/NUMBAT/NUMBAT_2023/NBT23FRI_Outputs_test.xlsx", sheet_name=None)
    NUMBAT23_mon = pd.read_excel("Data/NUMBAT/NUMBAT_2023/NBT23MON_Outputs_test.xlsx", sheet_name=None)
    NUMBAT23_twt = pd.read_excel("Data/NUMBAT/NUMBAT_2023/NBT23TWT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT23_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2023/NBT23SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT23_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2023/NBT23SUN_Outputs_test.xlsx", sheet_name=None)

    # read 2022 data
    NUMBAT22_fri = pd.read_excel("Data/NUMBAT/NUMBAT_2022/NBT22FRI_Outputs_test.xlsx", sheet_name=None)
    NUMBAT22_mon = pd.read_excel("Data/NUMBAT/NUMBAT_2022/NBT22MON_Outputs_test.xlsx", sheet_name=None)
    NUMBAT22_twt = pd.read_excel("Data/NUMBAT/NUMBAT_2022/NBT22TWT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT22_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2022/NBT22SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT22_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2022/NBT22SUN_Outputs_test.xlsx", sheet_name=None)

    # read 2019 data
    NUMBAT19_fri = pd.read_excel("Data/NUMBAT/NUMBAT_2019/NBT19FRI_Outputs_test.xlsx", sheet_name=None)
    NUMBAT19_mtt = pd.read_excel("Data/NUMBAT/NUMBAT_2019/NBT19MTT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT19_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2019/NBT19SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT19_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2019/NBT19SUN_Outputs_test.xlsx", sheet_name=None)

    # read 2018 data
    NUMBAT18_fri = pd.read_excel("Data/NUMBAT/NUMBAT_2018/NBT18FRI_Outputs_test.xlsx", sheet_name=None)
    NUMBAT18_mtt = pd.read_excel("Data/NUMBAT/NUMBAT_2018/NBT18MTT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT18_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2018/NBT18SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT18_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2018/NBT18SUN_Outputs_test.xlsx", sheet_name=None)

    # read 2017 data
    NUMBAT17_mtt = pd.read_excel("Data/NUMBAT/NUMBAT_2017/NBT17MTT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT17_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2017/NBT17SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT17_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2017/NBT17SUN_Outputs_test.xlsx", sheet_name=None)

    # read 2016 data
    NUMBAT16_mtt = pd.read_excel("Data/NUMBAT/NUMBAT_2016/NBT16MTT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT16_sat = pd.read_excel("Data/NUMBAT/NUMBAT_2016/NBT16SAT_Outputs_test.xlsx", sheet_name=None)
    NUMBAT16_sun = pd.read_excel("Data/NUMBAT/NUMBAT_2016/NBT16SUN_Outputs_test.xlsx", sheet_name=None)
    return NUMBAT23_fri, NUMBAT23_mon, NUMBAT23_twt, NUMBAT23_sat, NUMBAT23_sun, NUMBAT22_fri, NUMBAT22_mon, NUMBAT22_twt, NUMBAT22_sat, NUMBAT22_sun, NUMBAT19_fri, NUMBAT19_mtt, NUMBAT19_sat, NUMBAT19_sun, NUMBAT18_fri, NUMBAT18_mtt, NUMBAT18_sat, NUMBAT18_sun, NUMBAT17_mtt, NUMBAT17_sat, NUMBAT17_sun, NUMBAT16_mtt, NUMBAT16_sat, NUMBAT16_sun

NUMBAT23_fri, NUMBAT23_mon, NUMBAT23_twt, NUMBAT23_sat, NUMBAT23_sun, NUMBAT22_fri, NUMBAT22_mon, NUMBAT22_twt, NUMBAT22_sat, NUMBAT22_sun, NUMBAT19_fri, NUMBAT19_mtt, NUMBAT19_sat, NUMBAT19_sun, NUMBAT18_fri, NUMBAT18_mtt, NUMBAT18_sat, NUMBAT18_sun, NUMBAT17_mtt, NUMBAT17_sat, NUMBAT17_sun, NUMBAT16_mtt, NUMBAT16_sat, NUMBAT16_sun = file_reading()

print("Info of 23 sunday: ")
print(NUMBAT23_sun)
#########################################
###   example 2 NUMBAT data reading   ###
###   flows and service frequencies   ###
#########################################

def flows_brief(raw_data):
    flows_data = raw_data["Link_Loads"]
    flows_data_brief = flows_data.iloc[:, 0:17]  # Assuming the first 17 columns are relevant
    return flows_data_brief

# Load data for all years
NUMBAT23_sun_flows_brief = flows_brief(NUMBAT23_sun)
NUMBAT22_sun_flows_brief = flows_brief(NUMBAT22_sun)
NUMBAT19_sun_flows_brief = flows_brief(NUMBAT19_sun)
NUMBAT18_sun_flows_brief = flows_brief(NUMBAT18_sun)
NUMBAT17_sun_flows_brief = flows_brief(NUMBAT17_sun)
NUMBAT16_sun_flows_brief = flows_brief(NUMBAT16_sun)

print("NUMBAT23: brief")
print(NUMBAT23_sun_flows_brief.info())
print("NUMBAT22: brief")
print(NUMBAT22_sun_flows_brief.info())
print("NUMBAT19: brief")
print(NUMBAT19_sun_flows_brief.info())
print("NUMBAT18: brief")
print(NUMBAT18_sun_flows_brief.info())
print("NUMBAT17: brief")
print(NUMBAT17_sun_flows_brief.info())
print("NUMBAT16: brief")
print(NUMBAT16_sun_flows_brief.info())


columns_to_select = ['Link', 'Early', 'AM Peak', 'Midday', 'PM Peak', 'Evening', 'Late']

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Function to process a batch of links and make predictions
def process_batch(links_batch, seq_length=1):
    predictions = {}
    
    for link in links_batch:
        link_data = []
        
        # Collecting data for each year
        for year, data in zip([2016, 2017, 2018, 2019, 2022], 
                              [NUMBAT16_sun_flows_brief, NUMBAT17_sun_flows_brief, NUMBAT18_sun_flows_brief, NUMBAT19_sun_flows_brief, NUMBAT22_sun_flows_brief]):
            if link in data['Link'].values:
                instance = data[data['Link'] == link][columns_to_select].copy()
                instance['Year'] = year
                link_data.append(instance)
        
        if not link_data:
            continue  # Skip if no data for the link
        
        # Combine yearly data
        combined_instance = pd.concat(link_data)
        combined_instance.reset_index(drop=True, inplace=True)
        combined_instance = combined_instance.sort_values('Year')

        # Drop 'Year' and 'Link' columns and normalize data
        data = combined_instance.drop(['Year', 'Link'], axis=1).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Create sequences for LSTM
        X, y = create_sequences(data_scaled, seq_length)
        
        if len(X) == 0 or len(y) == 0:
            continue  # Skip if not enough data to create sequences

        X = X.reshape((X.shape[0], seq_length, X.shape[2]))

        # Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(seq_length, X.shape[2])))
        model.add(Dense(X.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=1, verbose=0)  # Adjust epochs as needed

        # Predict 2023 data
        last_sequence = data_scaled[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, X.shape[2]))
        predicted_2023 = model.predict(last_sequence)
        predicted_2023 = scaler.inverse_transform(predicted_2023)

        predicted_2023[predicted_2023 < 1] = 0

        predictions[link] = predicted_2023

    return predictions

# Parallel processing to handle the links
def parallel_process(links, num_processes=4, seq_length=1):
    pool = mp.Pool(num_processes)
    batch_size = len(links) // num_processes
    links_batches = [links[i:i + batch_size] for i in range(0, len(links), batch_size)]
    results = pool.map(process_batch, links_batches)
    pool.close()
    pool.join()
    
    # Combine all predictions
    all_predictions = {}
    for result in results:
        all_predictions.update(result)
    
    return all_predictions

def prepare_time_period_data(predictions, selected_links):
    # Initialize a dictionary to hold data for all time periods for each link
    time_period_data = {link: [] for link in selected_links}
    
    for link in selected_links:
        # Append all time period values to the list for this link
        time_period_data[link] = predictions[link].flatten()
    
    return time_period_data

def create_correlation_matrix(time_period_data):
    # Convert the time period data dictionary into a DataFrame
    # Each key (link) will be a column, and each row will represent a time period value
    time_period_df = pd.DataFrame(time_period_data)
    print("Time period DataFrame:")
    print(time_period_df)
    
    # Calculate the correlation matrix across the columns (links)
    correlation_matrix = time_period_df.corr()
    
    return correlation_matrix

# Function to plot the predictions against historical and actual 2023 data
def plot_predictions(predictions, actual_data, historical_data, selected_links):
    time_periods = ['Early', 'AM Peak', 'Midday', 'PM Peak', 'Evening', 'Late']
    
    for link in selected_links:
        plt.figure(figsize=(10, 6))
        
        # Initialize a dictionary to store the data for each time period
        plot_data = {period: [] for period in time_periods}
        years = []

        # Collect historical data
        for year, data in historical_data.items():
            if link in data['Link'].values:
                link_data = data[data['Link'] == link][time_periods].values.flatten()
                for i, period in enumerate(time_periods):
                    plot_data[period].append(link_data[i])
                years.append(year)
        
        # Add actual 2023 data
        if link in actual_data['Link'].values:
            actual_2023_data = actual_data[actual_data['Link'] == link][time_periods].values.flatten()
            for i, period in enumerate(time_periods):
                plot_data[period].append(actual_2023_data[i])
            years.append(2023)

        # Add predicted 2023 data
        if link in predictions:
            predicted_2023_data = predictions[link].flatten()
            for i, period in enumerate(time_periods):
                plot_data[period].append(predicted_2023_data[i])
            years.append('Predicted 2023')
        
        # Plot each time period
        for period in time_periods:
            plt.plot(years, plot_data[period], marker='o', label=period)
        
        plt.title(f'Typical Sunday, Link: {link}')
        plt.xlabel('Year')
        plt.ylabel('Load')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_predictions2(predictions, selected_links):
    time_periods = ['Early', 'AM Peak', 'Midday', 'PM Peak', 'Evening', 'Late']
    
    plt.figure(figsize=(10, 6))
    
    for link in selected_links:
        if link in predictions:
            predicted_values = predictions[link].flatten()
            plt.plot(time_periods, predicted_values, marker='o', label=link)
    
    plt.title("Predictions for 10 Selected Links Across Time Periods Sunday")
    plt.xlabel("Time Periods")
    plt.ylabel("Predicted Value")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

def prepare_all_predictions_data(predictions):
    # Initialize a dictionary to hold flattened data for all links
    all_predictions_data = {link: predictions[link].flatten() for link in predictions.keys()}
    
    # Convert the dictionary to a DataFrame
    all_predictions_df = pd.DataFrame(all_predictions_data)

    all_predictions_df = all_predictions_df.iloc[:, :50]
    
    return all_predictions_df

def create_correlation_matrix_all(predictions_df):
    # Calculate the correlation matrix for the predictions DataFrame
    correlation_matrix = predictions_df.corr()
    return correlation_matrix

def plot_correlation_matrix(correlation_matrix, title):
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title(title)
    plt.show()

# Main block to run the script
if __name__ == "__main__":
    # Get the list of common links between all years
    common_links = set(NUMBAT16_sun_flows_brief['Link']).intersection(
        NUMBAT17_sun_flows_brief['Link'],
        NUMBAT18_sun_flows_brief['Link'],
        NUMBAT19_sun_flows_brief['Link'],
        NUMBAT22_sun_flows_brief['Link'],
        NUMBAT23_sun_flows_brief['Link']
    )

    # Run the parallel processing for all common links
    all_predictions2023_sun = parallel_process(list(common_links), num_processes=4, seq_length=1)
    print("final sun prediction is: ")
    print(all_predictions2023_sun)

    selected_links = random.sample(list(all_predictions2023_sun.keys()), 10)

    # Prepare historical data in a dictionary
    historical_data = {
        2016: NUMBAT16_sun_flows_brief,
        2017: NUMBAT17_sun_flows_brief,
        2018: NUMBAT18_sun_flows_brief,
        2019: NUMBAT19_sun_flows_brief,
        2022: NUMBAT22_sun_flows_brief
    }

    # Plot predictions vs actual and historical data
    plot_predictions(all_predictions2023_sun, NUMBAT23_sun_flows_brief, historical_data, selected_links)

    plot_predictions2(all_predictions2023_sun, selected_links)

    # Prepare time period data for the selected links
    time_period_data = prepare_time_period_data(all_predictions2023_sun, selected_links)

    # Create the correlation matrix for selected links
    correlation_matrix_selected = create_correlation_matrix(time_period_data)

    # Display the correlation matrix for selected links
    print("Correlation matrix for selected 10 links:")
    print(correlation_matrix_selected)

    # Optionally, plot the correlation matrix for selected links as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix_selected, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix_selected.columns)), correlation_matrix_selected.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix_selected.index)), correlation_matrix_selected.index)
    plt.title("Correlation Matrix for Sunday Selected Links")
    plt.show()

    # Prepare data for all predictions
    all_predictions_df = prepare_all_predictions_data(all_predictions2023_sun)

    # Create the correlation matrix for all predictions
    correlation_matrix_all = create_correlation_matrix_all(all_predictions_df)

    # Display the correlation matrix for all predictions
    print("Correlation matrix for all predictions:")
    print(correlation_matrix_all)

    # Plot the correlation matrix for all predictions
    plot_correlation_matrix(correlation_matrix_all, "Correlation Matrix for 50 Sunday Predictions Across Links")
