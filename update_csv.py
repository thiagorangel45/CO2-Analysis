import pandas as pd
import os

# Read the main CSV file
main_csv_path = 'updated_2024_data_atWP.csv'
main_df = pd.read_csv(main_csv_path)

# Get the list of scan names
scan_names = main_df.iloc[:, 0].tolist()  # Assuming scan names are in the first column

# Directory containing the other CSV files
csv_folder = 'Data_2024/'

# Create new columns for the extracted data
main_df['Muon CS Err'] = None
main_df['Gamma CS Err'] = None

# Iterate over each scan name and process the corresponding CSV file
for scan_name in scan_names:
    # Construct the file name
    file_name = f'WP_{scan_name}.csv'
    file_path = os.path.join(csv_folder, file_name)
    
    if os.path.exists(file_path):
        # Read the corresponding CSV file
        df = pd.read_csv(file_path)
        
        # Check if the file has at least 14 columns
        if df.shape[1] >= 14:
            # Extract the 13th and 14th columns (12th and 13th indices)
            col13 = df.iloc[:, 12]
            col14 = df.iloc[:, 13]
            
            # Write the first value of these columns to the corresponding row in the main DataFrame
            main_df.loc[main_df.iloc[:, 0] == scan_name, 'Muon CS Err'] = col13.iloc[0] if len(col13) > 0 else None
            main_df.loc[main_df.iloc[:, 0] == scan_name, 'Gamma CS Err'] = col14.iloc[0] if len(col14) > 0 else None
        else:
            print(f'File {file_path} does not have at least 14 columns.')
    else:
        print(f'File {file_path} does not exist.')

# Save the updated main CSV file
#updated_csv_path = 'updated_2024_data_atWP.csv'
#main_df.to_csv(updated_csv_path, index=False)


# import pandas as pd

# # Path to the original CSV file
# input_csv_path = '2023_data_atWP.csv'

# # Read the CSV file
# df = pd.read_csv(input_csv_path)

# # Drop columns 8 to 40 (7 to 39 in 0-based indexing)
# columns_to_drop = df.columns[7:9]
# df.drop(columns=columns_to_drop, inplace=True)

# # Path to save the updated CSV file
output_csv_path = '2023_data_atWP.csv'

# # Save the updated DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)
