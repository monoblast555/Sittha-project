import os
import pandas as pd

#Folder path where files are stored
folder_path = 'D://Research project//MALDI data//Lipids//4th bio GRE 11-7-2014'

#Create an empty dataframe
bio1_df = pd.DataFrame()

#Loop through the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):  #Check if the file is a .txt file
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as file:
            #Read lines from the file
            lines = file.readlines()

            #get the last 3 digits of the header
            header = lines[0].strip()
            row_index = str(header[-3:])

            #read the rest of the lines (data) and split into two columns
            data = [line.strip().split('\t') for line in lines[1:]]

            #Convert the data into a DataFrame
            df = pd.DataFrame(data, columns=[file_name, row_index])

            #Use the first column as the column names and the second column as the data
            df = df.set_index(file_name)

            #Append the row data with last 3 digits of the header (row index) to the result dataframe
            if bio1_df.empty:
                bio1_df = df
            else:
                bio1_df = pd.concat([bio1_df, df], axis=1)

#Set the first column (last 3 digits of the header from the files) as the index
bio1_df.reset_index(inplace=True)
bio1_df.rename(columns={'index': 'Samples'}, inplace=True)


#Save file
bio1_df.to_csv('D://Research project//MALDI data//pythonProject4//lipid_bio4.csv', index=False)

