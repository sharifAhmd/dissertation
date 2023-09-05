import os
import pandas as pd

TRAIN_DIR = "solar_flare_training_data"
os.makedirs(TRAIN_DIR, exist_ok = True)

def split_data_in_train_and_test(df, col_name, window):
    # target value 
    #M flare - between 10^-5 and 10^-4
    #X Flare - above 10^-4
    target_value = 1E-5
    # Initialize lists to store X_train and Y_train
    X_train_list = []
    Y_train_list = []

    # Iterate through the DataFrame to find X_train and Y_train
    for i in range(len(df)):
        if df.loc[i, col_name] >= target_value and i+1 > window:
            X_train = df.loc[i-window:i-1, col_name].values
            Y_train = df.loc[i, col_name]
            
            X_train_list.append(X_train)
            Y_train_list.append(Y_train)

    # Create a new DataFrame for X_train and Y_train
    X_train_df = pd.DataFrame(X_train_list, columns=['X_train_{}'.format(i) for i in range(window)])
    Y_train_df = pd.DataFrame({'Y_train': Y_train_list})
    
    X_train_df.to_csv(os.path.join(TRAIN_DIR, "X_Train.csv"), index = False)
    Y_train_df.to_csv(os.path.join(TRAIN_DIR, "Y_Train.csv"), index = False)
    
if __name__ == "__main__":
    print("Before running it keep it same as data source like sol_22_23_24_data_no_missing_v1.0.csv")
    print("Enter column name: (this should be xs or xl)")
    col_name = input()
    print("Enter window size:")
    window_size = int(input())
    df = pd.read_csv("sol_22_23_24_data_no_missing_v1.0.csv")
    split_data_in_train_and_test(df, col_name, window_size)
