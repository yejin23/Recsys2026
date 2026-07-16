import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tarfile  
import io   
def filter_k_core(df, k=5):
    print(f"Applying {k}-core filtering...")
    while True:
        user_count = df.groupby('user').size()
        item_count = df.groupby('item').size()
        
        mask = (df['user'].isin(user_count[user_count >= k].index)) & \
               (df['item'].isin(item_count[item_count >= k].index))
        
        new_df = df[mask]
        
        if len(new_df) == len(df):
            break
        df = new_df
    return df

def generate_groupsize_datasets(input_path, output_root, dataset_name, group_sizes=[2, 5, 10], n_iterations=5, sep=','):
    print(f"\n--- Processing Group Size Experiment for RecBole: {dataset_name} ---")
    
    if dataset_name in ['ml-1m']:
        df = pd.read_csv(input_path, sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        
    elif dataset_name == 'ml-100k':
        df = pd.read_csv(input_path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df = df[['user', 'item', 'timestamp']]
 
    elif 'amazon' in dataset_name:
        df = pd.read_json(input_path, lines=True, compression='gzip')
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'timestamp'})
        df = df[['user', 'item', 'timestamp']]
        
            
        df = df.rename(columns={'track_id': 'item'})
        df = df[['user', 'item', 'timestamp']]
        print("Converting Last.fm timestamps to unix format...")
        df['timestamp'] = pd.to_datetime(df['timestamp']).view(np.int64) // 10**9

    df = df.dropna()
    
    df = filter_k_core(df, k=5)

    all_users = df['user'].unique()
    user_dict = {uid: group for uid, group in df.groupby('user')}

    size_to_level = {2: 'low', 5: 'mid', 10: 'high'}

    for i in range(1, n_iterations + 1):
        group_folder_name = f"group_{i:02d}"
        print(f"\n[{group_folder_name}] Generating Shared Accounts...")
        
        size_results = {}
        for size in group_sizes:
            current_users = all_users.copy()
            np.random.shuffle(current_users)
            
            shared_accounts = []
            for j in range(0, len(current_users) - size + 1, size):
                group_uids = current_users[j : j + size]
                group_data = pd.concat([user_dict[uid] for uid in group_uids])
                if len(group_data) >= 3:
                    shared_accounts.append(group_data[['item', 'timestamp']])
            size_results[size] = shared_accounts

        target_account_count = len(size_results[max(group_sizes)])

        for size, accounts in size_results.items():
            level = size_to_level[size]
            
            if len(accounts) > target_account_count:
                selected_indices = np.random.choice(len(accounts), target_account_count, replace=False)
                final_accounts = [accounts[idx] for idx in selected_indices]
            else:
                final_accounts = accounts

            group_data_list = []
            for new_uid, acc_df in enumerate(final_accounts):
                temp_df = acc_df.copy().sort_values('timestamp')
                temp_df['user_id'] = new_uid + 1
                group_data_list.append(temp_df[['user_id', 'item', 'timestamp']])
            
            if group_data_list:
                final_df = pd.concat(group_data_list)
                
                final_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
                
                exp_name = f"{dataset_name}_groupsize_{level}_{group_folder_name}"
                target_dir = os.path.join(output_root, exp_name)
                os.makedirs(target_dir, exist_ok=True)
                
                file_path = os.path.join(target_dir, f"{exp_name}.inter")
                final_df.to_csv(file_path, sep='\t', index=False)
                
                current_u = final_df['user_id:token'].nunique()
                current_i = final_df['item_id:token'].nunique()
                unique_n = len(final_df.drop_duplicates(['user_id:token', 'item_id:token']))
                real_sparsity = 1 - (unique_n / (current_u * current_i))
                
                print(f" > Level {level:4s} (Size {size:2d}): U={current_u}, I={current_i}, Sparsity={real_sparsity:.6f} -> Saved in {target_dir}")