import pandas as pd
import numpy as np
import os
from tqdm import tqdm

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

def generate_seqlen_datasets(input_path, output_root, dataset_name, n_iterations=5, sep=','):
    print(f"\n--- Processing Sequence Length Split for RecBole: {dataset_name} ---")
    
    if dataset_name in ['ml-1m']:
        df = pd.read_csv(input_path, sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        
    elif dataset_name == 'ml-100k':
        df = pd.read_csv(input_path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df = df[['user', 'item', 'timestamp']]
    
    elif 'amazon' in dataset_name:
        df = pd.read_json(input_path, lines=True, compression='gzip')
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'timestamp'})
        df = df[['user', 'item', 'timestamp']]

    df = df.dropna()
    df = filter_k_core(df, k=5)

    all_users = df['user'].unique()
    user_dict = {uid: group for uid, group in df.groupby('user')}

    for i in range(1, n_iterations + 1):
        group_folder_name = f"group_{i:02d}"
        print(f"\n[{group_folder_name}] Pair Formation & Length Calculation...")
        
        current_users = all_users.copy()
        np.random.shuffle(current_users)
        
        paired_accounts = []
        lengths = []
        
        for j in tqdm(range(0, len(current_users)-1, 2), desc="Pairing"):
            u1, u2 = current_users[j], current_users[j+1]
            u1_data = user_dict[u1]
            u2_data = user_dict[u2]
            
            merged = pd.concat([u1_data, u2_data]).sort_values('timestamp')
            
            if len(merged) >= 3:
                paired_accounts.append(merged[['item', 'timestamp']])
                lengths.append(len(merged))

        len_series = pd.Series(lengths)
        try:
            groups = pd.qcut(len_series, 3, labels=['low', 'mid', 'high'], duplicates='drop')
            if len(groups.unique()) < 3:
                sorted_idx = len_series.sort_values().index
                n = len(sorted_idx)
                groups = pd.Series(index=len_series.index, dtype='object')
                groups.iloc[sorted_idx[:n//3]] = 'low'
                groups.iloc[sorted_idx[n//3:2*n//3]] = 'mid'
                groups.iloc[sorted_idx[2*n//3:]] = 'high'
        except ValueError:
            sorted_idx = len_series.sort_values().index
            n = len(sorted_idx)
            groups = pd.Series(index=len_series.index, dtype='object')
            groups.iloc[sorted_idx[:n//3]] = 'low'
            groups.iloc[sorted_idx[n//3:2*n//3]] = 'mid'
            groups.iloc[sorted_idx[2*n//3:]] = 'high'
        
        for level in ['high', 'mid', 'low']:
            group_indices = groups[groups == level].index
            group_data_list = []
            
            for new_uid, idx in enumerate(group_indices):
                temp_df = paired_accounts[idx].copy()
                temp_df['user_id'] = new_uid + 1
                group_data_list.append(temp_df[['user_id', 'item', 'timestamp']])
            
            if group_data_list:
                final_df = pd.concat(group_data_list)
                final_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
                
                exp_name = f"{dataset_name}_seqlen_{level}_{group_folder_name}"
                target_dir = os.path.join(output_root, exp_name)
                os.makedirs(target_dir, exist_ok=True)
                
                file_path = os.path.join(target_dir, f"{exp_name}.inter")
                final_df.to_csv(file_path, sep='\t', index=False)
                
                current_u = final_df['user_id:token'].nunique()
                current_i = final_df['item_id:token'].nunique()
                unique_n = len(final_df.drop_duplicates(['user_id:token', 'item_id:token']))
                real_sparsity = 1 - (unique_n / (current_u * current_i))
                print(f" > Level {level:4s} (Size 2): U={current_u}, I={current_i}, Sparsity={real_sparsity:.6f} -> Saved in {target_dir}")