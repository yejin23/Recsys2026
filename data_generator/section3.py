
import re
from pathlib import Path
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

def generate_natural_merge_datasets(input_path, output_root, dataset_name, n_iterations=5, sep=','):
    print(f"\n--- Processing Natural Merge Experiment for RecBole: {dataset_name} ---")
    
    if dataset_name in ['ml-1m', 'ml-10m']:
        df = pd.read_csv(input_path, sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        
    elif dataset_name == 'ml-100k':
        df = pd.read_csv(input_path, sep='\t', names=['user', 'item', 'rating', 'timestamp'], engine='python')
        df = df[['user', 'item', 'timestamp']]
    
    elif dataset_name == 'steam':
        df = pd.read_csv(input_path, sep='\t', low_memory=False)
        rename_dict = {
            'user_id:token': 'user', 
            'product_id:token': 'item', 
            'timestamp:float': 'timestamp'
        }
        if 'item_id:token' in df.columns:
            rename_dict['item_id:token'] = 'item'
            
        df = df.rename(columns=rename_dict)
        df = df[['user', 'item', 'timestamp']]
        
    elif dataset_name == 'kuairec':
        df = pd.read_csv(input_path, sep=sep)
        if 'video_id' in df.columns:
            df = df.rename(columns={'user_id': 'user', 'video_id': 'item'})
        df = df[['user', 'item', 'timestamp']]
    
    elif 'amazon' in dataset_name:
        df = pd.read_json(input_path, lines=True, compression='gzip')
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'timestamp'})
        df = df[['user', 'item', 'timestamp']]
        
    elif dataset_name == 'lastfm':
        print(f"Opening tar.gz file: {input_path}")
        with tarfile.open(input_path, "r:gz") as tar:
            member = [m for m in tar.getmembers() if 'userid-timestamp' in m.name][0]
            f = tar.extractfile(member)
            df = pd.read_csv(io.BytesIO(f.read()), sep='\t', 
                            names=['user', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'], 
                            on_bad_lines='skip', engine='python')
            
        df = df.rename(columns={'track_id': 'item'})
        df = df[['user', 'item', 'timestamp']]
        print("Converting Last.fm timestamps to unix format...")
        df['timestamp'] = pd.to_datetime(df['timestamp']).view(np.int64) // 10**9

    df = df.dropna()
    df = filter_k_core(df, k=5)

    user_dict = {uid: group.sort_values('timestamp') for uid, group in df.groupby('user')}
    all_users = list(user_dict.keys())

    for i in range(1, n_iterations + 1):
        group_folder_name = f"exp_{i:02d}"
        print(f"\n[{group_folder_name}] Generating Natural Merge Datasets...")
        
        current_users = all_users.copy()
        np.random.shuffle(current_users)

        single_accounts_list = []
        shared_accounts_list = []
        
        single_account_id = 1
        shared_account_id = 1
        
        for j in range(0, len(current_users) - 1, 2):
            uid1 = current_users[j]
            uid2 = current_users[j+1]
            
            user1_data = user_dict[uid1]
            user2_data = user_dict[uid2]
            
            single1 = user1_data.copy()
            single2 = user2_data.copy()
            
            raw_shared_data = pd.concat([user1_data, user2_data]).sort_values(by=['timestamp', 'user'])
            
            if len(single1) < 3 or len(single2) < 3 or len(raw_shared_data) < 3:
                continue
            
            single1['user_id'] = single_account_id
            single_accounts_list.append(single1[['user_id', 'item', 'timestamp']])
            single_account_id += 1
            
            single2['user_id'] = single_account_id
            single_accounts_list.append(single2[['user_id', 'item', 'timestamp']])
            single_account_id += 1
            
            shared_data = raw_shared_data.copy()
            shared_data['user_id'] = shared_account_id
            shared_accounts_list.append(shared_data[['user_id', 'item', 'timestamp']])
            shared_account_id += 1

        for acc_type, acc_list in [('single_natural', single_accounts_list), ('shared_natural', shared_accounts_list)]:
            if not acc_list:
                continue
                
            final_df = pd.concat(acc_list)
            final_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
            
            exp_name = f"{dataset_name}_{acc_type}_{group_folder_name}"
            target_dir = os.path.join(output_root, exp_name)
            os.makedirs(target_dir, exist_ok=True)
            
            file_path = os.path.join(target_dir, f"{exp_name}.inter")
            final_df.to_csv(file_path, sep='\t', index=False)
            
            current_u = final_df['user_id:token'].nunique()
            current_i = final_df['item_id:token'].nunique()
            avg_seq_len = len(final_df) / current_u
            unique_n = len(final_df.drop_duplicates(['user_id:token', 'item_id:token']))
            real_sparsity = 1 - (unique_n / (current_u * current_i))
            
            print(f" > {acc_type:18s}: U={current_u}, I={current_i}, Avg_Len={avg_seq_len:.2f}, Sparsity={real_sparsity:.6f} -> Saved in {target_dir}")

output_folder = ''