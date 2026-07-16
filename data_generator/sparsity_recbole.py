import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def filter_k_core(df, k=5):
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

def generate_interaction_drop_datasets(input_path, output_root, dataset_name, drop_rates, n_iterations=5, sep=','):    
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
    print(f"Pre-grouping {len(all_users)} users...")
    user_dict = {uid: group for uid, group in df.groupby('user')}

    for rate in drop_rates:
        level = f"{int(rate*100)}"
        print(f"\n>> Target Interaction Drop Rate: {level}%")

        for i in range(1, n_iterations + 1):
            group_folder_name = f"group_{i:02d}"
            print(f" Generating {group_folder_name}...")

            current_users = all_users.copy()
            np.random.shuffle(current_users)
            
            shared_data_list = []
            pair_count = 0
            keep_prob = 1 - rate
            
            for j in range(0, len(current_users)-1, 2):
                u1, u2 = current_users[j], current_users[j+1]
                
                combined = pd.concat([user_dict[u1], user_dict[u2]])
                
                if len(combined) > 0:
                    sampled_data = combined.sample(frac=keep_prob).sort_values('timestamp')
                    
                    if len(sampled_data) >= 3:
                        sampled_data['user_id'] = pair_count + 1
                        shared_data_list.append(sampled_data[['user_id', 'item', 'timestamp']])
                        pair_count += 1
            
            if shared_data_list:
                final_df = pd.concat(shared_data_list)
                
                final_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
                
                exp_name = f"{dataset_name}_interaction_drop_{level}_{group_folder_name}"
                target_dir = os.path.join(output_root, exp_name)
                os.makedirs(target_dir, exist_ok=True)
                
                file_path = os.path.join(target_dir, f"{exp_name}.inter")
                final_df.to_csv(file_path, sep='\t', index=False)
                
                current_u = final_df['user_id:token'].nunique()
                current_i = final_df['item_id:token'].nunique()
                unique_n = len(final_df.drop_duplicates(['user_id:token', 'item_id:token']))
                real_sparsity = 1 - (unique_n / (current_u * current_i))
                
                print(f" > Level {level:4s} (Drop {rate:.1f}): U={current_u}, I={current_i}, Sparsity={real_sparsity:.6f} -> Saved in {target_dir}")

rates = [0.3, 0.5, 0.7]