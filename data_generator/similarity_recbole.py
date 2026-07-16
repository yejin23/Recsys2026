import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
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

def generate_similarity_based_datasets(input_path, output_root, dataset_name, n_iterations=5, sep=','):
    print(f"\n--- Processing Similarity Split for RecBole: {dataset_name} ---")
    
    if dataset_name in ['ml-1m']:
        df = pd.read_csv(input_path, sep='::', names=['user', 'item', 'rating', 'timestamp'], engine='python')
    
    elif 'amazon' in dataset_name:
        df = pd.read_json(input_path, lines=True, compression='gzip')
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'timestamp'})
        df = df[['user', 'item', 'timestamp']]

    df = df.dropna()
    df = filter_k_core(df, k=5)

    print("Building User-Item Matrix for Similarity...")
    user_cat = df['user'].astype('category')
    item_cat = df['item'].astype('category')
    
    row = user_cat.cat.codes
    col = item_cat.cat.codes
    data = np.ones(len(df))
    
    user_item_matrix = csr_matrix((data, (row, col)), 
                                  shape=(user_cat.cat.categories.size, item_cat.cat.categories.size))
    
    all_users = user_cat.cat.categories.values
    user_dict = {uid: group for uid, group in df.groupby('user')}
    
    def get_cosine_sim(u1_idx, u2_idx):
        vec1 = user_item_matrix[u1_idx]
        vec2 = user_item_matrix[u2_idx]
        return cosine_similarity(vec1, vec2)[0][0]

    for i in range(1, n_iterations + 1):
        group_folder_name = f"group_{i:02d}"
        print(f"\n[{group_folder_name}] Pair Formation & Similarity Calculation...")
        
        current_users_indices = np.arange(len(all_users))
        np.random.shuffle(current_users_indices)
        
        pairs = []
        sim_scores = []
        
        for j in tqdm(range(0, len(current_users_indices)-1, 2), desc="Processing Pairs"):
            idx1, idx2 = current_users_indices[j], current_users_indices[j+1]
            u1, u2 = all_users[idx1], all_users[idx2]
            
            u1_data = user_dict[u1]
            u2_data = user_dict[u2]
            
            if len(u1_data) + len(u2_data) >= 3:
                sim = get_cosine_sim(idx1, idx2)
                merged = pd.concat([u1_data, u2_data]).sort_values('timestamp')
                pairs.append(merged[['item', 'timestamp']])
                sim_scores.append(sim)

        sim_series = pd.Series(sim_scores)
        try:
            groups = pd.qcut(sim_series, 2, labels=['low', 'high'], duplicates='drop')
            if len(groups.unique()) < 2:
                ranks = sim_series.rank(method='first')
                groups = pd.Series(np.where(ranks <= len(ranks)/2, 'low', 'high'), index=sim_series.index)
        except ValueError:
            ranks = sim_series.rank(method='first')
            groups = pd.Series(np.where(ranks <= len(ranks)/2, 'low', 'high'), index=sim_series.index)

        for level in ['high', 'low']:
            group_mask = (groups == level)
            group_indices = groups[group_mask].index
            
            avg_sim = sim_series[group_mask].mean()
            
            group_data_list = []
            for new_uid, idx in enumerate(group_indices):
                temp_df = pairs[idx].copy()
                temp_df['user_id'] = new_uid + 1
                group_data_list.append(temp_df[['user_id', 'item', 'timestamp']])
            
            if group_data_list:
                final_df = pd.concat(group_data_list)
                final_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
                
                exp_name = f"{dataset_name}_similarity_{level}_{group_folder_name}"
                target_dir = os.path.join(output_root, exp_name)
                os.makedirs(target_dir, exist_ok=True)
                
                final_df.to_csv(os.path.join(target_dir, f"{exp_name}.inter"), sep='\t', index=False)
                
                current_u = final_df['user_id:token'].nunique()
                current_i = final_df['item_id:token'].nunique()
                unique_n = len(final_df.drop_duplicates(['user_id:token', 'item_id:token']))
                real_sparsity = 1 - (unique_n / (current_u * current_i))
                
                print(f" > Level {level:4s} (Size 2): U={current_u:6d}, I={current_i:6d}, "
                      f"Sim={avg_sim:.6f}, Sparsity={real_sparsity:.6f} -> Saved in {target_dir}")