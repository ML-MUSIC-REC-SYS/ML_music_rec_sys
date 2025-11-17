import pandas as pd
import numpy as np
import scipy.sparse as sparse
import pickle
import time
from sklearn.preprocessing import normalize
from collections import defaultdict

print("## FINAL HYBRID FILTERING AND ANALYSIS ##")

# --- 1. Load data and models ---
try:
    # Load final content features (item features)
    df_content_final = pd.read_csv('data/df_content_final_sampled.csv', index_col='song_index')
    # Load training and test interaction matrices
    R_train = sparse.load_npz('data/R_train_sampled.npz').astype(np.float32).tocsr()
    R_test = sparse.load_npz('data/R_test_sampled.npz').tocsr()
    # Load project variables
    with open('data/project_vars_sampled.pkl', 'rb') as f:
        project_vars = pickle.load(f)
        VALID_SONG_INDICES = project_vars['VALID_SONG_INDICES']
        SONG_ID_MAPPER = project_vars['SONG_ID_MAPPER']
    # Load Collaborative Filtering (CF) inferred top genres per user
    with open('data/user_top_genres.pkl', 'rb') as f:
        USER_TOP_GENRES = pickle.load(f)
except FileNotFoundError:
    print("Error: Required file not found. Please run msd_data_sample.py and model_genre_cf.py first.")
    exit()

# --- 2. CBF Create User Profile ---
def create_user_profiles(R_train_csr, df_content, valid_song_indices):
    """
    Creates user profiles by aggregating the features of songs they listened to.
    This is the Content-Based Filtering (CBF) part.
    """
    df_features = df_content.copy()
    num_songs_total = R_train_csr.shape[1]
    feature_dim = df_features.shape[1]
    
    # 1. Create a full item feature matrix, zero-padded for songs not in df_content or invalid
    item_feature_matrix = np.zeros((num_songs_total, feature_dim))
    valid_idx_in_df = df_features.index.intersection(valid_song_indices)
    item_feature_matrix[valid_idx_in_df] = df_features.loc[valid_idx_in_df].values
    
    # Handle NaN (shouldn't happen after preprocessing, but safe)
    item_feature_matrix = np.nan_to_num(item_feature_matrix)
    # Normalize item features (L2 norm)
    item_feature_matrix = normalize(item_feature_matrix, norm='l2', axis=1)
    
    # 2. Create user profiles by dot product: R_train * Item_Features
    user_feature_matrix = R_train_csr.dot(item_feature_matrix)
    user_feature_matrix = np.nan_to_num(user_feature_matrix)
    # Normalize user profiles (L2 norm)
    user_feature_matrix = normalize(user_feature_matrix, norm='l2', axis=1)
    
    # Handle cold-start users (users with no plays in R_train or only plays for invalid songs)
    zero_users = np.where(np.sum(user_feature_matrix, axis=1) == 0)[0]
    if len(zero_users) > 0:
        mean_profile = np.mean(user_feature_matrix[np.sum(user_feature_matrix, axis=1) != 0], axis=0)
        user_feature_matrix[zero_users] = mean_profile
        print(f"Handled {len(zero_users)} cold-start users with mean profile.")
        
    return user_feature_matrix, item_feature_matrix

print("\n2. CBF Start to create a User Profile...")
user_profiles, normalized_item_features = create_user_profiles(R_train, df_content_final, VALID_SONG_INDICES)
print("CBF Create User Profile Done.")

# --- 3. Preparing Genre Features ---
# Extract only the tag group columns (genres)
tag_group_cols = [
    col for col in df_content_final.columns
    if not col.startswith(('tempo', 'loudness', 'duration', 'key_', 'mode_', 'ts_'))
]
# Normalized tag features for genre similarity calculation
normalized_tag_features = normalize(df_content_final[tag_group_cols].values, norm='l2', axis=1)
# Mapping from global song index to the index within the normalized_tag_features matrix
song_index_map = {idx: i for i, idx in enumerate(df_content_final.index)}

# --- 4. Hybrid Recommendation and Evaluation: Real Precision/Recall + Genre Similarity ---
def recommend_hybrid_genre(user_profiles, item_features_norm, R_train_csr, R_test_csr, K=5):
    """
    Performs hybrid recommendation (CBF score + Genre-based re-ranking) and evaluates
    using Precision@K, Recall@K, and Genre Similarity.
    """
    start_time = time.time()
    # Get all users who have test data
    all_test_users = np.unique(R_test_csr.nonzero()[0])
    MAX_EVAL_USERS = min(1000, len(all_test_users))
    np.random.seed(42)
    # Sample a subset of users for evaluation to save time
    test_users = np.random.choice(all_test_users, size=MAX_EVAL_USERS, replace=False)

    total_genre_match_score = 0
    total_precision = 0
    total_recall = 0
    valid_users_count = 0

    for user_index in test_users:
        user_vector = user_profiles[user_index]
        if np.all(user_vector == 0): # Skip cold-start users handled by mean profile (though they'll get mean recommendations)
            continue
        
        # 1. Base CBF Score Calculation (Cosine Similarity: User Profile dot Item Feature)
        scores = user_vector.dot(item_features_norm.T)
        scores = np.nan_to_num(scores)
        
        # Initial score normalization/clipping
        # Normalization based on 95th percentile to scale diverse score ranges
        p95_initial = np.percentile(scores, 95)
        scores = np.clip(scores / (p95_initial + 1e-10), 0, 1)
        scores = np.maximum(scores, 1e-10)

        # 2. Hybridization: Genre-based Re-ranking/Attenuation
        top_genres = USER_TOP_GENRES.get(user_index, ['POP', 'ROCK']) # Get CF-inferred top genres
        
        # Attenuate scores for songs whose genres are NOT among the user's top genres
        for song_index in df_content_final.index:
            row = df_content_final.loc[song_index]
            # Check if song belongs to any of the user's top genres (weight > 0)
            in_top = any((g in row.index and row[g] > 0) for g in top_genres)
            if not in_top:
                scores[song_index] *= 0.4 # Reduce score by 60% if genre is mismatched
        scores = np.nan_to_num(scores)
        
        # 3. Filter out already listened songs
        liked_indices = R_train_csr.indices[R_train_csr.indptr[user_index]:R_train_csr.indptr[user_index+1]]
        scores[liked_indices] = -np.inf # Exclude training data items
        
        # Final score normalization (post-genre filtering)
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) > 0:
            p95_final = np.percentile(finite_scores, 95)
            scores = np.clip(scores / (p95_final + 1e-10), 0, 1)
        else:
            scores[:] = 0  # Cold-start or user already listened to all available songs
        scores = np.maximum(scores, 1e-10) # Ensure no zeros for log-likelihood in some models, or just maintain positive scores
        
        # 4. Extract Top-K Recommendations
        top_k_indices = np.argsort(scores)[::-1][:K]
        
        # 5. Get Ground Truth (Test set items)
        actual_items_indices = R_test_csr.indices[R_test_csr.indptr[user_index]:R_test_csr.indptr[user_index+1]]
        if len(actual_items_indices) == 0:
            continue

        # 6. Calculate Precision/Recall
        top_k_set = set(top_k_indices)
        test_item_set = set(actual_items_indices)
        hit_count = len(top_k_set & test_item_set)
        
        precision = hit_count / K
        recall = hit_count / max(len(test_item_set), 1)
        
        total_precision += precision
        total_recall += recall
        
        # 7. Calculate Genre Similarity (Cosine Similarity)
        # Calculate the average genre vector of the actual liked items in the test set
        actual_feats = [normalized_tag_features[song_index_map[a]] 
                        for a in actual_items_indices if a in song_index_map]
        
        if not actual_feats:
            continue
            
        avg_actual = np.mean(actual_feats, axis=0)
        
        match_score = 0
        for rec in top_k_indices:
            if rec in song_index_map:
                rec_feat = normalized_tag_features[song_index_map[rec]]
                match_score += np.dot(rec_feat, avg_actual) # Sum of dot products (cosine sim)
                
        total_genre_match_score += match_score / K
        valid_users_count += 1
        
    avg_precision = total_precision / max(valid_users_count, 1)
    avg_recall = total_recall / max(valid_users_count, 1)
    avg_genre_sim = total_genre_match_score / max(valid_users_count, 1)
    
    print(f"\n--- Final Results (Evaluated on {valid_users_count} users) ---")
    print(f"Actual Precision@{K}: {avg_precision:.4f}")
    print(f"Actual Recall@{K}: {avg_recall:.4f}")
    print(f"Genre Similarity (Cosine): {avg_genre_sim:.4f}")
    
    return avg_precision, avg_recall, avg_genre_sim, test_users

# --- 5. Execution ---
K = 5
print(f"\n4. Starting Hybrid Recommendation and Evaluation (Precision@{K})...")
avg_prec, avg_rec, avg_genre_sim, eval_users = recommend_hybrid_genre(user_profiles, normalized_item_features, R_train, R_test, K)

# --- 6. Qualitative Analysis (Sample Recommendation/Ground Truth Genre Comparison) ---
def get_recommendation_details(user_index, k=5):
    """
    Gets the details for the top K recommendations for a specific user.
    """
    user_vector = user_profiles[user_index]
    
    # 1. Base CBF Score Calculation
    scores = user_vector.dot(normalized_item_features.T)
    scores = np.nan_to_num(scores)
    p95_initial = np.percentile(scores, 95)
    scores = np.clip(scores / (p95_initial + 1e-10), 0, 1)
    scores = np.maximum(scores, 1e-10)

    # 2. Hybridization: Genre-based Re-ranking/Attenuation
    top_genres = USER_TOP_GENRES.get(user_index, ['POP', 'ROCK'])
    for song_index in df_content_final.index:
        row = df_content_final.loc[song_index]
        in_top = any((g in row.index and row[g] > 0) for g in top_genres)
        if not in_top:
            scores[song_index] *= 0.4
            
    # 3. Filter out already listened songs
    liked_indices = R_train.indices[R_train.indptr[user_index]:R_train.indptr[user_index+1]]
    scores[liked_indices] = -np.inf
    
    # Final score normalization (post-genre filtering)
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) > 0:
        p95_final = np.percentile(finite_scores, 95)
        scores = np.clip(scores / (p95_final + 1e-10), 0, 1)
    else:
        scores[:] = 0 
    
    # 4. Extract Top-K Recommendations
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    recs = []
    for rank, idx in enumerate(top_k_indices):
        item_genres = [col for col in tag_group_cols if df_content_final.loc[idx, col] > 0.01] # Check for meaningful weight
        recs.append({
            "Rank": rank + 1,
            "Song_Index": idx,
            "Predicted_Genres": top_genres,
            "Item_Genres": item_genres,
            "Score": round(float(scores[idx]), 4)
        })
    return recs

# Print results for a sample user
if len(eval_users) > 0:
    sample_user_index = eval_users[0]
    print(f"\n--- Qualitative Analysis: User {sample_user_index} ---")
    print(f"CF Inferred Top Genres: {USER_TOP_GENRES.get(sample_user_index, ['POP', 'ROCK'])}")
    
    # Get ground truth items from the test set
    actual_items_indices = R_test.indices[R_test.indptr[sample_user_index]:R_test.indptr[sample_user_index+1]]
    test_genres = []
    for idx in actual_items_indices:
        if idx in df_content_final.index:
            test_genres.extend([col for col in tag_group_cols if df_content_final.loc[idx, col] > 0.01])
    print(f"Test Set (Ground Truth) Item Genres: {list(set(test_genres))}")
    
    # Display top recommendations
    print("\nTop Recommendations:")
    print(pd.DataFrame(get_recommendation_details(sample_user_index)))
else:
    print("\nNo users selected for qualitative analysis.")