import implicit
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import pickle
import time
import os
from implicit.nearest_neighbours import bm25_weight

# Path Configuration
DATA_DIR = 'data'
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1: Weighted Combination of User-Item (Song) Matrix
print("## 1: Starting User-Item (Song) Matrix Creation and Weighted Combination")

R_train = sparse.load_npz(os.path.join(DATA_DIR, 'R_train_sampled.npz')).tocsr()
# Binary R matrix (user has listened: 1, otherwise: 0)
R_binary = (R_train > 0).astype(np.float32)
# Weighted combination (Play count * 0.7 + Binary presence * 0.3)
R_weighted = R_train.multiply(0.7) + R_binary.multiply(0.3)

print(f"R_weighted shape: {R_weighted.shape}")
print("Weighted combination complete.\n")

# 2: Genre-based CF (ALS) Implementation (for Hybrid)
print("## 2: Starting Genre-based CF (ALS) Implementation (for Hybrid)")

df_content_final = pd.read_csv(os.path.join(DATA_DIR, 'df_content_final_sampled.csv'), index_col='song_index')

# Select only genre columns
tag_group_cols = [c for c in df_content_final.columns if c not in 
                  ['tempo','loudness','duration'] and 
                  not c.startswith(('key_','mode_','ts_'))]

if 'OTHER_GENRE' in tag_group_cols:
    tag_group_cols.remove('OTHER_GENRE')
    print("Note: 'OTHER_GENRE' genre has been excluded from CF training.")

num_songs_total = R_weighted.shape[1]
num_genre = len(tag_group_cols)

# Create the Item-Genre matrix
item_genre_matrix = np.zeros((num_songs_total, num_genre))
item_genre_matrix[df_content_final.index.values] = df_content_final[tag_group_cols].values

# Mitigate genre bias: Boost rare genres
genre_counts = np.sum(item_genre_matrix > 0, axis=0)
genre_weights = np.median(genre_counts) / (genre_counts + 1e-5)
item_genre_matrix = item_genre_matrix * genre_weights

# Calculate User-Genre Matrix: R_weighted * Item_Genre_Matrix
R_genre = R_weighted.dot(item_genre_matrix)
R_genre_sparse = sparse.csr_matrix(R_genre)

print(f"User-Genre Matrix shape: {R_genre_sparse.shape}")

# ALS Training
start_time = time.time()

model_genre_als = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=50,
    random_state=42
)

# Apply BM25 weighting to the User-Genre matrix
R_genre_weighted = bm25_weight(R_genre_sparse, K1=1.2, B=0.75)
model_genre_als.fit(R_genre_weighted)

print(f"Genre-based CF training complete. (Time taken: {time.time() - start_time:.2f} seconds)\n")

# Result Calculation: User-specific Genre Scores and Top 2 Extraction
user_factors = model_genre_als.user_factors
genre_factors = model_genre_als.item_factors

# Predict user-genre preference scores
user_genre_scores = user_factors.dot(genre_factors.T)
user_genre_df = pd.DataFrame(user_genre_scores, columns=tag_group_cols)

# Top 2 Genre Selection Logic (Applying Relative Threshold)
user_top_genres = {}
REL_THRESHOLD = 0.1  # 10% relative to the user's highest score

for uid in range(user_genre_df.shape[0]):
    scores = user_genre_df.loc[uid]
    top_score = scores.max()
    
    # Apply relative threshold
    selected = [g for g in scores.index if scores[g] >= REL_THRESHOLD * top_score]
    
    # Sort by highest score
    selected = sorted(selected, key=lambda g: scores[g], reverse=True)
    
    # Select a maximum of 2
    top_genres = selected[:2]
    
    # Safety handling
    if len(top_genres) == 0:
        top_genres = ['POP','ROCK'] # Default genres if no preference found
    elif len(top_genres) == 1:
        top_genres.append('')  # Second genre is empty
    
    user_top_genres[uid] = top_genres

# Sample Output
print("Sample 10 user results:", dict(list(user_top_genres.items())[:10]))

# ----------------------------------------------------
# Save Results
# ----------------------------------------------------
with open(os.path.join(OUTPUT_DIR, 'user_top_genres.pkl'), 'wb') as f:
    pickle.dump(user_top_genres, f)

with open(os.path.join(OUTPUT_DIR, 'model_genre_als.pkl'), 'wb') as f:
    pickle.dump(model_genre_als, f)

print("\nModel and results saved successfully")