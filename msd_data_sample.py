import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py
import os
import time
from sklearn.preprocessing import MinMaxScaler
import pickle
import scipy.sparse as sparse
import warnings
from collections import defaultdict
import h5py

try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# --- Paths and Configuration ---
TRIPLET_FILE_PATH = 'data/train_triplets.txt'
H5_DIRECTORY = 'data/MillionSongSubset/'
MAX_USERS_SAMPLE = 50000 
MIN_PLAY_COUNT_PER_USER = 20 # Remove users with less than 20 plays
MIN_PLAY_COUNT_PER_SONG = 10  # Remove songs with less than 10 plays
# --------------------

print("Starting data preprocessing...")

# --- 1-1. Load Triplet File and User Filtering/Sampling ---
start_time = time.time()
print(f"1-1. Loading Triplet file: {TRIPLET_FILE_PATH}")
raw_data = pd.read_csv(TRIPLET_FILE_PATH, sep='\t', header=None, names=['user_id', 'song_id', 'play_count'])
print(f"Full loading complete. (Time taken: {time.time() - start_time:.2f} seconds)")

# 1. User Sampling (1 million -> 50k)
all_unique_users = raw_data['user_id'].unique()
np.random.seed(42)
if len(all_unique_users) > MAX_USERS_SAMPLE:
    sample_user_ids = np.random.choice(all_unique_users, size=MAX_USERS_SAMPLE, replace=False)
    raw_data = raw_data[raw_data['user_id'].isin(sample_user_ids)].copy()
print(f"*** Data Reduction: User sampling completed from {len(all_unique_users)} users to {len(raw_data['user_id'].unique())} users ***")

# 2. ID Mapping (Initial)
unique_users_temp = raw_data['user_id'].astype('category').cat.categories
user_to_index_temp = {user_id: i for i, user_id in enumerate(unique_users_temp)}
raw_data['user_index'] = raw_data['user_id'].map(user_to_index_temp)
NUM_USERS_TEMP = len(user_to_index_temp)

unique_songs_temp = raw_data['song_id'].astype('category').cat.categories
song_to_index_temp = {song_id: i for i, song_id in enumerate(unique_songs_temp)}
raw_data['song_index'] = raw_data['song_id'].map(song_to_index_temp)
NUM_SONGS_TEMP = len(song_to_index_temp)

R_sparse_temp = csr_matrix((raw_data['play_count'], (raw_data['user_index'], raw_data['song_index'])), 
                           shape=(NUM_USERS_TEMP, NUM_SONGS_TEMP))

# 3. User Filtering (Remove users with less than 20 plays)
print(f"\n*** Starting User Filtering: Removing users with total play count less than {MIN_PLAY_COUNT_PER_USER} ***")
user_counts = R_sparse_temp.sum(axis=1).A1
valid_user_indices = np.where(user_counts >= MIN_PLAY_COUNT_PER_USER)[0]
num_users_removed = NUM_USERS_TEMP - len(valid_user_indices)

# Update raw_data
valid_user_ids = raw_data['user_id'].unique()[valid_user_indices]
raw_data = raw_data[raw_data['user_id'].isin(valid_user_ids)].copy()

# 4. ID Mapping and Final R_sparse Generation (Recalculated with filtered data)
unique_users = raw_data['user_id'].astype('category').cat.categories
user_to_index = {user_id: i for i, user_id in enumerate(unique_users)}
raw_data['user_index'] = raw_data['user_id'].map(user_to_index)

unique_songs = raw_data['song_id'].astype('category').cat.categories
song_to_index = {song_id: i for i, song_id in enumerate(unique_songs)}
raw_data['song_index'] = raw_data['song_id'].map(song_to_index)

R_sparse = csr_matrix((raw_data['play_count'], (raw_data['user_index'], raw_data['song_index'])), 
                      shape=(len(user_to_index), len(song_to_index)))

# Update Global Variables
NUM_USERS = R_sparse.shape[0]
NUM_SONGS = R_sparse.shape[1]
SONG_ID_MAPPER = raw_data[['song_id', 'song_index']].drop_duplicates().set_index('song_id')['song_index'].to_dict()

print(f"Final number of users: {NUM_USERS} (Users removed: {num_users_removed})")
print(f"Final number of songs: {NUM_SONGS}")
print(f"Final total number of play records: {R_sparse.nnz}")


# --- 1-2. H5 Feature Extraction ---
def get_features_from_h5_revised(h5_file_path):
    features = {}
    tags = []
    try:
        with h5py.File(h5_file_path, 'r') as f:
            meta_data = f['/metadata/songs'][0]
            features['song_id'] = meta_data['song_id'].decode('utf-8')
            features['artist_name'] = meta_data['artist_name'].decode('utf-8')
            features['title'] = meta_data['title'].decode('utf-8')

            analysis_data = f['/analysis/songs'][0]
            features['tempo'] = analysis_data['tempo']
            features['key'] = analysis_data['key']
            features['mode'] = analysis_data['mode']
            features['time_signature'] = analysis_data['time_signature']
            features['loudness'] = analysis_data['loudness']
            features['duration'] = analysis_data['duration']
            
            artist_terms = f['/metadata/artist_terms'][:]
            artist_terms_weight = f['/metadata/artist_terms_weight'][:]
            
            for term_bytes, weight in zip(artist_terms, artist_terms_weight):
                term = term_bytes.decode('utf-8')
                if term:
                    tags.append({'song_id': features['song_id'], 'tag': term, 'tag_count': float(weight)})
    except Exception as e:
        return None, None
    return features, tags

all_song_features = []
all_song_tags = []
h5_files_processed = 0
total_start_time_h5 = time.time()
print(f"\n1-2. Starting H5 Feature Extraction.")
for root, dirs, files in os.walk(H5_DIRECTORY):
    for file_name in files:
        if file_name.endswith('.h5'):
            file_path = os.path.join(root, file_name)
            feature_data, tag_data = get_features_from_h5_revised(file_path)
            if feature_data:
                all_song_features.append(feature_data)
            if tag_data:
                all_song_tags.extend(tag_data)
            h5_files_processed += 1
            if h5_files_processed % 1000 == 0:
                 print(f"-> Processing {h5_files_processed} files...")
            
df_features = pd.DataFrame(all_song_features)
df_tags = pd.DataFrame(all_song_tags)
print(f"\nTotal {h5_files_processed} H5 files processed. (Time taken: {time.time() - total_start_time_h5:.2f} seconds)")

# --- 1-3. Tag Feature Matrix Generation and Grouping ---
start_time_tag = time.time()
print("\n1-3. Starting Explicit Tag Feature Matrix Generation...")

df_tags['song_index'] = df_tags['song_id'].map(SONG_ID_MAPPER)
df_tags_mapped = df_tags.dropna(subset=['song_index'])

TOP_N_TAGS = 500 
top_tags = df_tags_mapped['tag'].value_counts().nlargest(TOP_N_TAGS).index
df_song_tags_filtered = df_tags_mapped[df_tags_mapped['tag'].isin(top_tags)]

df_tag_features = df_song_tags_filtered.pivot_table(
    index='song_index',
    columns='tag',
    values='tag_count',
    fill_value=0
)
df_tag_features.index = df_tag_features.index.astype(int)
df_tag_features.sort_index(inplace=True)

# 1-3.5. Manual Grouping
TAG_GROUPS = {
    'HIP_HOP': ['hip hop', 'rap', 'underground rap', 'alternative hip hop', 'pop rap', 'g funk', 'west coast rap', 'east coast hip hop', 'dirty south rap', 'gangsta', 'gangster rap', 'hardcore rap'],
    'ROCK': ['rock', 'classic rock', 'album rock', 'american trad rock', 'hard rock', 'soft rock', 'southern rock', 'post rock', 'indie rock', 'alternative rock', 'art rock', 'punk', 'punk rock', 'pop rock', 'garage rock', 'funk rock', 'experimental rock', 'psychedelic rock', 'math rock', 'noise rock', 'glam rock', 'shock rock', "rock 'n roll", 'rockabilly', 'surf music', 'swamp rock'],
    'METAL': ['metal', 'heavy metal', 'alternative metal', 'nu metal', 'glam metal', 'thrash metal', 'death metal', 'black metal', 'doom metal', 'power metal', 'speed metal', 'industrial metal', 'viking metal', 'metalcore', '80s metal'],
    'ELECTRONIC': ['electronic', 'electronica', 'club', 'club dance', 'house', 'deep house', 'techno', 'trance', 'progressive trance', 'hard trance', 'dubstep', 'drum and bass', 'dub', 'trip hop', 'downtempo', 'ambient', 'dark ambient', 'idm', 'electro', 'synthpop', 'industrial', 'breakbeat', 'breakcore', 'big beat', 'gabba', 'hardcore', 'hard house', 'hardstyle', 'intelligent dance music', 'new beat', 'progressive house', 'progressive trance', 'tribal house', 'uk garage'],
    'POP': ['pop', 'dance pop', 'electropop', 'europop', 'pop punk', 'pop metal', 'pop rock', 'dance', 'euro-house', 'eurodance', 'j-pop', 'k-pop', 'adult contemporary', 'brill building pop', 'contemporary pop', 'contemporary country', 'country pop', 'dance-punk', 'folks-pop', 'indie pop', 'latin pop', 'pop rap', 'pop rock', 'spanish pop'],
    'FOLK_COUNTRY': ['folk', 'folk rock', 'country', 'country music', 'country rock', 'americana', 'bluegrass', 'honky tonk', 'celtic', 'contemporary folk', 'traditional country'],
    'JAZZ_BLUES': ['jazz', 'cool jazz', 'smooth jazz', 'vocal jazz', 'jazz fusion', 'blues', 'electric blues', 'delta blues', 'modern electric blues', 'soul jazz', 'acid jazz', 'big band', 'afro-cuban jazz', 'contemporary jazz', 'free jazz', 'future jazz', 'latin jazz', 'nu jazz', 'piano blues', 'texas blues', 'chicago blues', 'louisiana blues'],
    'R_B_SOUL': ['r&b', 'soul', 'neo soul', 'blue-eyed soul', 'funk', 'funk soul', 'soulful', 'motown', 'northern soul', 'southern soul', 'lovers rock'],
    'LATIN_WORLD': ['latin', 'latin jazz', 'latin pop', 'world music', 'world', 'salsa', 'bossa nova', 'tango', 'samba', 'afrobeat', 'cumbia', 'tropical', 'world fusion', 'world reggae', 'flamenco', 'chanson francaise', 'mambo', 'merengue', 'rumba', 'spanish', 'catalan'],
    'REGGAE_SKA': ['reggae', 'roots reggae', 'rock steady', 'dancehall', 'ska', 'ska punk', 'rasta'],
    'CHILL_DREAM': ['chill-out', 'ambient', 'downtempo', 'relax', 'dreamy', 'mellow', 'ethereal', 'dark', 'melancholia', 'sad', 'slow', 'soft', 'smooth', 'meditation', 'lounge'],
    'ERA_90S_PLUS': ['90s', '00s', 'contemporary'],
    'ERA_60S_80S': ['60s', '70s', '80s', 'oldies', 'classic', 'old'],
    'VOCAL_INST': ['singer-songwriter', 'female vocalist', 'male vocalist', 'instrumental', 'piano', 'guitar', 'voice', 'vocal', 'composer', 'solo', 'song writer', 'lyrics', 'percussion', 'saxophone', 'trumpet', 'turnablism', 'vocals', 'acoustic', 'acoustic guitar', 'banjo', 'bass', 'drums', 'guitarist', 'guitar virtuoso'],
    'RELIGION': ['christian', 'christian music', 'christian pop', 'christian rock', 'gospel', 'black gospel', 'contemporary gospel', 'southern gospel', 'worship music', 'inspirational', 'praise', 'praise & worship', 'religious music', 'spiritual'],
    
    # --- OTHER_GENRE Sub-categorization ---
    'AMBIENT_EXP': ['abstract', 'avant-garde', 'experimental', 'glitch', 'minimal', 'noise', 'dark ambient', 'free improvisation', 'free jazz', 'post-rock', 'ambient'], # Experimental/Atmospheric
    'SOUNDTRACK': ['soundtrack', 'original score', 'ost', 'film music', 'opera', 'classical', 'modern classical', 'neoclassical'], # Narrative/Classical
    'DANCE_BEAT': ['beat', 'beats', 'dj', 'remix', 'club', 'up beat', 'turnablism', 'groove'], # Rhythm/Mixing
    'OTHER_MISC': ['crossover', 'eclectic', 'fusion', 'original', 'parody', 'poetry', 'political', 'raw', 'retro', 'spoken word', 'stand-up comedy', 'urban', 'holiday', 'christmas music', 'album', 'compilation', 'cover', 'live'], # Other miscellaneous
}

TAG_TO_GROUP = {}
for group, tags in TAG_GROUPS.items():
    for tag in tags:
        TAG_TO_GROUP[tag] = group

df_tags_unpivoted = df_tag_features.stack().reset_index(name='tag_weight')
df_tags_unpivoted.rename(columns={'level_1': 'tag'}, inplace=True)

df_tags_unpivoted['tag_group'] = df_tags_unpivoted['tag'].map(TAG_TO_GROUP).fillna('OTHER_GENRE') # Changed to OTHER_GENRE

df_grouped_tag_features = df_tags_unpivoted.groupby(['song_index', 'tag_group'])['tag_weight'].sum().unstack(fill_value=0)

df_tag_features_final = df_grouped_tag_features
print(f"1-3. Tag Feature Matrix Generation Complete. (Time taken: {time.time() - start_time_tag:.2f} seconds)")

#df tag binarization
# df_tag_features_final_binary = df_tag_features_final.copy()
# df_tag_features_final_binary[df_tag_features_final_binary > 0] = 1

# --- 1-4. Final Integration and Item Filtering ---
start_time_tech = time.time()
print("\n1-4. Starting Technical Feature Refinement and Final Content Feature Integration...")

# 1. Connect song_index to H5 feature data and filter
df_features['song_index'] = df_features['song_id'].map(SONG_ID_MAPPER)
df_features_mapped = df_features.dropna(subset=['song_index']).copy()
df_features_mapped['song_index'] = df_features_mapped['song_index'].astype(int)
df_features_mapped.set_index('song_index', inplace=True)

# 2. Select technical features for content-based filtering
CONTENT_FEATURES_TECH = ['tempo', 'loudness', 'duration', 'key', 'mode', 'time_signature']
df_content_tech = df_features_mapped[CONTENT_FEATURES_TECH].copy()

# 3. Handle missing values (NaN) and prepare for scaling
for col in ['tempo', 'loudness', 'duration']:
    df_content_tech.loc[:, col] = df_content_tech.loc[:, col].fillna(df_content_tech[col].median())
    
# 4. One-Hot Encoding for Categorical Features (Key, Mode, Time Signature)
df_content_tech = pd.get_dummies(df_content_tech, columns=['key', 'mode', 'time_signature'], prefix=['key', 'mode', 'ts'], drop_first=True)

# 5. Scaling Numerical Features (Normalize scale between 0 and 1)
numerical_cols = ['tempo', 'loudness', 'duration']
scaler = MinMaxScaler()
df_content_tech[numerical_cols] = scaler.fit_transform(df_content_tech[numerical_cols])

# 6. Merge Technical Features and Tag Features (Final Merge!)
df_content_final = df_content_tech.merge(
    df_tag_features_final,
    left_index=True,
    right_index=True,
    how='left' 
).fillna(0) 

# 7. Item Filtering (Remove songs with less than 5 plays)
print(f"\n*** Starting Item Filtering: Removing songs with total play count less than {MIN_PLAY_COUNT_PER_SONG} (Only saving indices, no physical removal) ***")

song_counts = R_sparse.sum(axis=0).A1
valid_song_indices = np.where(song_counts >= MIN_PLAY_COUNT_PER_SONG)[0]
num_songs_removed = R_sparse.shape[1] - len(valid_song_indices)

# --- R_sparse and df_content_final are NOT physically removed and maintain full size. ---
# Instead, only valid indices are saved.
VALID_SONG_INDICES = valid_song_indices
# ---------------------------------------------------------------------------------

print(f"Number of songs to be removed: {num_songs_removed}")
print(f"Number of valid songs (can be used for model training): {len(VALID_SONG_INDICES)}")

# Sort indices and generate final matrix (without physical removal)
df_content_final.sort_index(inplace=True)
CONTENT_FEATURE_MATRIX = df_content_final.values

print(f"1-4. Final Content Feature Integration Complete. (Time taken: {time.time() - start_time_tech:.2f} seconds)")
print(f"Total number of features (dimensions): {len(df_content_final.columns)}")


# --- 1-5. Split and Save Training/Evaluation Data ---
def train_test_split_sparse(R_sparse, test_ratio=0.2):
    np.random.seed(42)  
    
    rows, cols = R_sparse.nonzero()
    data = R_sparse.data # <--- Extract data (play counts) from R_sparse
    user_indices = rows

    user_to_data = defaultdict(list)
    for idx, user in enumerate(user_indices):
        user_to_data[user].append(idx)
    
    test_indices = []
    users_processed = 0
    total_users = len(user_to_data)
    start_time_split = time.time()
    
    for user, indices in user_to_data.items():
        if len(indices) > 1:
            num_test = max(1, int(len(indices) * test_ratio))
            chosen = np.random.choice(indices, size=num_test, replace=False)
            test_indices.extend(chosen)
            
        users_processed += 1
        if users_processed % 10000 == 0:
            elapsed_time = time.time() - start_time_split
            progress = users_processed / total_users * 100
            print(f"-> R/T Split in progress: {users_processed}/{total_users} users processed ({progress:.1f}%), Elapsed Time: {elapsed_time:.0f}s")
            
    test_indices_set = set(test_indices)
    train_indices = [i for i in range(len(rows)) if i not in test_indices_set]
    
    # Create R_train: Use only data corresponding to train_indices
    R_train = csr_matrix((data[train_indices], 
                           (rows[train_indices], cols[train_indices])), 
                           shape=R_sparse.shape)

    # Create R_test: Use only data corresponding to test_indices
    R_test = csr_matrix((data[test_indices], 
                          (rows[test_indices], cols[test_indices])), 
                          shape=R_sparse.shape)
    
    print(f"-> After split R_train NNZ: {R_train.nnz:,} / R_test NNZ: {R_test.nnz:,}")
    print("-> R/T Split complete. Generating final matrices...")
    
    return R_train.tocsr(), R_test.tocsr() # Ensure R_train is also CSR

print("\nTask 1-5: Starting Train/Test Split")
R_train, R_test = train_test_split_sparse(R_sparse, test_ratio=0.2)

print("\n--- [Task 1-5 Result: Training/Evaluation Data Split] ---")
print(f"Final matrix shape: {R_sparse.shape}")
print(f"R_train (Training) play records: {R_train.nnz} ({R_train.nnz / R_sparse.nnz * 100:.2f}%)")
print(f"R_test (Evaluation) play records: {R_test.nnz} ({R_test.nnz / R_sparse.nnz * 100:.2f}%)")

FINAL_CONTENT_FEATURES = df_content_final.columns.tolist()

# Final Save
print("\n\n*** Starting Final Data Save for Stage 1 ***")

sparse.save_npz('data/R_train_sampled.npz', R_train)
sparse.save_npz('data/R_test_sampled.npz', R_test)
df_content_final.to_csv('data/df_content_final_sampled.csv')

project_vars = {
    'SONG_ID_MAPPER': SONG_ID_MAPPER,
    'user_to_index': user_to_index,
    'NUM_USERS': NUM_USERS,
    'NUM_SONGS': NUM_SONGS,
    'FINAL_CONTENT_FEATURES': FINAL_CONTENT_FEATURES,
    'VALID_SONG_INDICES': VALID_SONG_INDICES # <-- Store list of valid song indices (Crucial)
}
with open('data/project_vars_sampled.pkl', 'wb') as f:
    pickle.dump(project_vars, f)

print("*** Final Data Save Complete: R_train_sampled.npz, R_test_sampled.npz, df_content_final_sampled.csv ***")