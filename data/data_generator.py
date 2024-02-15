import pandas as pd
import numpy as np

sample = pd.read_csv('E:\source\ISS\AD_project\ML\data\hdbdata.csv')

flat_types = sample['flat_type'].unique().tolist()
HDB_towns = sample['town'].unique().tolist()
flat_models = sample['flat_model'].unique().tolist()

def generate_recom_data(num_users=1000,num_properties=1500):
    np.random.seed(42)
    print(flat_types)

    users = pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'preferred_town': np.random.choice(HDB_towns, num_users),
        'preferred_flat_type': np.random.choice(flat_types, num_users),
        'low_price': np.random.randint(200000, 400000, num_users),
        'high_price': np.random.randint(400001, 800000, num_users)
    })

    for flat_type in flat_types:
        users[f'prefers_{flat_type}'] = np.random.randint(0, 2, num_users)


    properties = pd.DataFrame({
        'property_id': range(1, num_properties + 1),
        'town': np.random.choice(HDB_towns, num_properties),
        'flat_type': np.random.choice(flat_types, num_properties),
        'floor_area_sqm': np.random.randint(65, 150, num_properties),
        'flat_model': np.random.choice(flat_models, num_properties),
        'resale_price': np.random.randint(300000, 800000, num_properties)
    })
    
    return users,properties

def merge_df(users,properties):
    users['key'] = 1
    properties['key'] = 1
    merged_df = pd.merge(users, properties, on='key').drop('key', axis=1)
    merged_df['town_match'] = (merged_df['preferred_town'] == merged_df['town']).astype(int)

    for flat_type in flat_types:
        merged_df[f'{flat_type}_match'] = (merged_df['flat_type'] == flat_type) & (merged_df[f'prefers_{flat_type}'] == 1).astype(int)

    merged_df['price_in_range'] = ((merged_df['low_price'] <= merged_df['resale_price']) & 
                                (merged_df['resale_price'] <= merged_df['high_price'])).astype(int)

    return merged_df

def define_intrest(merged_df):
    np.random.seed(42)
    #merged_df['interest_level'] = np.random.randint(0, 2, size=len(merged_df))
    room_match_score = merged_df[['3 ROOM_match', '4 ROOM_match', 'EXECUTIVE_match', '2 ROOM_match', '5 ROOM_match', "MULTI-GENERATION_match"]].sum(axis=1) / 6 * 0.3
    
    town_match_score = merged_df['town_match'] * 0.5
    price_in_range_score = merged_df['price_in_range'] * 0.2

    match_score = town_match_score + room_match_score + price_in_range_score
    match_score_normalized = (match_score - match_score.min()) / (match_score.max() - match_score.min())
    random_offset = np.random.uniform(-0.05, 0.05, size=match_score_normalized.shape)
    interest_level_with_offset = match_score_normalized + random_offset
    
    interest_level_with_offset = np.clip(interest_level_with_offset, 0, 1)

    merged_df['interest_level'] = interest_level_with_offset

def generate_interaction():

    np.random.seed(42)

    # Assume there are 1000 users and 100 items
    num_users = 1000
    num_items = 1500

    # Generate simulated user-item interaction data
    user_ids = np.random.randint(1, num_users + 1, size=5000)  # Randomly generate user IDs for 5000 interactions
    item_ids = np.random.randint(1, num_items + 1, size=5000)  # Randomly generate item IDs for 5000 interactions

    # Generate interaction counts, simplifying by randomly generating counts between 1 and 5 for each user-item pair
    interaction_counts = np.random.randint(1, 6, size=5000)

    # Create DataFrame
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'property_id': item_ids,
        'interaction_count': interaction_counts
    })

    return interactions_df
if __name__=="__main__":
    #SAVE THE DF AS CSV
    # user, property = generate_recom_data()
    # property.to_csv("property.csv",index=False)
    # merged_df = merge_df(user,property)
    # define_intrest(merged_df)
    # merged_df.to_csv('property_data.csv',index=False)

    # df = generate_interaction()
    # df.to_csv("interaction.csv",index=False)
    print(HDB_towns)

