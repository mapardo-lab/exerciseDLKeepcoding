import pandas as pd

def process_data(df): 
    df['NumTags'] = df['tags'].apply(eval).apply(len)
    df['categories'] = df['categories'].apply(eval)
    df['Likes_Dislikes'] = df['Likes'] - df['Dislikes']
    df['target'] = df['Likes_Dislikes'].apply(lambda x: 1 if x > 0 else 0)
    return df