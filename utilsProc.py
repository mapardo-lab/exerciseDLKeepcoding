import pandas as pd

def process_data(df): #
    df['NumTags'] = df['tags'].apply(eval).apply(len)
    df['categories'] = df['categories'].apply(eval)
    df['Likes_Dislikes'] = df['Likes'] - df['Dislikes']
    return df