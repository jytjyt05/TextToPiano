import pandas as pd
import re
import numpy as np


df = pd.read_csv('data/maestro-v3.0.0.csv')

# Extract tonality
def extract_tonic(title):
    # match = re.search(r'(?i)in ([A-G](?:-flat|-sharp)? Major|[A-G](?:-flat|-sharp)? Minor)', title)
    match = re.search(r'.*([A-G](?:\sflat|\ssharp|\sFlat|\sSharp)?\s(Major|Minor))', title)
    # print(match)
    if match:
        return match.group(1)
    else:
        return np.nan

df['tonic'] = df['canonical_title'].apply(extract_tonic)
dropped_df=df.dropna(subset=['tonic'])
dropped_df=dropped_df.reset_index()

dropped_df.to_csv('new.csv', index=False)


