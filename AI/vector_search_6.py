# RAG Search and Answer

# Again, the main goal for RAG is to retrieve the closest set of tensors, enhance its shape using LLM, so it would
# return the answer, already converted from vectors to plain text.


# Similarity Search

# Comparing embeddings is known as similarity search. Or semantic search.
# In our case, let's try to query the viatours docs to help him to get the most
# corresponding answer

# by searching e.g. "How to delete my account?" the LLM should give the closest enhanced answer possible.

import numpy as np
import pandas as pd
import torch

device = 'cuda'

# import texts and vectors df

text_chunks_and_vectors_df = pd.read_csv('../data/text_chunks_and_vectors_df.csv')

# convert to a numpy array to be able to use it.
text_chunks_and_vectors_df['embedding'] = text_chunks_and_vectors_df['embedding'].apply(
  lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Convert our vectors into torch.tensor
vectors = torch.tensor(np.stack(text_chunks_and_vectors_df['embedding'].tolist(), axis=0))

# convert texts and vectors df into a list of dicts
pages_and_chunks = text_chunks_and_vectors_df.to_dict(orient='records')

# print(f'''{text_chunks_and_vectors_df}''')
# print(f'''{vectors.shape}''')



