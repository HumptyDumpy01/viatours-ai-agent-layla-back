# Save embeddings to file
import pandas as pd

from AI.split_pages_onto_sentences_2 import pages_and_chunks_over_min_token_len

text_chunks_vectors_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
vectors_df_save_path = '../data/text_chunks_and_vectors_df.csv'

text_chunks_vectors_df.to_csv(vectors_df_save_path, index=False)
