# embed all chunks in batches
from AI.embed_text_chunks_3 import text_chunks, embedding_model

text_chunk_vectors = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)

print(f'''{text_chunk_vectors}''')
