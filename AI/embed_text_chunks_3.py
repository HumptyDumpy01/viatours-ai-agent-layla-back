# Embedding our text chunks!

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from AI.split_pages_onto_sentences_2 import pages_and_chunks_over_min_token_len

embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device='cpu')

# create a list of sentences
sentences = ['The Sentence Transformer Library provides an easy way to embed or vectorize the chunks',
             'Sentences can be embedded one by one or in a list', 'I love JavaScript']

# sentences are encoded/embedded by calling model.encode()

vectors = embedding_model.encode(sentences)
vectors_dict = dict(zip(sentences, vectors))

# see the embeddings
for sentence, vector in vectors_dict.items():
  print(f'''Sentence: {sentence}''')
  print(f'''Vector: {vector}''')
  print(f'''''')

embedding_model.to('cuda')

# Embed each chunk one by one
for item in tqdm(pages_and_chunks_over_min_token_len):
  item['embedding'] = embedding_model.encode(item['sentence_chunk'])


text_chunks = [item['sentence_chunk'] for item in pages_and_chunks_over_min_token_len]
# print(f'''{text_chunks}''')
# print(f'''{len(text_chunks)}''')

