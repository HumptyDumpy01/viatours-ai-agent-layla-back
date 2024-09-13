import random
import re

import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm

# define split size to turn groups of sentences into chunks
from AI.open_and_read_pdf_1 import pages_and_texts

# Further text processing (splitting pages onto sentences)

nlp = English()

# Add a sentencizer pipeline
nlp.add_pipe("sentencizer")

# creating a document instance as an example
doc = nlp("This is a test sentence. Howdy! I do adore Full-Stack Development!")

assert len(list(doc.sents)) == 3

# print out sentences split
# print(f'''{list(doc.sents)}''')
# print(f'''pages and texts: {pages_and_texts[0]}''')


for item in tqdm(pages_and_texts):
  item['sentences'] = list(nlp(item['text']).sents)

  # making sure that all sentences are strings
  item['sentences'] = [str(sentence) for sentence in item['sentences']]
  # count the sentences
  item['page_sentence_count_spacy'] = len(item['sentences'])

print(f'''{random.sample(pages_and_texts, k=1)}''')

# Chunking our sentences together


num_sentence_chunk_size = 10


# create a function to split lists of texts recursively into chunk size
# e.g. [20] -> [10, 10]
# e.g [25] -> [10, 10, 5]

def split_list(input_list: list[str], slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
  return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]


test_list = list(range(25))
print(f'''{split_list(test_list)}''')

# loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
  item['sentence_chunks'] = split_list(input_list=item['sentences'], slice_size=num_sentence_chunk_size)

item['num_chunks'] = len(item['sentence_chunks'])

print(f'''{random.sample(pages_and_texts, k=1)}''')

# Splitting each chunk into its own item
pages_and_chunks = []

for item in tqdm(pages_and_texts):
  for sentence_chunk in item['sentence_chunks']:
    chunk_dict = {}
    chunk_dict['page_number'] = item['page_number']

    # Join the sentences onto a paragraph-like structure
    joined_sentence_chunk = ''.join(sentence_chunk).replace('  ', ' ').strip()
    joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
    chunk_dict['sentence_chunk'] = joined_sentence_chunk

    # get some stats on our stats
    chunk_dict['chunk_char_count'] = len(joined_sentence_chunk)
    chunk_dict['chunk_word_count'] = len([word for word in joined_sentence_chunk.split(" ")])
    chunk_dict['chunk_token_count'] = len(joined_sentence_chunk) / 4
    pages_and_chunks.append(chunk_dict)

# print(f'''{len(pages_and_chunks)}''')

# print(f'''{random.sample(pages_and_chunks, k=1)}''')


# Filter all chunks that are smaller than 25 tokens
min_token_length = 25

df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)

for row in df[df['chunk_token_count'] <= min_token_length].sample(5).iterrows():
  print(f'''Chink token count: {row[1]['chunk_token_count']} | Text: {row[1]['sentence_chunk']}''')

pages_and_chunks_over_min_token_len = df[df['chunk_token_count'] >= min_token_length].to_dict(orient='records')
print(f'''{pages_and_chunks_over_min_token_len[:2]}''')
