import os

import openai
from dotenv import load_dotenv

from embed_the_query_7 import retrieve_relevant_resources, pages_and_chunks
from vector_search_6 import vectors

query = 'I forgot my password. How to reset it?'

scores, indices = retrieve_relevant_resources(query=query, embeddings=vectors,
                                              n_resources_to_return=5)

top_vectors_converted_to_text = [pages_and_chunks[idx]['sentence_chunk'] for idx in indices]

# this comes an array with one long string.
joined_text = ' '.join(top_vectors_converted_to_text)

# limit the length of the text to 700 chars if the text is too long
if len(joined_text) > 800:
  joined_text = joined_text[:800]

print(f'''Top Vectors Converted to Text: {top_vectors_converted_to_text}''')

prefix = '''You are Viatours Artificial Intelligence Agent model developed by Nikolas Tuz, Software Engineer.
Your name is Layla AI Agent. You are developed for one purpose: To help Viatours Customers with their questions.
 If the user answer is not cohesive and just random, tell him to be more specific. Also, if your answer becomes huge, it would be good
 if it won't succeed 800 characters. Please, answer the following viatours user question:'''

final_prompt = f"{prefix} {query}\n\nContext: {joined_text}"
print(f'''{final_prompt}''')

# Connect to ChatGPT API and generate the answer based on the query and the top vectors converted to text snippets(RAG).

load_dotenv()

OPEN_API_API = os.getenv('OPEN_AI_API')


# query  = 'parse the query from api'

# Function to call the ChatGPT API
def generate_answer(prompt, api_key):
  openai.api_key = api_key
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": final_prompt},
      {"role": "user", "content": prompt}
    ],
    max_tokens=150,
    n=1,
    stop=None,
    temperature=0.7,
  )
  return response.choices[0].message['content'].strip()


# Prepare the prompt
api_key = OPEN_API_API
prompt = f"{prefix} {query}\n\nContext: {joined_text}"

# print(f'''Prompt: {prompt}''')
# Generate the answer

# IMPORTANT: THESE LINES OF CODE ARE COMMENTED OUT BECAUSE OF THE API USAGE.
# IMPORTANT: BY UNCOMMENTING THESE LINES, YOU WILL USE THE API CREDITS.
answer = generate_answer(prompt, api_key)
print(f"Generated Answer: {answer}")
