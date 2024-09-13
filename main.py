import os

import numpy as np
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, status, APIRouter, HTTPException, Path
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer, util

import models

# import texts and vectors df

text_chunks_and_vectors_df = pd.read_csv('text_chunks_and_vectors_df.csv')

# convert to a numpy array to be able to use it.
text_chunks_and_vectors_df['embedding'] = text_chunks_and_vectors_df['embedding'].apply(
  lambda x: np.fromstring(x.strip('[]'), sep=' '))

# Convert our vectors into torch.tensor
vectors = torch.tensor(np.stack(text_chunks_and_vectors_df['embedding'].tolist(), axis=0))

# convert texts and vectors df into a list of dicts
pages_and_chunks = text_chunks_and_vectors_df.to_dict(orient='records')

# Load the embedding model and move it to CUDA
embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device='cuda')

# Load the embeddings and move them to CUDA
vectors = torch.tensor(np.stack(text_chunks_and_vectors_df['embedding'].tolist(), axis=0)).to('cuda')


# print(f'''VECTORS: {vectors}''')
# Ensure vectors are of the same dtype
# vectors = vectors.to(dtype=torch.float32)
############################

def retrieve_relevant_resources(
        query: str,
        embeddings: torch.tensor,
        model: SentenceTransformer = embedding_model,
        n_resources_to_return: int = 5,
        print_time: bool = True
):
  """Vectorizes a query with the model and returns topK scores and indices from embeddings."""
  # Embed the query
  query_embedding = model.encode(query, convert_to_tensor=True)

  # transform query embedding to type float32
  query_embedding = query_embedding.to(dtype=torch.float32)
  # transform embeddings to type float32
  embeddings = embeddings.to(dtype=torch.float32)

  # Get dot product scores on embeddings
  # start_time = timer()
  dot_scores = util.dot_score(query_embedding, embeddings)[0]

  scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
  return scores, indices


############################
app = FastAPI()

router = APIRouter(
  prefix="/viatours-agent",
  tags=["viatours-agent"]
)


@router.get("/health-check", status_code=status.HTTP_200_OK, description="A health check API endpoint")
async def root():
  try:
    return {
      "message": "Healthy!",
      'status': status.HTTP_200_OK
    }
  except Exception as e:
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to check health! {e}')


@router.get("/get-response/{query}", status_code=status.HTTP_200_OK, description="Get response from AI Chat Bot")
async def get_response(query: str = Path(..., min_length=1, max_length=500)):
  try:
    queryObj = models.Query(query=query)
    queryObj.date = queryObj.date[:10] + ' ' + queryObj.date[11:19]
    print(f'''{queryObj}''')

    query = queryObj.query

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

    load_dotenv()

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

    ############################
    OPEN_API_API = os.getenv('OPEN_AI_API')
    ############################
    # Prepare the prompt
    api_key = OPEN_API_API
    prompt = f"{prefix} {query}\n\nContext: {joined_text}"

    # IMPORTANT: THESE LINES OF CODE ARE COMMENTED OUT BECAUSE OF THE API USAGE.
    # IMPORTANT: BY UNCOMMENTING THESE LINES, YOU WILL USE THE API CREDITS.
    answer = generate_answer(prompt, api_key)
    print(f"Generated Answer: {answer}")

    return {
      'response': answer,
      'status': status.HTTP_200_OK,
      'query': queryObj.query,
      'date': queryObj.date
    }

  except ValidationError as e:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid request! {e}')

  except Exception as e:
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to get response! {e}')


app.include_router(router)
