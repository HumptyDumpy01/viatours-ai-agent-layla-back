# main.py

import os

from dotenv import load_dotenv
from fastapi import FastAPI, status, APIRouter, HTTPException, Path
from pydantic import ValidationError

import models
from AI.embed_the_query_7 import retrieve_relevant_resources, pages_and_chunks
from AI.generate_answer_8 import generate_answer
from AI.vector_search_6 import vectors

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

    # Retrieve relevant resources
    scores, indices = retrieve_relevant_resources(query=query, embeddings=vectors, n_resources_to_return=5)
    top_vectors_converted_to_text = [pages_and_chunks[idx]['sentence_chunk'] for idx in indices]
    joined_text = ' '.join(top_vectors_converted_to_text)

    # Limit the length of the text to 700 chars if the text is too long
    if len(joined_text) > 800:
      joined_text = joined_text[:800]

    prefix = '''You are Viatours Artificial Intelligence Agent model developed by Nikolas Tuz, Software Engineer.
    Your name is Layla AI Agent. You are developed for one purpose: To help Viatours Customers with their questions.
    If the user answer is not cohesive and just random, tell him to be more specific. Also, if your answer becomes huge, it would be good
    if it won't succeed 800 characters. Please, answer the following viatours user question:'''

    final_prompt = f"{prefix} {query}\n\nContext: {joined_text}"

    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('OPEN_AI_API')

    # Generate the answer
    answer = generate_answer(final_prompt, api_key)
    return {
      'query': query,
      "response": answer,
      'status': status.HTTP_200_OK,
      'date': queryObj.date
    }

  except ValidationError as e:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid request! {e}')

  except Exception as e:
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to get response! {e}')


app.include_router(router)
