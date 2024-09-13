import textwrap

import torch
from sentence_transformers import SentenceTransformer, util

from AI.split_pages_onto_sentences_2 import pages_and_chunks

embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device='cpu')


def print_wrapped(text, wrap_length=80):
  wrapped_text = textwrap.fill(text, wrap_length)
  print(f'''{wrapped_text}''')


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

  # Ensure both tensors have the same data type
  if query_embedding.dtype != embeddings.dtype:
    query_embedding = query_embedding.to(embeddings.dtype)

  # Get dot product scores on embeddings
  dot_scores = util.dot_score(query_embedding, embeddings)[0]

  scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
  return scores, indices


def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict] = pages_and_chunks,
                                 n_resources_to_return: int = 5):
  """Find the closest top vectors based on the query vector and print the scores with its results."""
  scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings,
                                                n_resources_to_return=n_resources_to_return)

  for score, idx in zip(scores, indices):
    print(f'''Score: {score:4f}''')
    print(f'''Text \n''')
    print_wrapped(pages_and_chunks[idx]['sentence_chunk'])
    print(f'''Page Number: {pages_and_chunks[idx]['page_number']}''')
    print(f'''\n''')
