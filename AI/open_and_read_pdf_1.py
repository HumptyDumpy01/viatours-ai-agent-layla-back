# we already have a PDF, now let's open it!
# let's use pymupdf library

import fitz
from tqdm.auto import tqdm

# import random

pdf_file = "..data/viatours-docs.pdf"


# from main import pdf_file


def text_formatter(text: str) -> str:
  """Performs minor formatting for text."""
  cleaned_text = text.replace("\n", " ").strip()

  # the point is, the cleaner your chunks, the better response would be.
  return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
  doc = fitz.open(pdf_file)
  pages_and_text = []
  for page_number, page in tqdm(enumerate(doc)):
    text = page.get_text()
    text = text_formatter(text=text)
    pages_and_text.append({
      'page_number': page_number,
      'page_char_count': len(text),
      'page_word_count': len(text.split(" ")),
      'page_sentence_count_raw': len(text.split(". ")),
      'page_token_count': len(text) / 4,  # 1 token equals 4 chars
      'text': text
    })
  return pages_and_text


pages_and_texts = open_and_read_pdf(pdf_path=pdf_file)

# print(f'''{pages_and_texts[:2]}''')
# print(f'''{random.choice(pages_and_texts)}''')
