# FastAPI backend for Viatours AI RAG Agent Layla

## How to run the backend

- Install the dependencies with `pip install -r requirements.txt`
- Create a `.env` file with the following content:

```
OPEN_AI_API=your_api_key
```

- Run `pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Run the backend with `uvicorn main:app --reload`
- The backend will be available at `http://localhost:8000/docs`
- You can test the backend by sending a POST request to `http://localhost:8000/viatours-agent/get-response` with the
  following JSON payload:

```
{
  "query": "What is the capital of France?",
}
```

## Extra Information

- The Application does not have a database because the overall size of the data is small(150 vectors);
- It uses the ChatGPT API key instead of generating the response from the model directly;
- The Application uses the Hugging Face Transformers library;
- The Application uses the FastAPI as its backend framework;
- The Application uses the Pydantic library for data validation;
- The response from an API endpoint is rapid, usually less than 1-2 seconds;

This Application is done by me, as a part of the Viatours.



