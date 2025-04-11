from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from voice_assistant.nlu.nlu_pipeline import parse_text
from voice_assistant.search.search_workouts import search_workouts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/search")
async def search_endpoint(request: Request):
    data = await request.json()
    query_text = data.get("text", "")
    parsed = parse_text(query_text)
    if parsed["intent"] == "search_class":
        return search_workouts(parsed["intent"], parsed["entities"], top_k=10)
    return []
