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
        results = search_workouts(parsed["intent"], parsed["entities"], top_k=10)

        print("\n================ SEARCH & RECOMMENDATION START ==================\n")
        for res in results:
            print(
                f"- [{res['score']}] {res['title']} | {res['duration']} min | {res['instructor']} | {res['intensity']} | {res['type']}"
            )

        return results
    return []
