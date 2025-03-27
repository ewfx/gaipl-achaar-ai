import uvicorn
import asyncio
import json
import random
import uuid
from datetime import datetime

from fastapi.staticfiles import StaticFiles # Add this line
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_tasks = {}  # In-memory storage for search tasks (replace with a database for production)
active_connections = {} # in memory to store active web sockets

class SearchRequest(BaseModel):
    query: str
    filters: dict = {}

async def mock_search_process(search_id: str, websocket: WebSocket):
    sources = ["Enterprise Docs", "ServiceNow Incidents", "Past RCA Doc DB", "Workarounds", "Known Issues"]
    progress_per_source = {}
    results = {}

    for source in sources:
        progress_per_source[source] = 0

    for source in sources:
        for i in range(1, 101):
            await asyncio.sleep(0.05)  # Simulate processing time
            progress_per_source[source] = i
            search_tasks[search_id]["details"] = [
                {"source": s, "status": "completed" if progress_per_source[s] == 100 else "in_progress" if progress_per_source[s] > 0 else "pending", "progress": progress_per_source[s], "message": f"Processing {s}"}
                for s in sources
            ]
            search_tasks[search_id]["progress"] = sum(progress_per_source.values()) // len(sources)
            await websocket.send_json(search_tasks[search_id])

        results[source] = f"Result from {source}: {random.choice(['Relevant info', 'Possible solution', 'Related incident'])}"
    search_tasks[search_id]["status"] = "completed"
    search_tasks[search_id]["results"] = results
    await websocket.send_json(search_tasks[search_id])

@app.post("/search")
async def initiate_search(request: SearchRequest):
    search_id = str(uuid.uuid4())
    search_tasks[search_id] = {
        "searchId": search_id,
        "status": "pending",
        "progress": 0,
        "details": [],
        "results": None,
        "error": None,
        "query" : request.query,
        "filters" : request.filters
    }
    return {"searchId": search_id, "status": "pending"}

@app.websocket("/ws/search/{search_id}")
async def websocket_endpoint(websocket: WebSocket, search_id: str):
    await websocket.accept()
    active_connections[search_id] = websocket
    try:
        if search_id in search_tasks:
            await mock_search_process(search_id, websocket)
        else:
            await websocket.send_json({"error": "search id not found"})
    except WebSocketDisconnect:
        del active_connections[search_id]
    finally:
        del active_connections[search_id]

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)