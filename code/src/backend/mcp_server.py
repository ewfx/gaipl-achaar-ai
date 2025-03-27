import uvicorn
import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Path, Body, Query
from transformers import AutoTokenizer, AutoModel
from typing import List
import sqlite3
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
import uuid
from typing import Optional, List, Dict, Callable
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",  # Allow requests from your frontend
    # Add other origins if needed
]
# Load E5 model and tokenizer
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model_name = "intfloat/e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


# Initialize FastAPI
app = FastAPI(title="Enterprise IT Logs Retrieval API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for request bodies
class TicketBase(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    # Add other common fields as needed based on your database schema

class TicketCreate(TicketBase):
    # Define fields required for creating a ticket
    pass

class TicketUpdate(TicketBase):
    # Define fields that can be updated
    pass


# Define Pydantic models for request bodies
class TicketBase(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    # Add other common fields as needed based on your database schema

class TicketResponseSmall(BaseModel):
    id: str
    created_date: Optional[datetime.datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    cmdb_ci_id: Optional[str] = None

class TicketResponse(BaseModel):
    id: str
    created_date: Optional[datetime.datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    reported_by: Optional[str] = None
    severity: Optional[str] = None
    assigned_group: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    closure_code: Optional[str] = None
    closed_date: Optional[datetime.datetime] = None
    incident_type: Optional[str] = None
    cmdb_ci_id: Optional[str] = None
    work_around_id: Optional[str] = None
    known_problem_id: Optional[str] = None
    rca_document_id: Optional[str] = None
    change_request_id: Optional[str] = None
    telemetry_id: Optional[str] = None
    logs_id: Optional[str] = None
    updated_date: Optional[datetime.datetime] = None

db_name = "snow.db"
# Assume these functions and classes are defined elsewhere in your code
def load_docs_from_db(db_name: str, table_name: str) -> List[dict]:
    """Loads documents from a SQLite database table."""
    # Replace this with your actual database loading logic
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

tickets_db = load_docs_from_db("snow.db", "incident")

# MCP API routes
@app.get("/api/tickets", response_model=List[TicketResponseSmall])
async def get_tickets(status: Optional[str] = Query(None), priority: Optional[str] = Query(None)):
    """Get all tickets or filter by query parameters"""
    
    filtered_tickets = tickets_db
    print("filtered_tickets", len(filtered_tickets),"status:",status, "priority:",priority)
    if status:
        filtered_tickets = [t for t in filtered_tickets if t.get('status', '').lower() == status.lower()]
    if priority:
        filtered_tickets = [t for t in filtered_tickets if t.get('priority', '').lower() == priority.lower()]
    print("filtered_tickets: ",len(filtered_tickets))
    return filtered_tickets

@app.get("/api/tickets/{ticket_id}", response_model=Optional[TicketResponseSmall])
async def get_ticket(ticket_id: str = Path(..., title="The ID of the ticket to get")):
    """Get a specific ticket by ID"""
    ticket = next((t for t in tickets_db if t.get('id', '').lower() == ticket_id.lower()), None)
    if ticket:
        return ticket
    raise HTTPException(status_code=404, detail="Ticket not found")

@app.post("/api/tickets", response_model=TicketResponseSmall, status_code=201)
async def create_ticket(ticket: TicketCreate = Body(...)):
    """Create a new ticket"""
    new_ticket = ticket.dict()
    new_ticket["id"] = str(uuid.uuid4())
    new_ticket["created_at"] = datetime.datetime.now()
    tickets_db.append(new_ticket)
    return new_ticket

@app.put("/api/tickets/{ticket_id}", response_model=Optional[TicketResponseSmall])
async def update_ticket(ticket_id: str = Path(..., title="The ID of the ticket to update"), updates: TicketUpdate = Body(...)):
    """Update an existing ticket"""
    ticket_index = next((i for i, t in enumerate(tickets_db) if t.get('id', '').lower() == ticket_id.lower()), None)
    if ticket_index is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    for key, value in updates.dict(exclude_unset=True).items():
        if key != 'id' and key != 'created_at':
            tickets_db[ticket_index][key] = value

    return tickets_db[ticket_index]


def get_upstream_dependencies_recursive(data, cmdbci_id):
    """
    Recursively finds all upstream dependent IDs for a given CMDBCI ID.

    Args:
        data: A list of dictionaries representing the CMDB data.
        cmdbci_id: The ID for which to find upstream dependencies.

    Returns:
        A set of all upstream dependent IDs.
    """
    dependencies = set()
    item_map = {item['id']: item for item in data}

    def _find_upstream(current_id, visited):
        if current_id in visited:
            return set()
        visited.add(current_id)
        upstream_deps = set()
        
        item = item_map.get(current_id)
        if item:
            upstream_ids = item.get('upstream_ids', '').split(',')
            upstream_ids = [uid.strip() for uid in upstream_ids if uid.strip()]
            
            for upstream_id in upstream_ids:
                if upstream_id and upstream_id not in visited:
                    upstream_deps.add(upstream_id)
                    upstream_deps.update(_find_upstream(upstream_id, visited))
                    
        return upstream_deps

    return _find_upstream(cmdbci_id, set())


# Function to compute embeddings with mean pooling
def get_embedding(text: str):
    """Generates embeddings using mean pooling."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape)
    
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1)
    embedding = sum_embeddings / sum_mask  # Mean pooling

    return embedding.cpu().numpy()

# Define request model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

def get_embedding(text: str) -> np.ndarray:
    """Generates an embedding for a given text."""
    # Replace this with your actual embedding generation logic (e.g., using Sentence Transformers)
    return np.random.rand(128)  # Dummy embedding for demonstration

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

class SearchResponse(BaseModel):
    status_code: int
    results: List[Dict]


# Define generic search function
def faiss_common_search(query: str, index: faiss.Index, common_db: List[dict], entity_key: str, embedding_function: Callable[[str], np.ndarray], top_k: int = 2) -> List[Dict]:
    """
    Search for the most relevant entities using FAISS and return all their fields.

    Args:
        query: The search query string.
        index: The FAISS index to search.
        common_db: The list of dictionaries representing the entities.
        entity_key: The key to use for the entity in the result dictionary (e.g., "incident", "knowledge").
        embedding_function: A function that takes text and returns its embedding.
        top_k: The number of top results to return.

    Returns:
        A list of dictionaries, where each dictionary contains the entity and its similarity score.
    """
    query_embedding = np.array([embedding_function(f"query: {query}")], dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    # Convert L2 distance to similarity scores
    similarity_scores = 1 / (1 + distances)

    results = []
    for rank, i in enumerate(indices[0]):
        if i < len(common_db):  # Ensure index is within bounds
            results.append({
                "entity": common_db[i],
                "similarity_score": float(similarity_scores[0][rank])
            })
    return results

#############################################
### Starting new entitiy for FAISS search ###
#############################################

# Load documents from a SQLite database
kb_db = load_docs_from_db("snow.db", "kb_knowledge")

# Sample IT logs - Store the original incident dictionaries
kb_documents = [k for k in kb_db
             if k["title"] is not None and len(k["title"]) > 0]

# Convert relevant text from kb_documents to embeddings
# kb_embeddings = np.array([get_embedding(k["title"] + ": "+k['content']) for k in kb_documents], dtype="float32").reshape(len(kb_documents), -1)
kb_embeddings = np.array([get_embedding(k["title"]) for k in kb_documents], dtype="float32").reshape(len(kb_documents), -1)
# Initialize FAISS index
index_kb = faiss.IndexFlatL2(kb_embeddings.shape[1])
index_kb.add(kb_embeddings)

# API Endpoint for searching logs
@app.post("/api/mcp_search_knowledge", response_model=SearchResponse)
async def search_knowledge_kb(request: QueryRequest):
    """Retrieve top-k relevant knowledge articles with all their fields for a given query."""
    search_results = faiss_common_search(
        query=request.query,
        index=index_kb,
        common_db=kb_db,
        entity_key="knowledge",
        embedding_function=get_embedding,
        top_k=request.top_k
    )
    return {"status_code": 200, "results": search_results}


#############################################
### Starting new entitiy for FAISS search ###
#############################################

# Load documents from a SQLite database
incident_db = load_docs_from_db("snow.db", "incident")

# Sample IT logs - Store the original incident dictionaries
incident_documents = [k for k in incident_db
             if k["title"] is not None and len(k["title"]) > 0]

# Convert relevant text from kb_documents to embeddings
incident_embeddings = np.array([get_embedding(k["title"] + ": "+k['description']) for k in incident_documents], dtype="float32").reshape(len(incident_documents), -1)

# Initialize FAISS index
index_incident = faiss.IndexFlatL2(incident_embeddings.shape[1])
index_incident.add(incident_embeddings)

# API Endpoint for searching logs
@app.post("/api/mcp_search_incident", response_model=SearchResponse)
async def mcp_search_incident(request: QueryRequest):
    """Retrieve top-k relevant incidents with all their fields for a given query."""
    search_results = faiss_common_search(
        query=request.query,
        index=index_incident,
        common_db=incident_db,
        entity_key="knowledge",
        embedding_function=get_embedding,
        top_k=request.top_k
    )
    return {"status_code": 200, "results": search_results}


#############################################
### Starting new entitiy for FAISS search ###
#############################################

# Load documents from a SQLite database
kp_db = load_docs_from_db("snow.db", "known_problem")

# Sample IT logs - Store the original incident dictionaries
kp_documents = [k for k in kp_db
             if k["name"] is not None and len(k["name"]) > 0]

# Convert relevant text from kb_documents to embeddings
kp_embeddings = np.array([get_embedding(k["name"] + ": "+k['description']) for k in kp_documents], dtype="float32").reshape(len(kp_documents), -1)

# Initialize FAISS index
index_kp = faiss.IndexFlatL2(kp_embeddings.shape[1])
index_kp.add(kp_embeddings)

# API Endpoint for searching logs
@app.post("/api/mcp_search_known_problem", response_model=SearchResponse)
async def mcp_search_known_problem(request: QueryRequest):
    """Retrieve top-k relevant incidents with all their fields for a given query."""
    search_results = faiss_common_search(
        query=request.query,
        index=index_kp,
        common_db=kp_db,
        entity_key="known_problem",
        embedding_function=get_embedding,
        top_k=request.top_k
    )
    return {"status_code": 200, "results": search_results}



#############################################
### Starting new entitiy for FAISS search ###
#############################################

# Load documents from a SQLite database
wa_db = load_docs_from_db("snow.db", "work_around")

# Sample IT logs - Store the original incident dictionaries
wa_documents = [k for k in wa_db
             if k["name"] is not None and len(k["name"]) > 0]

# Convert relevant text from kb_documents to embeddings
wa_embeddings = np.array([get_embedding(k["name"] + ": "+k['description']) for k in wa_documents], dtype="float32").reshape(len(wa_documents), -1)

# Initialize FAISS index
index_wa = faiss.IndexFlatL2(wa_embeddings.shape[1])
index_wa.add(wa_embeddings)

# API Endpoint for searching logs
@app.post("/api/mcp_search_work_around", response_model=SearchResponse)
async def mcp_search_work_around(request: QueryRequest):
    """Retrieve top-k relevant incidents with all their fields for a given query."""
    search_results = faiss_common_search(
        query=request.query,
        index=index_wa,
        common_db=wa_db,
        entity_key="known_problem",
        embedding_function=get_embedding,
        top_k=request.top_k
    )
    return {"status_code": 200, "results": search_results}


#############################################
### Starting new entitiy for FAISS search ###
#############################################

# Load documents from a SQLite database
rd_db = load_docs_from_db("snow.db", "rca_document")

# Sample IT logs - Store the original incident dictionaries
rd_documents = [k for k in rd_db
             if k["root_cause_analysis"] is not None and len(k["root_cause_analysis"]) > 0]

# Convert relevant text from kb_documents to embeddings
rd_embeddings = np.array([get_embedding(k["root_cause_analysis"] + ": "+k['corrective_actions']) for k in rd_documents], dtype="float32").reshape(len(rd_documents), -1)

# Initialize FAISS index
index_rd = faiss.IndexFlatL2(rd_embeddings.shape[1])
index_rd.add(rd_embeddings)

# API Endpoint for searching logs
@app.post("/api/mcp_search_rca_document", response_model=SearchResponse)
async def mcp_search_rca_document(request: QueryRequest):
    """Retrieve top-k relevant incidents with all their fields for a given query."""
    search_results = faiss_common_search(
        query=request.query,
        index=index_rd,
        common_db=rd_db,
        entity_key="known_problem",
        embedding_function=get_embedding,
        top_k=request.top_k
    )
    return {"status_code": 200, "results": search_results}

class QueryRequest(BaseModel):
    ci: str
    


def load_ci_name_mapping(db_name="snow.db"):
    """
    Loads the mapping between configuration item IDs and names from the database.

    Args:
        db_name (str): The name of the SQLite database file.

    Returns:
        dict: A dictionary where keys are CI IDs and values are CI names.
    """
    ci_name_map = {}
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # SQL query to select id and name from configuration_items
        cursor.execute("SELECT id, name FROM configuration_items")
        rows = cursor.fetchall()

        # Populate the dictionary
        for row in rows:
            ci_id, ci_name = row
            ci_name_map[ci_id] = ci_name

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
    return ci_name_map

ci_map = load_ci_name_mapping()

# API Endpoint for searching logs
@app.post("/api/get_ci")
async def get_ci(request: QueryRequest):
    """Retrieve top-k relevant incidents with all their fields for a given query."""
    db_name = "snow.db"  # Ensure this is defined
    ci_db = load_docs_from_db(db_name, "configuration_items")

    target_id = request.ci
    upstream_ids_set = get_upstream_dependencies_recursive(ci_db, target_id)

    print(f"Upstream dependencies for {target_id}: {upstream_ids_set}")

    # ✅ Convert set to a sorted list (ensures order consistency)
    upstream_ids = sorted(upstream_ids_set) if upstream_ids_set else []

    if not upstream_ids:  # Avoid SQL error if the list is empty
        return {"status": 200, "results": "No relevant incidents found."}

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # ✅ Create placeholders dynamically
        placeholders = ', '.join(['?'] * len(upstream_ids))
        query = f"""
        SELECT id, title, description, cmdb_ci_id
        FROM incident
        WHERE cmdb_ci_id IN ({placeholders})
        AND status <> 'Resolved'
        """

        print(f"Executing query: {query} with {upstream_ids}")

        cursor.execute(query, upstream_ids)
        records = cursor.fetchall()

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return {"status": 500, "error": f"Database error: {str(e)}"}

    finally:
        if conn:
            conn.close()

    # Process results
    incident_context = ""
    for record in records:
        try:
            incident_id, title, description, ci = record
            incident_context += f"Incident ID for CI item {ci_map[ci]}: {incident_id}\nTitle: {title}\nDescription: {description}\n\n"
        except ValueError:
            print("Warning: One or more records do not match expected fields. Skipping.")
            continue

    change_context = await search_change_requests(upstream_ids)
    know_problems_context = await search_know_problems(upstream_ids)
    rca_documents_context = await search_work_rca_documents(upstream_ids)
    print("rca_documents_context",rca_documents_context)
    context = ""
    if incident_context!="":
        context = "Incidents related to the CI:\n\n" + incident_context
    if change_context!="":
        context += "\n\nChange Requests related to the CI:\n\n" + change_context
    if know_problems_context!="":
        context += "\n\nKnown Problems related to the CI:\n\n" + know_problems_context
    if rca_documents_context!="":
        context += "\n\nRoot Cause Analysis Documents related to the CI:\n\n" + rca_documents_context

    return {"status": 200, "results": context or "No relevant incidents found."}


async def search_work_rca_documents(upstream_ids):
    """Retrieve relevant change requests for a given configuration item."""
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?'] * len(upstream_ids))
        query = f"""
        SELECT id, cmdb_ci_id
        FROM incident
        WHERE cmdb_ci_id IN ({placeholders})
        """
        
        print(f"Executing query: {query} with {upstream_ids}")
        cursor.execute(query, upstream_ids)
        incident_records = cursor.fetchall()
    
    except sqlite3.Error as e:
        return ""

    try:
        incident_ids = [str(record[0]) for record in incident_records]
        ci_incidents = {record[0]: record[1] for record in incident_records}
        if not incident_ids:
            return ""
        
        # Query RCA documents for the incidents
        placeholders = ', '.join(['?'] * len(incident_ids))
        query = f"""
        SELECT incident_id, timeline_of_events, root_cause_analysis, contributing_factors, corrective_actions, follow_up, created_date
        FROM rca_document
        WHERE incident_id IN ({placeholders})
        """
        
        print(f"Executing query: {query} with {incident_ids}")
        cursor.execute(query, incident_ids)
        rca_records = cursor.fetchall()

    except sqlite3.Error as e:
        return {"status": 500, "error": f"Database error: {str(e)}"}
    
    finally:
        if conn:
            conn.close()

    rca_context = ""
    for record in rca_records:
        try:
            (incident_id, timeline_of_events, root_cause_analysis, contributing_factors, 
             corrective_actions, follow_up, created_date) = record
            rca_context += f"""Incident ID for {ci_map[ci_incidents[incident_id]]}: {incident_id}\n
                        Timeline of Events: {timeline_of_events}
                        Root Cause Analysis: {root_cause_analysis}+
                        Contributing Factors: {contributing_factors}+
                        Corrective Actions: {corrective_actions}+
                        Follow Up: {follow_up}+
                        Created Date: {created_date}\n\n\n"""
        except ValueError:
            print("Warning: One or more records do not match expected fields. Skipping.")
            continue


    return rca_context or ""

async def search_know_problems(upstream_ids):
    """Retrieve relevant change requests for a given configuration item."""
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?'] * len(upstream_ids))
        query = f"""
        SELECT id, name, description, cmdb_ci_id, created_date
        FROM known_problem
        WHERE cmdb_ci_id IN ({placeholders})
        """
        
        print(f"Executing query: {query} with {upstream_ids}")
        cursor.execute(query, upstream_ids)
        records = cursor.fetchall()
    
    except sqlite3.Error as e:
        return {"status": 500, "error": f"Database error: {str(e)}"}
    
    finally:
        if conn:
            conn.close()
    
    context = ""
    for record in records:
        try:
            cr_id, name, description, ci, created_date = record
            context += f"""Known Problem Request ID for {ci_map[ci]}: {cr_id}\n
                        f"Name: {name}\n
                        f"Description: {description}\n
                        f"Created Date: {created_date}\n\n"""
        except ValueError:
            print("Warning: One or more records do not match expected fields. Skipping.")
            continue
    
    return context or ""

async def search_change_requests(upstream_ids):
    """Retrieve relevant change requests for a given configuration item."""
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        placeholders = ', '.join(['?'] * len(upstream_ids))
        query = f"""
        SELECT id, name, description, cmdb_ci_id, created_date
        FROM change_request
        WHERE cmdb_ci_id IN ({placeholders})
        """
        
        print(f"Executing query: {query} with {upstream_ids}")
        cursor.execute(query, upstream_ids)
        records = cursor.fetchall()
    
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return {"status": 500, "error": f"Database error: {str(e)}"}
    
    finally:
        if conn:
            conn.close()
    
    context = ""
    for record in records:
        try:
            cr_id, name, description, ci, created_date = record
            context += f"""Change Request ID for {ci_map[ci]}: {cr_id}\n
                        f"Name: {name}\n
                        f"Description: {description}\n
                        f"Created Date: {created_date}\n\n"""
        except ValueError:
            print("Warning: One or more records do not match expected fields. Skipping.")
            continue
    
    return context or ""


if __name__ == "__main__":
    print("MCP Server running on http://localhost:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001)  # Use uvicorn.run
