from fastapi import FastAPI
import uvicorn
import pandas as pd
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",  # Allow requests from your frontend
    # Add other origins if needed
]
# Apply nest_asyncio to avoid runtime conflicts in Jupyter
nest_asyncio.apply()

# ✅ Sample data from the table
data = [
    {"id": "CMDBCI0001", "name": "Web Server 01", "description": "Primary web server for customer portal", "host": "web01.example.com", "owner": "IT Operations"},
    {"id": "CMDBCI0002", "name": "Database Server 02", "description": "Secondary database server for application data", "host": "db02.example.com", "owner": "Database Admins"},
    {"id": "CMDBCI0003", "name": "Network Switch 03", "description": "Core network switch in the data center", "host": "switch03.example.com", "owner": "Network Engineers"},
]

# ✅ Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Function to simulate scenario based on description
def simulate_scenario(desc):
    if "database" in desc.lower():
        return {
            "status": "Memory usage is 85%",
            "action": "Auto-trigger cleanup and archive logs"
        }
    elif "web server" in desc.lower():
        return {
            "status": "Traffic spike detected",
            "action": "Auto-scale by adding 2 instances"
        }
    elif "network switch" in desc.lower():
        return {
            "status": "High packet loss detected",
            "action": "Restart switch port and run diagnostic"
        }
    else:
        return {
            "status": "No critical issues detected",
            "action": "Monitoring continues"
        }

# ✅ API to filter by ID and trigger automated scenario
@app.get("/get-details/{cmdb_id}")
def get_details(cmdb_id: str):
    filtered_data = df[df['id'] == cmdb_id].to_dict(orient='records')
    if not filtered_data:
        return {"error": "ID not found"}
    
    details = filtered_data[0]
    scenario = simulate_scenario(details['description'])
    
    return {
        "id": details['id'],
        "name": details['name'],
        "host": details['host'],
        "owner": details['owner'],
        "scenario": scenario
    }

# ✅ Start the server
uvicorn.run(app, host="127.0.0.1", port=8001)