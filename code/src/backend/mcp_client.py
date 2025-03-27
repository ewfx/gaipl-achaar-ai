import uvicorn
from typing import Any, Dict, List, Optional, Union
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os
from autogen import ConversableAgent
import requests
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel # For type hinting
import asyncio
from fastapi.middleware.cors import CORSMiddleware


# Define the path or environment variable for the configuration list
config_path = os.getenv("OAI_CONFIG_LIST", "OAI_CONFIG_LIST")  # Ensure correct path or environment variable
if not config_path:
    raise ValueError("OAI_CONFIG_LIST environment variable is not set or empty.")

# Load configuration for the Gemini model
config_list_gemini = config_list_from_json(
    config_path,
    filter_dict={"model": ["gemini-2.0-flash"]},
)
config_list_openai = config_list_from_json(
    config_path,
    filter_dict={"model": ["gpt-4o-mini"]},
)


# MCP Server interaction functions
MCP_SERVER_URL = "http://127.0.0.1:5001/api/"  # Base URL for MCP server
TICKETS_ENDPOINT = MCP_SERVER_URL+"tickets"  # Endpoint for tickets
def get_tickets(status: Optional[str] = None, priority: Optional[str] = None) -> List[Dict]: # Added type hint
    """
    Get tickets from the MCP server, optionally filtered by status and priority.
    Returns the JSON response from the server.
    """
    params = {}
    if status:
        params['status'] = status
    if priority:
        params['priority'] = priority
    try:
        response = requests.get(TICKETS_ENDPOINT, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get tickets: {e}"}

def get_ticket_by_id(ticket_id: str) -> Dict: # Added type hint
    """
    Get a specific ticket by ID from the MCP server.
    Returns the JSON response from the server.
    """
    url = f"{TICKETS_ENDPOINT}/{ticket_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get ticket {ticket_id}: {e}"}


def get_plain_answer(agent, content):
    try:
        message = [{"content": content, "role": "user"}]
        agent_result = agent.generate_reply(messages=message)
        print("Successfully parsed items:", agent_result)
        if isinstance(agent_result, dict) and "content" in agent_result:
            return agent_result['content']
        else:
            return agent_result
    except Exception as e:
        print(f"Failed to query agent: {str(e)}")
        return ""


rca_summary_agent = ConversableAgent(
    "rca_agent",
    system_message=f"""You are an expert summarization Agent. Your task is to take a Probable Root Cause Analysis (RCA) document as input and summarize it concisely. The RCA document contains detailed information about an incident, its root cause analysis, contributing factors, corrective actions, and follow-up plans. Your goal is to extract the key information from each section and provide a brief summary of the incident and its resolution.

Identify the key information from each section of the RCA document provided below and synthesize it into a brief summary. Ensure the summary captures the main incident, the findings of the RCA, the actions taken, and any follow-up plans. Also, pay attention to any discrepancies or mismatches highlighted in the RCA document and include that in the summary.

""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER"
)


def get_rca_summary(content: str) -> Dict: # Added type hint
    """
    Get a concise summary of a Probable Root Cause Analysis (RCA) document.
    """
    try:
        response = get_plain_answer(rca_summary_agent, content)
        return response
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to generate summary: {e}"}
    


rca_agent = ConversableAgent(
    "rca_agent",
    system_message=f"""You are an expert Agent designed to take input content and format it into a well-structured HTML document for display in a browser.

Your goal is to present the information clearly and professionally in the following HTML format, adapting the structure based on the type of content provided.

**Instructions:**

1.  **Identify Content Type:** First, analyze the input content to determine its nature. Is it primarily a description of an incident and its analysis (similar to a Portable Root Cause Analysis), or is it more of a knowledge base article focusing on a specific problem and solution?

2.  **HTML Structure:** Structure the output using standard HTML tags.

3.  **Section Titles (<h2>):** Use `<h2>` tags for the main sections. The specific section titles will depend on the content type:

    * **For Incident/PRCA-related content:** Use the following section titles: "Incident Description", "Timeline of Events", "Root Cause Analysis", "Contributing Factors", "Corrective Actions", and "Follow Up".
    * **For Knowledge Base (KB) article-related content:** Use section titles that are relevant to the content. Examples include: "Problem Description", "Symptoms", "Cause", "Resolution", "Workaround", "Applies To", etc. Choose titles that accurately reflect the information provided in the input.

4.  **Main Content (<p>):** Use `<p>` tags for the main content within each section.

5.  **Lists (<ul> and <li>):** If any section (like "Timeline of Events", "Corrective Actions", "Follow Up" in incident reports, or "Symptoms", "Resolution Steps" in KB articles) contains multiple points, format them as an unordered list using `<ul>` and `<li>` tags.

6.  **Incident Discrepancy (<b>):** If the input content refers to a different incident than the one described in a potential "Incident Summary" within the input, clearly indicate this within each relevant section using `<b>` tags for emphasis. For example: `<p>The following timeline of events pertains to <b>Incident ID: [Specify Incident ID from the input]</b>...</p>`.

7.  **Formatting and Readability:** Ensure that the resulting HTML document is well-formatted and easy to read, with appropriate spacing and indentation in the HTML structure.

""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER"
)


def probable_rca(ticket_id: str) -> Dict:
    ticket = get_ticket_by_id(ticket_id)
    if "error" in ticket:
        return ticket  # Propagate error if fetching ticket failed
    print(ticket, ticket['cmdb_ci_id'])
    try:
        response = get_ci_related_info(ticket['cmdb_ci_id'])  # Likely returns dict already
        # content = get_plain_answer(rca_agent, response['results'])
        # return content 
        return response 
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get CI related info for ticket {ticket_id}: {e}"}



def create_ticket(title: str, description: str, status: str, priority: str, assigned_to: str) -> Dict: # Added type hint
    """
    Create a new ticket on the MCP server.
    Returns the JSON response from the server.
    """
    headers = {'Content-Type': 'application/json'}
    data = {
        "title": title,
        "description": description,
        "status": status,
        "priority": priority,
        "assigned_to": assigned_to
    }
    try:
        response = requests.post(TICKETS_ENDPOINT, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to create ticket: {e}"}

def update_ticket(ticket_id: str, updates: dict) -> Dict: # Added type hint
    """
    Update an existing ticket on the MCP server.
    'updates' should be a dictionary containing fields to update (e.g., {'status': 'closed'}).
    Returns the JSON response from the server.
    """
    url = f"{TICKETS_ENDPOINT}/{ticket_id}"
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.put(url, headers=headers, data=json.dumps(updates))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to update ticket {ticket_id}: {e}"}


def search_knowledge_base(query: str, top_k: int = 2) -> List[Dict]:
    """Searches the knowledge base for relevant articles."""
    headers = {'Content-Type': 'application/json'}
    data = {"query": query, "top_k": top_k}
    try:
        response = requests.post(MCP_SERVER_URL+"mcp_search_knowledge", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return [{"error": f"Failed to search knowledge base: {e}"}]  # Return error as a list of dict


def get_ci_related_info(ci_id:str) -> dict:
    """
    Retrieves information related to a Configuration Item (CI) from the MCP server.

    Args:
        ci_id (str): The ID of the Configuration Item to search for.

    Returns:
        dict: The JSON response from the server.
    """
    api_endpoint = f"{MCP_SERVER_URL}get_ci"
    headers = {'Content-Type': 'application/json'}
    payload = {'ci': ci_id}

    try:
        response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response from the server.")
        return None
    

# Define Assistant Agent
assistant = ConversableAgent(
    name="MCP_Assistant",
    system_message="""You are a helpful AI assistant designed to interact with the MCP (MyCoolProject) server API.
    You can perform actions like retrieving tickets, creating new tickets, and updating existing tickets.
    Use the provided tools to interact with the MCP server based on user requests.
    When you have successfully addressed the user's request and provided a response, or if you cannot fulfill the request, respond with 'TERMINATE' to end the conversation.
    If you encounter errors from the API, inform the user and suggest alternative actions or clarifications.
    """,
    llm_config={"config_list":config_list_openai}
)

# Define User Proxy Agent
user_proxy = UserProxyAgent(
    name="MCP_User_Proxy",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",  # Set to "ALWAYS" if you want human intervention
    # max_auto_reply=10, # set max auto reply to a reasonable number
    code_execution_config={"use_docker": False}, # set to True or False based on your requirement
)
assistant.register_for_llm(name="get_tickets", description="Gets a list of tickets, optionally filtered by status and priority.")(get_tickets)
assistant.register_for_llm(name="get_ticket_by_id", description="Gets a specific ticket by its ID.")(get_ticket_by_id)
assistant.register_for_llm(name="create_ticket", description="Creates a new ticket.")(create_ticket)
assistant.register_for_llm(name="update_ticket", description="Updates an existing ticket by ID.")(update_ticket)
assistant.register_for_llm(name="search_knowledge_base", description="Searches the knowledge base for given query and returns top-k results.")(search_knowledge_base)

assistant.register_for_llm(name="get_ci_related_info", description="Retrieves information related to a Configuration Item (CI) from the MCP server..")(get_ci_related_info)
assistant.register_for_llm(name="probable_rca", description="Generates Probable Root cause Analysis (RCA) based on incident id from the MCP server..")(probable_rca)

assistant.register_for_llm(name="get_rca_summary", description="Generates summary of Probable Root cause analysis (RCA) document")(get_rca_summary)


user_proxy.register_for_execution(name="get_tickets")(get_tickets)
user_proxy.register_for_execution(name="get_ticket_by_id")(get_ticket_by_id)
user_proxy.register_for_execution(name="create_ticket")(create_ticket)
user_proxy.register_for_execution(name="update_ticket")(update_ticket)
user_proxy.register_for_execution(name="search_knowledge_base")(search_knowledge_base)
user_proxy.register_for_execution(name="get_ci_related_info")(get_ci_related_info)
user_proxy.register_for_execution(name="probable_rca")(probable_rca)
user_proxy.register_for_execution(name="get_rca_summary")(get_rca_summary)



# # user_input = "get list of tickets"
# user_input = "get high priority tickets"
# # if user_input.lower() == 'exit':
# #     break

# chat_result = user_proxy.initiate_chat(assistant, message=user_input)

# # Process and print the last response from the assistant (which should contain the final output)
# last_message = chat_result.chat_history[-1]
# if last_message and last_message['role'] == 'assistant':
#     print(f"Assistant: {last_message['content']}")
# else:
#     print("No response from assistant.")


app = FastAPI()

origins = [
    "http://localhost:3000",  # Allow requests from your frontend
    "http://localhost:8000", 
    "ws://localhost:5002", 
    # Add other origins if needed
]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
connected_clients: List[WebSocket] = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # <-- Frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

# --- WebSocket connection manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send message: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def format_tool_event(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content  # fallback if not JSON
    
    # Case 1: Knowledge base search failed
    if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
        return "🔍 Searching knowledge base... not found. Proceeding to create a ticket."

    # Case 2: Found related incidents
    if isinstance(data, list) and len(data) > 0 and "id" in data[0] and data[0].get("title"):
        incidents = "\n".join([f"- {inc['id']}: {inc['title']}" for inc in data])
        return f"✅ Found related incidents:\n{incidents}"

    # Case 3: Tool responded with a ticket object
    if isinstance(data, dict) and data.get("id"):
        if data.get("title"):  # Means this is likely an existing ticket (found or confirmed)
            return f"🟣 Confirmed existing ticket `{data['id']}`: {data.get('title', 'No title')}"
        else:  # title is None/null => likely freshly created
            return f"🎫 Created a new incident ticket with ID `{data['id']}`."

    return "ℹ️ Tool responded, but no user-friendly message is available."

# --- MCP Chat API ---
@app.post("/mcp_chat")
async def mcp_chat(request: ChatRequest):
    try:
        # Call your existing MCP Assistant
        chat_result = user_proxy.initiate_chat(assistant, message=request.user_input)

        # --- Dynamically broadcast any tool-related events in the chat history ---
        for message in chat_result.chat_history:
            if message.get("role") == "tool":
                formatted = format_tool_event(message)
                await manager.broadcast({"type": "tool_event", "data": formatted})
        # --- Broadcast the assistant response ---
        last_message = chat_result.chat_history[-1] if chat_result.chat_history else None
        if last_message:
            assistant_response = last_message.get("content", "")
            content = get_plain_answer(rca_agent, assistant_response)
            print("assistant_response", assistant_response) 
            print("rca content", content)
            await manager.broadcast({"type": "assistant_message", "data": content})
            if assistant_response.strip() == "TERMINATE":
                return {"message": "Conversation terminated successfully."}
            return {"status": 200, "message": content}
        else:
            raise HTTPException(status_code=500, detail="No response from assistant.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    


@app.post("/mcp_chat2", response_model=dict)  # Use lowercase dict for the response model
async def mcp_chat(request: ChatRequest): # Use Pydantic model to receive input
    """Interacts with the MCP assistant to process user requests."""

    try:
        chat_result = user_proxy.initiate_chat(assistant, message=request.user_input)  # Access user_input from request

        # Extract the last assistant message (or handle other response structures as needed)
        last_message = chat_result.chat_history[-1]
        print(chat_result)
        print(last_message)
        # if last_message and 'assistant'.lower() in last_message['role'].lower() :
        if last_message:
            assistant_response = last_message['content']
            if assistant_response == "TERMINATE": # Assuming assistant uses "TERMINATE" only when done
                return {"message": "Conversation terminated successfully."} # Clear termination signal
            return {"status":200, "message": assistant_response} # If assistant produced something besides termination
        else:
            raise HTTPException(status_code=500, detail="No response from assistant.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

### Agents ###


incident_analyser = ConversableAgent(
    "incident_analyser",
    system_message=f"""You are an AI agent specializing in the initial triage of platform support incidents within ServiceNow. Your primary goal is to quickly understand the user's issue based on the description, automatically categorize the incident using the defined ServiceNow categories, determine the incident's priority based on the impact and urgency (if provided, or inferring based on the description), and assign it to the most relevant support group or agent based on the categorization and keywords. If the description is unclear, ask clarifying questions to gather necessary information for accurate triage. Provide a summary of your triage decisions and the assigned group/agent.""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER",  # Never ask for human input.
)


kb_analyser = ConversableAgent(
    "kb_analyser",
    system_message=f""""You are an AI agent designed to assist support agents by suggesting relevant knowledge base articles. When an agent is working on an incident, analyze the incident details (description, category, etc.) and search the knowledge base for potentially helpful articles. Provide a list of the top relevant articles with a brief summary of each to the agent. Your goal is to help agents find solutions quickly and efficiently.""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER",  # Never ask for human input.
)


problem_analyser = ConversableAgent(
    "problem_analyser",
    system_message=f"""You are an AI agent focused on problem management. Analyze historical incident data to identify trends and recurring issues. When a potential problem is identified (based on the number of related incidents or severity), create a problem record in ServiceNow and provide an initial analysis of the potential root causes based on the incident data. Suggest potential areas for further investigation to the problem management team. Your goal is to proactively address underlying issues to prevent future incidents.""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER",  # Never ask for human input.
)


change_request_analyser = ConversableAgent(
    "change_request_analyser",
    system_message=f""""You are an AI agent specializing in change request risk assessment. When presented with a new change request, analyze the description of the change, the affected components, the planned implementation steps, and the rollback plan. Based on this information and historical data on similar changes, identify potential risks and assess the potential impact of the change on the platform's stability and functionality. Provide a summary of your risk assessment and suggest any necessary precautions or considerations for the change approvers.""",
    llm_config={"config_list":config_list_openai},
    human_input_mode="NEVER",  # Never ask for human input.
)



print("MCP Chat Server running on http://127.0.0.1:5002")  # Different port from your other server
uvicorn.run(app, host="0.0.0.0", port=5002)
