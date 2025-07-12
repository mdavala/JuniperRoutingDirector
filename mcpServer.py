from fastmcp import FastMCP
import httpx
import sys, re
import json
import base64
import logging
import asyncio
import traceback
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from urllib.parse import urlencode
from dotenv import load_dotenv
from openai import OpenAI
import time

from l3vpn_parser import parse_l3vpn_json
from l2ckt_parser import parse_l2circuit_json
from l2vpn_evpn_parser import parse_evpn_json
from services_generator import EnhancedServiceConfigGenerator
from junos_cli_generator import JunosCLIGenerator

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the enhanced service generator
try:
    SERVICE_GENERATOR_AVAILABLE = True
    logger.info("Enhanced service generator module imported successfully")
except ImportError as e:
    SERVICE_GENERATOR_AVAILABLE = False
    logger.warning(f"Enhanced service generator not available: {e}. Create services_generator.py file.")
except Exception as e:
    SERVICE_GENERATOR_AVAILABLE = False
    logger.error(f"Error importing enhanced service generator: {e}")

# Import JUNOS CLI generator
try:
    CLI_GENERATOR_AVAILABLE = True
    logger.info("JUNOS CLI generator module imported successfully")
except ImportError as e:
    CLI_GENERATOR_AVAILABLE = False
    logger.warning(f"JUNOS CLI generator not available: {e}. Create junos_cli_generator.py file.")
except Exception as e:
    CLI_GENERATOR_AVAILABLE = False
    logger.error(f"Error importing JUNOS CLI generator: {e}")

# Create an MCP server
mcp = FastMCP("Routing Director MCP server with GPT-4 Intelligence and Enhanced Service Creation")
BASE_URL = "https://66.129.234.204:48800"
ORG_ID = os.getenv('ORG_ID', "0eaf8613-632d-41d2-8de4-c2d242325d7e")

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

# Initialize Enhanced Service Generator
enhanced_service_generator = None
if SERVICE_GENERATOR_AVAILABLE:
    try:
        enhanced_service_generator = EnhancedServiceConfigGenerator()
        logger.info("Enhanced Service Config Generator initialized successfully")
        
        # Test if data files are available
        if not enhanced_service_generator.customers_data:
            logger.warning("No customer data loaded - check services/customers.json file")
        else:
            logger.info(f"Loaded {len(enhanced_service_generator.customers_data)} customers")
            
        devices_count = len(enhanced_service_generator.devices_data.get('devices', []))
        if devices_count == 0:
            logger.warning("No device data loaded - check services/devices.json file")
        else:
            logger.info(f"Loaded {devices_count} devices")
            
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Service Config Generator: {e}")
        enhanced_service_generator = None
        SERVICE_GENERATOR_AVAILABLE = False

# Initialize JUNOS CLI Generator
cli_generator = None
if CLI_GENERATOR_AVAILABLE:
    try:
        cli_generator = JunosCLIGenerator()
        if cli_generator.available:
            logger.info("JUNOS CLI Generator initialized successfully")
        else:
            logger.warning("JUNOS CLI Generator initialized but OpenAI not available")
    except Exception as e:
        logger.error(f"Failed to initialize JUNOS CLI Generator: {e}")
        cli_generator = None
        CLI_GENERATOR_AVAILABLE = False

# Service type constants
SERVICE_TYPES = {
    "l3vpn": {
        "name": "L3VPN",
        "parser": parse_l3vpn_json,
        "keywords": ["l3vpn", "layer 3", "l3", "vpn", "routing", "ip vpn", "mpls vpn"]
    },
    "l2circuit": {
        "name": "L2 Circuit", 
        "parser": parse_l2circuit_json,
        "keywords": ["l2circuit", "l2 circuit", "layer 2 circuit", "l2ckt", "circuit", "pseudowire", "pw"]
    },
    "evpn": {
        "name": "L2VPN EVPN",
        "parser": parse_evpn_json, 
        "keywords": ["evpn", "l2vpn evpn", "ethernet vpn", "bgp evpn", "vxlan", "layer 2 vpn"]
    }
}

class APIEndpoint:
    """Class to represent API endpoint information"""
    def __init__(self, path: str, method: str, description: str, required_params: List[str] = None, optional_params: List[str] = None):
        self.path = path
        self.method = method
        self.description = description
        self.required_params = required_params or []
        self.optional_params = optional_params or []

# Define available endpoints for Routing Director
ENDPOINTS = {
    "get_instances": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order/instances",
        "GET",
        "Get all Service Instances for an Organization. Includes meta information and status from the latest Service Order per Service Instance. Use pagination for large datasets and filters to limit results.",
        required_params=["org_id"],
        optional_params=["per-page", "current-offset", "filter"]
    ),
    "get_orders": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order/orders",
        "GET",
        "Get all service orders for an Organization including history. Contains multiple entries for same Service Instance if multiple orders were executed. Use pagination and filters to manage large datasets.",
        required_params=["org_id"],
        optional_params=["per-page", "current-offset", "filter"]
    ),
    "create_order": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order",
        "POST",
        "Create a new service order. Requires JSON payload with service configuration.",
        required_params=["org_id"],
        optional_params=[]
    ),
    "execute_order": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order/customers/{customer_id}/instances/{instance_id}/exec",
        "POST",
        "Execute a created service order to provision the service.",
        required_params=["org_id", "customer_id", "instance_id"],
        optional_params=[]
    ),
    "get_instance": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order/customers/{customer_id}/instances/{instance_id}",
        "GET",
        "Get detailed status and information for a specific service instance.",
        required_params=["org_id", "customer_id", "instance_id"],
        optional_params=[]
    ),
    # ADD THIS NEW ENDPOINT
    "get_customers": APIEndpoint(
        "/service-orchestration/api/v1/orgs/{org_id}/order/customers",
        "GET",
        "Get all customers for an Organization with their customer IDs and details.",
        required_params=["org_id"],
        optional_params=["per-page", "current-offset", "filter"]
    ),
}

class ParagonAuth:
    """Handle authentication for Paragon Active Assurance API"""
    
    def __init__(self):
        self.username = os.getenv('USERNAME')
        self.password = os.getenv('PASSWORD')
        self.token = None
        self.token_expiry = None
        
        if not self.username or not self.password:
            logger.error("USERNAME and PASSWORD must be set in .env file")
            raise ValueError("Missing credentials in .env file")
        
        logger.info(f"Authentication configured for user: {self.username}")
    
    def get_basic_auth_header(self) -> str:
        """Generate Basic Authentication header"""
        if not self.username or not self.password:
            raise ValueError("Username and password are required for authentication")
        
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"

    def get_auth_token(self) -> Optional[str]:
        """Get authentication token from API (if token-based auth is used)"""
        try:
            auth_url = f"{BASE_URL}/active-assurance/api/v2/auth/token"
            
            auth_payload = {
                "username": self.username,
                "password": self.password
            }
            
            with httpx.Client(verify=False, timeout=30.0) as client:
                response = client.post(
                    auth_url,
                    json=auth_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.token = token_data.get('access_token')
                    logger.info("Token authentication successful")
                    return self.token
                else:
                    logger.error(f"Token authentication failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Token authentication error: {e}")
            return None
    
    def get_headers(self, use_basic_auth: bool = True) -> Dict[str, str]:
        """Get headers with authentication"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if use_basic_auth:
            # Use Basic Authentication
            headers["Authorization"] = self.get_basic_auth_header()
        elif self.token:
            # Use Bearer token if available
            headers["Authorization"] = f"Bearer {self.token}"
        
        return headers

# Initialize authentication
try:
    auth = ParagonAuth()
except ValueError as e:
    logger.error(f"Authentication initialization failed: {e}")
    auth = None

def dataframe_to_json_serializable(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert pandas DataFrame to a JSON-serializable format that can be easily reconstructed
    """
    if df is None or df.empty:
        return {"data": [], "columns": [], "index": []}
    
    return {
        "data": df.to_dict('records'),  # Convert to list of dictionaries
        "columns": list(df.columns),
        "index": list(df.index),
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def detect_service_type_from_query(user_query: str) -> str:
    """
    Detect service type from user query based on keywords
    """
    query_lower = user_query.lower()
    
    # Check for specific service type keywords
    for service_type, service_info in SERVICE_TYPES.items():
        for keyword in service_info["keywords"]:
            if keyword in query_lower:
                logger.info(f"Detected service type '{service_type}' from keyword '{keyword}'")
                return service_type
    
    # Default to l2circuit for now
    logger.info("No specific service type detected, defaulting to l2circuit")
    return "l2circuit"

def is_service_creation_request(user_query: str) -> bool:
    """Check if the user query is requesting service creation"""
    creation_keywords = [
        "create", "provision", "deploy", "add", "new", "generate", 
        "establish", "set up", "build", "configure"
    ]
    
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in creation_keywords)

def is_service_deletion_request(user_query: str) -> bool:
        """Check if the user query is requesting service deletion"""
        deletion_keywords = [
            "delete", "remove", "destroy", "terminate", "decommission", 
            "tear down", "uninstall", "drop", "cancel", "disable"
        ]
    
        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in deletion_keywords)

def analyze_query_with_gpt4(user_query: str) -> Dict[str, Any]:
    """
    Use GPT-4 to analyze user query and determine the best tool to call and service type
    """
    if not openai_client:
        return {
            "error": "OpenAI client not configured. Check OPENAI_API_KEY in .env file",
            "tool": "get_api_endpoints",
            "service_type": "l2circuit"
        }
    
    # Check if this is a service creation or deletion request
    is_creation = is_service_creation_request(user_query)
    is_deletion = is_service_deletion_request(user_query)
    
    # Get available tools with their descriptions
    tools_info = []
    tool_functions = {
        "fetch_all_instances": {
            "description": "Fetches Service Instances from Routing Director API. Use when user wants to see instances, services, or overall status. Can fetch specific service types (L3VPN, L2 Circuit, EVPN).",
        },
        "fetch_all_orders": {
            "description": "Fetches all Service Orders including history. Use when user wants to see order history, operations, or audit trails.",
        },
        "get_instance_details": {
            "description": "Get detailed information about a specific instance. Use when user wants complete details about one particular instance.",
        },
        "create_service_intelligent": {
            "description": "Create a new service using natural language processing. Use when user wants to create, provision, or deploy a new service with natural language descriptions. This will analyze requirements and generate forms if needed.",
        },
        "delete_service": {
            "description": "Delete an existing service by name using intelligent analysis. Use when user wants to delete, remove, terminate, or decommission a service. Handles natural language deletion requests and provides confirmation workflow.",
        },
        "fetch_all_customers": {
            "description": "Fetch all customers with their IDs and details. Use when you need customer information or as part of service operations.",
        },
        "get_api_endpoints": {
            "description": "List available API endpoints. Use when user asks about available APIs or system capabilities.",
        }
    }
    
    for tool_name, tool_info in tool_functions.items():
        tools_info.append(f"- **{tool_name}**: {tool_info['description']}")
    
    # Build service types description
    service_types_desc = []
    for service_type, service_info in SERVICE_TYPES.items():
        keywords_str = ", ".join(service_info["keywords"][:3])  # Show first 3 keywords
        service_types_desc.append(f"- **{service_type}** ({service_info['name']}): Keywords like {keywords_str}")
    
    # Adjust prompt based on whether this looks like a creation or deletion request
    if is_creation:
        operation_note = """
IMPORTANT: This query appears to be requesting SERVICE CREATION. If the user is asking to create, provision, deploy, or add new services, use the 'create_service_intelligent' tool which can handle natural language service creation requests and will prompt for additional details if needed.
"""
    elif is_deletion:
        operation_note = """
IMPORTANT: This query appears to be requesting SERVICE DELETION. If the user is asking to delete, remove, terminate, or decommission services, use the 'delete_service' tool which can handle natural language service deletion requests with intelligent analysis and confirmation workflow.
"""
    else:
        operation_note = ""
    
    prompt = f"""
You are an intelligent query analyzer for a Routing Director MCP server. Analyze the user's query and determine:
1. The best tool to call
2. The service type the user is interested in

USER QUERY: "{user_query}"

{operation_note}

AVAILABLE TOOLS:
{chr(10).join(tools_info)}

AVAILABLE SERVICE TYPES:
{chr(10).join(service_types_desc)}

ANALYSIS GUIDELINES:
1. If user asks about "instances", "services", "status", or wants to see all data -> use fetch_all_instances
2. If user asks about "orders", "history", "operations", "audit" -> use fetch_all_orders  
3. If user wants details about one specific instance -> use get_instance_details
4. If user wants to "create", "provision", "deploy", "add", "new service" -> use create_service_intelligent
5. If user wants to "delete", "remove", "terminate", "decommission" service -> use delete_service
6. If user asks about "customers" or needs customer information -> use fetch_all_customers
7. If user asks about "help", "capabilities", "endpoints", "APIs" -> use get_api_endpoints

SERVICE TYPE DETECTION:
- Analyze keywords to determine if user is asking about L3VPN, L2 Circuit, or EVPN services
- If no specific service type mentioned, default to "l2circuit" for create/delete operations, "l3vpn" for read operations
- Look for variations like "layer 3", "l3", "circuit", "evpn", "ethernet vpn", etc.

IMPORTANT GUIDELINES:
- For get_instance_details: extract the exact instance id mentioned
- For create_service_intelligent: determine the service type from keywords
- For delete_service: the tool will intelligently extract service name from natural language using GPT-4
- Always specify the detected service_type

Respond with a JSON object in this format:
{{
    "reasoning": "Brief explanation of why this tool and service type were chosen",
    "tool": "tool_name",
    "service_type": "l3vpn|l2circuit|evpn"
}}

Be precise with tool and service type selection.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise query analyzer that responds only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        gpt_response = response.choices[0].message.content.strip()
        logger.info(f"GPT-4 Response: {gpt_response}")
        
        try:
            analysis = json.loads(gpt_response)
            
            if analysis.get("service_type") not in SERVICE_TYPES:
                if analysis.get("tool") in ["create_service_intelligent", "delete_service"]:
                    analysis["service_type"] = "l2circuit"
                else:
                    analysis["service_type"] = "l3vpn"
            
            return analysis
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT-4 JSON response: {gpt_response}")
            return {
                "reasoning": "Failed to parse GPT-4 response, using fallback",
                "tool": "delete_service" if is_deletion else ("create_service_intelligent" if is_creation else "get_api_endpoints"),
                "service_type": detect_service_type_from_query(user_query)
            }
            
    except Exception as e:
        logger.error(f"Error calling GPT-4: {e}")
        return {
            "reasoning": f"GPT-4 error: {str(e)}",
            "tool": "delete_service" if is_deletion else ("create_service_intelligent" if is_creation else "get_api_endpoints"),
            "service_type": detect_service_type_from_query(user_query)
        }

def make_api_request_sync(endpoint: str, params: Dict[str, Any] = None, method: str = "GET", json_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make HTTP request to the API with authentication (synchronous version)"""
    if auth is None:
        return {"error": "Authentication not configured. Check .env file."}
    
    try:
        url = f"{BASE_URL}{endpoint}"
        logger.info("url ---> {}".format(url))
        
        headers = auth.get_headers(use_basic_auth=True)
        
        # Use httpx sync client instead of async
        with httpx.Client(verify=False, timeout=60.0) as client:
            if params and method == "GET":
                query_params = {k: v for k, v in params.items() if not endpoint.find(f"{{{k}}}") >= 0}
                if query_params:
                    url += "?" + urlencode(query_params)
            
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                if json_data:
                    response = client.post(url, headers=headers, json=json_data)
                else:
                    response = client.post(url, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            else:
                return {"success": True, "message": "Request completed successfully"}
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Authentication failed - check username/password in .env")
            return {"error": "Authentication failed - invalid credentials"}
        elif e.response.status_code == 403:
            return {"error": "Access forbidden - insufficient permissions"}
        else:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": str(e)}

def extract_service_info_from_json(service_json: Dict[str, Any]) -> Dict[str, str]:
    """Extract customer_id and instance_id from service JSON"""
    try:
        customer_id = service_json.get("customer_id")
        instance_id = service_json.get("instance_id")
        
        if not customer_id or not instance_id:
            raise ValueError("customer_id and instance_id are required in the service JSON")
        
        return {
            "customer_id": customer_id,
            "instance_id": instance_id
        }
    except Exception as e:
        raise ValueError(f"Error extracting service info: {str(e)}")

# New GPT-4 powered intelligent query analyzer tool
@mcp.tool()
def analyze_query(user_query: str) -> str:
    """
    WHEN TO USE: This is the main entry point for user queries. Use this tool to analyze natural language 
    queries and automatically determine the best tool to call and which service type to parse.
    
    DESCRIPTION: Uses GPT-4 to intelligently analyze user queries, determine the most appropriate tool to call,
    identify the requested service type (L3VPN, L2 Circuit, or EVPN), and execute the recommended action.
    This provides a natural language interface to all the Routing Director capabilities including service creation and deletion.
    
    Args:
        user_query: Natural language query from the user
    
    Returns:
        JSON string with analysis result and tool execution result
    """
    try:
        # Use synchronous GPT-4 analysis
        analysis = analyze_query_with_gpt4(user_query)
        
        if "error" in analysis:
            return json.dumps({
                "error": f"GPT-4 Analysis Error: {analysis['error']}",
                "analysis": analysis
            })
        
        # Extract tool and service type from GPT-4 analysis
        recommended_tool = analysis.get("tool", "get_api_endpoints")
        service_type = analysis.get("service_type", "l2circuit")
        reasoning = analysis.get("reasoning", "No reasoning provided")
        
        # Execute the recommended tool with appropriate parameters
        if recommended_tool == "fetch_all_instances":
            tool_result = execute_tool_by_name(recommended_tool, service_type=service_type)
        elif recommended_tool == "create_service_intelligent":
            tool_result = execute_tool_by_name(recommended_tool, user_query=user_query, service_type=service_type)
        elif recommended_tool == "delete_service":
            # Pass user_query and service_type for intelligent deletion workflow
            tool_result = execute_tool_by_name(recommended_tool, user_query=user_query, service_type=service_type)
        elif recommended_tool == "fetch_all_customers":
            tool_result = execute_tool_by_name(recommended_tool)
        elif recommended_tool == "fetch_all_orders":
            tool_result = execute_tool_by_name(recommended_tool)
        elif recommended_tool == "get_instance_details":
            # For get_instance_details, we need to extract the instance name from the query
            # This could be enhanced with GPT-4 as well, but for now use the existing logic
            tool_result = execute_tool_by_name(recommended_tool, service_type=service_type)
        elif recommended_tool == "get_api_endpoints":
            tool_result = execute_tool_by_name(recommended_tool)
        else:
            # Fallback for any other tools
            tool_result = execute_tool_by_name(recommended_tool, user_query=user_query, service_type=service_type)
        
        response = {
            "gpt4_analysis": {
                "reasoning": reasoning,
                "recommended_tool": recommended_tool,
                "service_type": service_type,
                "service_name": SERVICE_TYPES[service_type]["name"]
            },
            "result": tool_result
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during query analysis: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

def _fetch_all_customers() -> Dict[str, Any]:
    """
    Internal function to fetch all customers from Routing Director API.
    """
    try:
        # Construct API endpoint
        endpoint = ENDPOINTS["get_customers"]
        api_path = endpoint.path.format(org_id=ORG_ID)
        
        # Make API call using synchronous function
        result = make_api_request_sync(api_path)
        
        # Check for errors
        if "error" in result:
            return {"error": f"Error fetching customers: {result['error']}"}
        
        return {
            "success": True,
            "data_type": "customers",
            "count": len(result) if isinstance(result, list) else len(result.get("customers", [])),
            "data": result
        }
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def fetch_all_customers() -> str:
    """
    WHEN TO USE: Use this tool when you need to get all customers with their IDs and details.
    Essential for service operations that require customer ID lookup.
    
    DESCRIPTION: Fetches all customers from Routing Director API with their customer IDs,
    names, and other customer details. This is often needed as a prerequisite for
    service-specific operations like deletion.
    
    Returns:
        JSON string containing customer data with IDs and details
    """
    result = _fetch_all_customers()
    return json.dumps(result, indent=2)

def _find_service_by_customer(instance_name: str, customers_data: List[Dict]) -> Optional[str]:
    """
    Helper function to find customer ID for a service by searching through all customers
    """
    try:
        for customer in customers_data:
            customer_id = customer.get("customer_id") or customer.get("id")
            if customer_id:
                # Try to get instance details for this customer
                try:
                    instance_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_name}"
                    instance_result = make_api_request_sync(instance_endpoint)
                    
                    if "error" not in instance_result:
                        logger.info(f"Found service '{instance_name}' in customer '{customer_id}'")
                        return customer_id
                except Exception:
                    continue
        
        return None
    except Exception as e:
        logger.error(f"Error finding customer for service '{instance_name}': {e}")
        return None

def _extract_service_name_with_gpt4(user_query: str) -> Optional[str]:
    """
    Use GPT-4 to extract service/instance name from natural language deletion request
    """
    if not openai_client:
        logger.warning("OpenAI client not available, falling back to regex extraction")
        return _extract_service_name_regex(user_query)
    
    prompt = f"""
You are a service name extractor for network service deletion requests. 
Extract the exact service/instance name from the user's deletion request.

USER QUERY: "{user_query}"

INSTRUCTIONS:
1. Look for service names, instance names, or identifiers that the user wants to delete
2. Common patterns: "delete service_name", "remove l2circuit1-123", "terminate my-service-name"
3. Return ONLY the service name/ID, no other text
4. If multiple services mentioned, return the first one
5. If no specific service name found, return "NONE"

Examples:
- "Delete l2circuit service l2circuit1-182128" → "l2circuit1-182128"
- "Remove service my-test-service" → "my-test-service"
- "Terminate l3vpn-customer-001" → "l3vpn-customer-001"
- "Delete the service called test-circuit" → "test-circuit"

Respond with ONLY the service name or "NONE":
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise service name extractor. Respond with only the service name or 'NONE'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        extracted_name = response.choices[0].message.content.strip()
        logger.info(f"GPT-4 extracted service name: '{extracted_name}' from query: '{user_query}'")
        
        if extracted_name == "NONE" or not extracted_name:
            logger.warning(f"GPT-4 could not extract service name from: '{user_query}'")
            return None
        return extracted_name
        
    except Exception as e:
        logger.error(f"Error using GPT-4 for service name extraction: {e}")
        return None

def _delete_service_intelligent(instance_name: str, service_type: str = "l2circuit", confirm: bool = False) -> Dict[str, Any]:
    """
    Intelligent service deletion using confirmation workflow:
    1. Find customer ID for the service
    2. Get service details 
    3. Return confirmation request OR execute deletion
    """
    try:
        if not confirm:
            # Step 1: Find the service and show confirmation details
            logger.info(f"🔍 Finding service '{instance_name}' for deletion confirmation")
            
            # Get all customers to find the service
            customers_result = _fetch_all_customers()
            if "error" in customers_result:
                return {
                    "error": f"Cannot find service for confirmation: {customers_result['error']}"
                }
            
            customers_data = customers_result.get("data", [])
            if isinstance(customers_data, dict):
                customers_data = customers_data.get("customers", [])
            
            # Find customer_id for this service
            customer_id = _find_service_by_customer(instance_name, customers_data)
            
            if not customer_id:
                return {
                    "error": f"Service '{instance_name}' not found in any customer",
                    "suggestion": "Please check the service name and try again",
                    "service_type": service_type
                }
            
            # Get service details for confirmation
            instance_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_name}"
            instance_result = make_api_request_sync(instance_endpoint)
            
            if "error" in instance_result:
                return {
                    "error": f"Cannot get service details: {instance_result['error']}",
                    "service_type": service_type
                }
            
            # FIX: Handle case where API returns a list instead of dict
            if isinstance(instance_result, list):
                if len(instance_result) > 0:
                    # Take the first element if it's a list
                    service_data = instance_result[0]
                    logger.info(f"API returned list, using first element for service '{instance_name}'")
                else:
                    return {
                        "error": f"Service '{instance_name}' data is empty",
                        "service_type": service_type
                    }
            elif isinstance(instance_result, dict):
                # It's already a dictionary
                service_data = instance_result
            else:
                return {
                    "error": f"Unexpected response format for service '{instance_name}': {type(instance_result)}",
                    "service_type": service_type
                }
            
            # Extract service information for confirmation display
            service_details = {
                "instance_id": service_data.get("instance_id"),
                "customer_id": service_data.get("customer_id"),
                "instance_status": service_data.get("instance_status", "unknown"),
                "operation": service_data.get("operation", "unknown"),
                "created_time": service_data.get("created_time"),
                "updated_time": service_data.get("updated_time"),
                "design_id": service_data.get("design_id"),
                "design_version": service_data.get("design_version")
            }
            
            # Try to extract customer name and other details
            customer_name = "Unknown"
            source_node = "Unknown"
            dest_node = "Unknown"
            
            try:
                if service_type == "l2circuit" and "l2vpn_ntw" in service_data:
                    vpn_service = service_data["l2vpn_ntw"]["vpn_services"]["vpn_service"][0]
                    customer_name = vpn_service.get("customer_name", "Unknown")
                    
                    vpn_nodes = vpn_service.get("vpn_nodes", {}).get("vpn_node", [])
                    if len(vpn_nodes) >= 2:
                        source_node = vpn_nodes[0].get("vpn_node_id", "Unknown")
                        dest_node = vpn_nodes[1].get("vpn_node_id", "Unknown")
                    elif len(vpn_nodes) == 1:
                        source_node = vpn_nodes[0].get("vpn_node_id", "Unknown")
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Could not extract service details: {e}")
                pass
            
            return {
                "action_required": "delete_confirmation",
                "service_type": service_type,
                "service_name": SERVICE_TYPES.get(service_type, {}).get("name", service_type.upper()),
                "instance_name": instance_name,
                "customer_id": customer_id,
                "customer_name": customer_name,
                "source_node": source_node,
                "dest_node": dest_node,
                "service_details": service_details,
                "warning": f"This will permanently DELETE service '{instance_name}'. This action cannot be undone.",
                "message": "Please review the service details above and confirm deletion.",
                "delete_config": service_data  # Store the full service data for deletion
            }
        
        else:
            # Step 2: Execute the deletion workflow
            return _execute_service_deletion_workflow(instance_name, service_type)
        
    except Exception as e:
        error_msg = f"Delete service analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": error_msg}

@mcp.tool()
def delete_service(user_query: str = "", service_type: str = "l2circuit") -> str:
    """
    WHEN TO USE: Use this tool when user wants to delete/remove/terminate an existing service.
    
    DESCRIPTION: Deletes a service using intelligent analysis and confirmation workflow:
    1. Extract service name from natural language query using GPT-4
    2. Find the service and show confirmation details with full service JSON
    3. Return confirmation request for user approval
    
    Args:
        user_query: Natural language deletion request (e.g., "delete l2circuit1-135006")
        service_type: Type of service ("l3vpn", "l2circuit", "evpn"). Default: "l2circuit"
    
    Returns:
        JSON string containing deletion confirmation details or execution results
    """
    try:
        # Extract service name from natural language query
        if user_query:
            instance_name = _extract_service_name_with_gpt4(user_query)
            if not instance_name:
                return json.dumps({
                    "error": "Could not extract service/instance name from query. Please specify the exact service name.",
                    "user_query": user_query,
                    "suggestion": "Try: 'delete service_name' or 'remove instance_id'"
                })
        else:
            return json.dumps({
                "error": "Please provide the service deletion request",
                "suggestion": "Example: 'delete l2circuit1-135006'"
            })
        
        # Call the intelligent delete service workflow (returns confirmation request)
        result = _delete_service_intelligent(instance_name, service_type, confirm=False)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Service deletion failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
def confirm_delete_service(instance_name: str, service_type: str = "l2circuit") -> str:
    """
    WHEN TO USE: Use this tool to execute the actual service deletion after user confirmation.
    
    DESCRIPTION: Executes the confirmed service deletion workflow:
    1. GET service configuration and modify operation to "delete"
    2. POST to create order
    3. POST to execute deletion
    4. Verify deletion status
    
    Args:
        instance_name: Name of the service/instance to delete
        service_type: Type of service ("l3vpn", "l2circuit", "evpn"). Default: "l2circuit"
    
    Returns:
        JSON string containing the complete deletion workflow result
    """
    try:
        # Execute deletion with confirmation
        result = _delete_service_intelligent(instance_name, service_type, confirm=True)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Service deletion execution failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

def _execute_service_deletion_workflow(instance_name: str, service_type: str) -> Dict[str, Any]:
    """
    Execute the actual service deletion workflow:
    1. Get service configuration
    2. Modify operation to "delete"
    3. POST to create_order
    4. POST to execute_order
    """
    try:
        workflow_result = {
            "success": True,
            "instance_name": instance_name,
            "service_type": service_type,
            "steps": [],
            "final_status": None
        }
        
        logger.info(f"🗑️ Starting deletion workflow for service '{instance_name}'")
        
        # Step 1: Find customer and get service details
        customers_result = _fetch_all_customers()
        if "error" in customers_result:
            workflow_result["success"] = False
            workflow_result["error"] = f"Failed to fetch customers: {customers_result['error']}"
            return workflow_result
        
        customers_data = customers_result.get("data", [])
        if isinstance(customers_data, dict):
            customers_data = customers_data.get("customers", [])
        
        customer_id = _find_service_by_customer(instance_name, customers_data)
        if not customer_id:
            workflow_result["success"] = False
            workflow_result["error"] = f"Service '{instance_name}' not found"
            return workflow_result
        
        # Step 2: Get service configuration
        logger.info(f"📄 Step 1: Getting service configuration for '{instance_name}'")
        
        instance_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_name}"
        instance_result = make_api_request_sync(instance_endpoint)
        
        if "error" in instance_result:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 1,
                "action": "get_service_config",
                "status": "failed",
                "error": instance_result["error"]
            })
            return workflow_result
        
        # FIX: Handle case where API returns a list instead of dict
        if isinstance(instance_result, list):
            if len(instance_result) > 0:
                # Take the first element if it's a list
                service_config = instance_result[0]
                logger.info(f"API returned list, using first element for service '{instance_name}'")
            else:
                workflow_result["success"] = False
                workflow_result["steps"].append({
                    "step": 1,
                    "action": "get_service_config",
                    "status": "failed",
                    "error": "Service configuration data is empty"
                })
                return workflow_result
        elif isinstance(instance_result, dict):
            # It's already a dictionary
            service_config = instance_result
        else:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 1,
                "action": "get_service_config",
                "status": "failed",
                "error": f"Unexpected response format: {type(instance_result)}"
            })
            return workflow_result
        
        workflow_result["steps"].append({
            "step": 1,
            "action": "get_service_config",
            "status": "success",
            "original_operation": service_config.get("operation", "unknown")
        })
        
        # Step 3: Modify operation to "delete" and create order
        logger.info(f"🔄 Step 2: Modifying operation to 'delete' and creating order")
        
        delete_config = service_config.copy()
        delete_config["operation"] = "delete"
        
        create_order_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order"
        create_result = make_api_request_sync(create_order_endpoint, method="POST", json_data=delete_config)
        
        if "error" in create_result:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 2,
                "action": "create_delete_order",
                "status": "failed",
                "error": create_result["error"]
            })
            return workflow_result
        
        workflow_result["steps"].append({
            "step": 2,
            "action": "create_delete_order",
            "status": "success",
            "modified_operation": "delete",
            "response": create_result
        })
        
        # Step 4: Execute the delete order
        logger.info(f"⚡ Step 3: Executing delete order for '{instance_name}'")
        
        time.sleep(2)  # Brief pause between operations
        
        execute_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_name}/exec"
        execute_result = make_api_request_sync(execute_endpoint, method="POST")
        
        if "error" in execute_result:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 3,
                "action": "execute_delete_order",
                "status": "failed",
                "error": execute_result["error"]
            })
            return workflow_result
        
        workflow_result["steps"].append({
            "step": 3,
            "action": "execute_delete_order",
            "status": "success",
            "response": execute_result
        })
        
        # Step 5: Verify deletion
        logger.info(f"🔍 Step 4: Verifying deletion status")
        
        time.sleep(5)  # Wait for deletion to process
        
        try:
            verify_result = make_api_request_sync(instance_endpoint, method="GET")
            
            if "error" in verify_result:
                # Service not found = deletion successful
                workflow_result["final_status"] = "deleted"
                workflow_result["steps"].append({
                    "step": 4,
                    "action": "verify_deletion",
                    "status": "success",
                    "instance_status": "deleted",
                    "note": "Service not found - deletion successful"
                })
                logger.info(f"🎉 Service '{instance_name}' successfully deleted")
            else:
                # Handle the case where verify_result might also be a list
                if isinstance(verify_result, list):
                    if len(verify_result) > 0:
                        service_status_data = verify_result[0]
                    else:
                        # Empty list means service is likely deleted
                        workflow_result["final_status"] = "deleted"
                        workflow_result["steps"].append({
                            "step": 4,
                            "action": "verify_deletion",
                            "status": "success",
                            "instance_status": "deleted",
                            "note": "Service not found - deletion successful"
                        })
                        return workflow_result
                else:
                    service_status_data = verify_result
                
                instance_status = service_status_data.get("instance_status", "unknown")
                workflow_result["final_status"] = instance_status
                workflow_result["steps"].append({
                    "step": 4,
                    "action": "verify_deletion",
                    "status": "warning",
                    "instance_status": instance_status,
                    "note": "Service still exists, deletion may be in progress"
                })
                logger.info(f"⏳ Service '{instance_name}' status: {instance_status}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not verify deletion status: {e}")
            workflow_result["steps"].append({
                "step": 4,
                "action": "verify_deletion",
                "status": "warning",
                "error": f"Status verification failed: {str(e)}"
            })
        
        logger.info(f"✅ Deletion workflow completed for '{instance_name}'")
        return workflow_result
        
    except Exception as e:
        logger.error(f"❌ Deletion workflow failed: {e}")
        return {
            "success": False,
            "error": f"Deletion workflow failed: {str(e)}",
            "steps": []
        }

def _create_service_intelligent(user_query: str, service_type: str = "l2circuit") -> str:
    """
    Internal function for intelligent service creation using EnhancedServiceConfigGenerator
    """
    try:
        if not enhanced_service_generator:
            return json.dumps({
                "error": "Enhanced service generator not available. Please ensure services_generator.py is properly configured.",
                "action_required": "manual_upload",
                "service_type": service_type,
                "message": "Please use the manual file upload method to create services."
            })
        
        logger.info(f"🚀 Processing service creation request with Enhanced Generator")
        logger.info(f"📝 User Query: {user_query}")
        logger.info(f"🔧 Service Type: {service_type}")
        
        # Use the enhanced service generator's new form-based workflow
        # Always show forms for L2 circuits to get user approval of default values
        results = enhanced_service_generator.process_service_request_with_forms(user_query)
        
        # Check if forms are needed
        if results.get("needs_form"):
            logger.info(f"📋 Forms required for {len(results.get('form_templates', []))} service(s)")
            return json.dumps({
                "action_required": "form_input",
                "service_type": results.get("service_type", service_type),
                "message": results.get("message", "Please review and confirm the configuration with default values"),
                "form_templates": results.get("form_templates", []),
                "parsed_request": results.get("parsed_request", {}),
                "original_query": user_query
            })
        
        # Check for validation errors
        if not results["success"] and results.get("validation_errors"):
            return json.dumps({
                "error": "Service request validation failed",
                "validation_errors": results.get("validation_errors", []),
                "parsed_request": results.get("parsed_request", {}),
                "original_query": user_query,
                "suggestions": [
                    "Ensure customer name is valid and active",
                    "Check that source and destination nodes exist in the device inventory",
                    "Verify the service type is supported"
                ]
            })
        
        # If there's an error, return it
        if not results["success"]:
            return json.dumps({
                "error": results.get("error", "Unknown error occurred"),
                "parsed_request": results.get("parsed_request", {}),
                "original_query": user_query
            })
        
        # Normal successful configuration generation
        configs = results.get("json_configs", [])
        junos_configs = results.get("junos_configs", [])
        parsed_request = results.get("parsed_request", {})
        
        if not configs:
            return json.dumps({
                "error": "No configurations generated",
                "parsed_request": parsed_request
            })
        
        logger.info(f"✅ Generated {len(configs)} service configuration(s)")
        logger.info(f"🛠️ Generated {len(junos_configs)} JUNOS configuration(s)")
        
        # Prepare summary for user confirmation
        summary = {
            "action_required": "user_confirmation",
            "service_type": parsed_request.get("service_type", service_type),
            "service_name": SERVICE_TYPES.get(parsed_request.get("service_type", service_type), {}).get("name", "L2 Circuit"),
            "total_services": len(configs),
            "customer_name": parsed_request.get("customer_name", "Unknown"),
            "source_node": parsed_request.get("source_node", "Unknown"),
            "dest_node": parsed_request.get("dest_node", "Unknown"),
            "generated_configs": []
        }
        
        # Add summary and CLI for each config
        for i, config in enumerate(configs):
            config_summary = {
                "index": i + 1,
                "instance_id": config["instance_id"],
                "customer_id": config["customer_id"],
                "vlan_id": None,
                "cli_configs": {}
            }
            
            # Extract VLAN ID for L2 circuits
            if parsed_request.get("service_type") == "l2circuit":
                try:
                    vpn_nodes = config["l2vpn_ntw"]["vpn_services"]["vpn_service"][0]["vpn_nodes"]["vpn_node"]
                    if vpn_nodes:
                        vlan_id = vpn_nodes[0]["vpn_network_accesses"]["vpn_network_access"][0]["connection"]["encapsulation"]["dot1q"]["c_vlan_id"]
                        config_summary["vlan_id"] = vlan_id
                except (KeyError, IndexError):
                    pass
            
            # Add JUNOS CLI configurations from enhanced generator
            junos_config_for_service = next(
                (jc for jc in junos_configs if jc.get("service_name") == config["instance_id"]), 
                None
            )
            
            if junos_config_for_service and junos_config_for_service.get("junos_config"):
                junos_cli = junos_config_for_service["junos_config"]
                
                # Parse the JUNOS config to separate by device if possible
                # For now, we'll assume it's a single config for both devices
                config_summary["cli_configs"] = {
                    parsed_request.get("source_node", "Source Device"): junos_cli,
                    parsed_request.get("dest_node", "Destination Device"): junos_cli
                }
                logger.info(f"✅ Added JUNOS CLI for service {config['instance_id']}")
            else:
                config_summary["cli_configs"] = {"info": "JUNOS CLI generation completed but not available in this response"}
            
            summary["generated_configs"].append(config_summary)
        
        # Store configurations for later use
        summary["configurations"] = configs
        summary["original_results"] = results  # Store full results for debugging
        
        logger.info(f"🎉 Service creation preparation completed successfully")
        logger.info(f"📊 Ready for user confirmation: {len(configs)} services")
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        error_msg = f"Service creation request failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return json.dumps({"error": error_msg})

@mcp.tool()
def create_service_intelligent(user_query: str, service_type: str = "l2circuit") -> str:
    """
    WHEN TO USE: Use this tool when user wants to create services using natural language descriptions.
    
    DESCRIPTION: Uses the Enhanced Service Config Generator to parse natural language service creation requests,
    analyze mandatory requirements, and either generate configurations directly or request additional information
    through interactive forms.
    
    Args:
        user_query: Natural language description of the service to create
        service_type: Type of service to create ("l3vpn", "l2circuit", "evpn"). Default: "l2circuit"
    
    Returns:
        JSON string containing either generated service configurations or form templates for user input
    """
    return _create_service_intelligent(user_query, service_type)

@mcp.tool()
def submit_service_forms(form_data: str) -> str:
    """
    WHEN TO USE: Use this tool to process submitted service configuration forms.
    
    DESCRIPTION: Processes filled service configuration forms and generates the final JSON configurations
    and JUNOS CLI configurations for user confirmation.
    
    Args:
        form_data: JSON string containing the filled form data for all services
    
    Returns:
        JSON string containing generated configurations for user confirmation
    """
    try:
        if not enhanced_service_generator:
            return json.dumps({
                "error": "Enhanced service generator not available",
                "action_required": "manual_upload"
            })
        
        # Parse form data
        try:
            forms = json.loads(form_data)
            if not isinstance(forms, list):
                return json.dumps({
                    "error": "Invalid form data format. Expected list of forms."
                })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid JSON format for form data"
            })
        
        logger.info(f"🚀 Processing {len(forms)} submitted form(s)")
        
        # Generate configurations from form data
        results = enhanced_service_generator.generate_config_from_form_data(forms)
        
        if not results["success"]:
            return json.dumps({
                "error": results.get("error", "Failed to generate configurations"),
                "form_data": forms
            })
        
        # Extract results
        configs = results.get("json_configs", [])
        junos_configs = results.get("junos_configs", [])
        
        if not configs:
            return json.dumps({
                "error": "No configurations generated from form data",
                "form_data": forms
            })
        
        logger.info(f"✅ Generated {len(configs)} configuration(s) from forms")
        
        # Get service details from first form
        first_form = forms[0] if forms else {}
        field_values = {field["field_name"]: field["value"] for field in first_form.get("fields", [])}
        
        # Prepare summary for user confirmation
        summary = {
            "action_required": "user_confirmation",
            "service_type": field_values.get("service_type", "l2circuit"),
            "service_name": SERVICE_TYPES.get(field_values.get("service_type", "l2circuit"), {}).get("name", "L2 Circuit"),
            "total_services": len(configs),
            "customer_name": field_values.get("customer_name", "Unknown"),
            "source_node": field_values.get("source_node", "Unknown"),
            "dest_node": field_values.get("dest_node", "Unknown"),
            "generated_configs": []
        }
        
        # Add summary and CLI for each config
        for i, config in enumerate(configs):
            config_summary = {
                "index": i + 1,
                "instance_id": config["instance_id"],
                "customer_id": config["customer_id"],
                "vlan_id": None,
                "cli_configs": {}
            }
            
            # Extract VLAN ID for L2 circuits
            if field_values.get("service_type") == "l2circuit":
                try:
                    vpn_nodes = config["l2vpn_ntw"]["vpn_services"]["vpn_service"][0]["vpn_nodes"]["vpn_node"]
                    if vpn_nodes:
                        vlan_id = vpn_nodes[0]["vpn_network_accesses"]["vpn_network_access"][0]["connection"]["encapsulation"]["dot1q"]["c_vlan_id"]
                        config_summary["vlan_id"] = vlan_id
                except (KeyError, IndexError):
                    pass
            
            # Add JUNOS CLI configurations
            junos_config_for_service = next(
                (jc for jc in junos_configs if jc.get("service_name") == config["instance_id"]), 
                None
            )
            
            if junos_config_for_service and junos_config_for_service.get("junos_config"):
                junos_cli = junos_config_for_service["junos_config"]
                config_summary["cli_configs"] = {
                    field_values.get("source_node", "Source Device"): junos_cli,
                    field_values.get("dest_node", "Destination Device"): junos_cli
                }
                logger.info(f"✅ Added JUNOS CLI for service {config['instance_id']}")
            else:
                config_summary["cli_configs"] = {"info": "JUNOS CLI generation completed but not available in this response"}
            
            summary["generated_configs"].append(config_summary)
        
        # Store configurations for later use
        summary["configurations"] = configs
        summary["form_data"] = forms
        
        logger.info(f"🎉 Form processing completed successfully")
        logger.info(f"📊 Ready for user confirmation: {len(configs)} services")
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        error_msg = f"Form submission processing failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return json.dumps({"error": error_msg})

@mcp.tool()
def execute_service_creation(configurations: str, confirm: bool = False) -> str:
    """
    WHEN TO USE: Use this tool to execute the actual service creation after user confirmation.
    
    DESCRIPTION: Executes the 2-step service creation workflow for multiple services:
    1. POST to upload service to Routing Director
    2. POST to deploy the service using instance_id
    
    Args:
        configurations: JSON string containing the service configurations to create
        confirm: Boolean confirmation from user
    
    Returns:
        JSON string containing the execution results for all services
    """
    try:
        if not confirm:
            return json.dumps({
                "error": "Service creation not confirmed by user",
                "message": "User must confirm service creation before execution"
            })
        
        # Parse configurations
        try:
            configs_data = json.loads(configurations)
            if "configurations" not in configs_data:
                return json.dumps({"error": "Invalid configuration format"})
            
            configs = configs_data["configurations"]
            logger.info(f"🚀 Starting execution of {len(configs)} service(s)")
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON format for configurations"})
        
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"📦 Creating service {i+1}/{len(configs)}: {config['instance_id']}")
            
            # Execute the 2-step service creation workflow
            result = _execute_service_deployment(config)
            results.append({
                "service_index": i + 1,
                "instance_id": config["instance_id"],
                "result": result
            })
            
            # Add a small delay between service creations
            if i < len(configs) - 1:
                logger.info("⏳ Waiting before next service creation...")
                time.sleep(3)
        
        # Summary
        successful_services = [r for r in results if r["result"].get("success", False)]
        failed_services = [r for r in results if not r["result"].get("success", False)]
        
        summary = {
            "success": len(failed_services) == 0,
            "total_services": len(configs),
            "successful_count": len(successful_services),
            "failed_count": len(failed_services),
            "service_results": results,
            "summary": {
                "successful_services": [s["instance_id"] for s in successful_services],
                "failed_services": [s["instance_id"] for s in failed_services]
            }
        }
        
        logger.info(f"🎯 Execution completed: {len(successful_services)}/{len(configs)} successful")
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        error_msg = f"Service execution failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

def _execute_service_deployment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the 2-step deployment workflow for a single service:
    1. POST to upload service to Routing Director
    2. POST to deploy the service
    """
    try:
        # Extract service information
        customer_id = config.get("customer_id")
        instance_id = config.get("instance_id")
        
        if not customer_id or not instance_id:
            return {
                "success": False,
                "error": "Missing customer_id or instance_id in configuration",
                "steps": []
            }
        
        workflow_result = {
            "success": True,
            "customer_id": customer_id,
            "instance_id": instance_id,
            "steps": [],
            "final_status": None
        }
        
        # Step 1: Upload Service to Routing Director (POST to create order)
        logger.info(f"📤 Step 1: Uploading service '{instance_id}' to Routing Director")
        
        create_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order"
        create_result = make_api_request_sync(create_endpoint, method="POST", json_data=config)
        
        if "error" in create_result:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 1,
                "action": "upload_service",
                "status": "failed",
                "error": create_result["error"]
            })
            return workflow_result
        
        workflow_result["steps"].append({
            "step": 1,
            "action": "upload_service",
            "status": "success",
            "response": create_result
        })
        logger.info(f"✅ Step 1 completed: Service uploaded successfully")
        
        # Step 2: Deploy the Service (POST to execute order)
        logger.info(f"🚀 Step 2: Deploying service '{instance_id}'")
        
        # Wait a moment before deployment
        time.sleep(2)
        
        deploy_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_id}/exec"
        deploy_result = make_api_request_sync(deploy_endpoint, method="POST")
        
        if "error" in deploy_result:
            workflow_result["success"] = False
            workflow_result["steps"].append({
                "step": 2,
                "action": "deploy_service",
                "status": "failed",
                "error": deploy_result["error"]
            })
            return workflow_result
        
        workflow_result["steps"].append({
            "step": 2,
            "action": "deploy_service",
            "status": "success",
            "response": deploy_result
        })
        
        logger.info(f"✅ Step 2 completed: Service deployment initiated")
        
        # Step 3: Verify deployment status after some delay
        logger.info(f"🔍 Step 3: Verifying deployment status for '{instance_id}'")
        time.sleep(5)  # Wait for deployment to process
        
        try:
            status_endpoint = f"/service-orchestration/api/v1/orgs/{ORG_ID}/order/customers/{customer_id}/instances/{instance_id}"
            status_result = make_api_request_sync(status_endpoint, method="GET")
            
            if "error" not in status_result:
                instance_status = status_result.get("instance_status", "unknown")
                workflow_result["final_status"] = instance_status
                workflow_result["steps"].append({
                    "step": 3,
                    "action": "verify_deployment",
                    "status": "success",
                    "instance_status": instance_status,
                    "response": status_result
                })
                
                if instance_status == "active":
                    logger.info(f"🎉 Service '{instance_id}' is now ACTIVE")
                else:
                    logger.info(f"⏳ Service '{instance_id}' status: {instance_status}")
            else:
                workflow_result["steps"].append({
                    "step": 3,
                    "action": "verify_deployment",
                    "status": "warning",
                    "error": "Could not verify status, but deployment was initiated"
                })
        except Exception as e:
            logger.warning(f"⚠️ Could not verify status for '{instance_id}': {e}")
            workflow_result["steps"].append({
                "step": 3,
                "action": "verify_deployment",
                "status": "warning",
                "error": f"Status verification failed: {str(e)}"
            })
        
        return workflow_result
        
    except Exception as e:
        logger.error(f"❌ Service deployment failed for '{instance_id}': {e}")
        return {
            "success": False,
            "error": f"Service deployment failed: {str(e)}",
            "steps": []
        }

def execute_tool_by_name(tool_name: str, **kwargs) -> Any:
    """ Execute a tool by name with optional parameters """
    try:
        if tool_name == "fetch_all_instances":
            service_type = kwargs.get("service_type", "l3vpn")
            return _fetch_all_instances(service_type=service_type)
        elif tool_name == "fetch_all_orders":
            return _fetch_all_orders()
        elif tool_name == "fetch_all_customers":
            return _fetch_all_customers()
        elif tool_name == "get_api_endpoints":
            return _get_api_endpoints()
        elif tool_name == "create_service_intelligent":
            user_query = kwargs.get("user_query", "")
            service_type = kwargs.get("service_type", "l2circuit")
            # Call the function directly and parse the JSON response
            result_str = _create_service_intelligent(user_query, service_type)
            try:
                return json.loads(result_str)
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON response from create_service_intelligent: {result_str}"}
        elif tool_name == "delete_service":
            user_query = kwargs.get("user_query", "")
            service_type = kwargs.get("service_type", "l2circuit")
            
            # Extract service name from natural language query (same logic as delete_service tool)
            if user_query:
                instance_name = _extract_service_name_with_gpt4(user_query)
                if not instance_name:
                    return {
                        "error": "Could not extract service/instance name from query. Please specify the exact service name.",
                        "user_query": user_query,
                        "suggestion": "Try: 'delete service_name' or 'remove instance_id'"
                    }
            else:
                return {
                    "error": "Please provide the service deletion request",
                    "suggestion": "Example: 'delete l2circuit1-135006'"
                }
            
            # Call the internal delete service workflow (returns confirmation request)
            result = _delete_service_intelligent(instance_name, service_type, confirm=False)
            return result
        elif tool_name == "get_instance_details":
            # This would need enhancement to extract instance name from query
            # For now, return an error suggesting specific instance name
            return {"error": "Please specify the exact instance name for detailed information"}
        else:
            return {"error": f"Unknown tool: {tool_name}"}
            
    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Tool execution error: {str(e)}"}

def _create_service_workflow(service_json: str, service_type: str = "l2circuit") -> Dict[str, Any]:
    """
    Internal function to create a service using the 2-step workflow:
    1. POST to upload service to Routing Director
    2. POST to deploy service using instance_id
    """
    try:
        # Parse the JSON string
        try:
            service_data = json.loads(service_json)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format: {str(e)}"}
        
        # Use the new deployment function
        return _execute_service_deployment(service_data)
        
    except Exception as e:
        return {"error": f"Service creation workflow failed: {str(e)}"}

@mcp.tool()
def create_service(service_json: str, service_type: str = "l2circuit") -> str:
    """
    WHEN TO USE: Use this tool to create a new service (L2 Circuit, L3VPN, or EVPN) by providing the complete
    service configuration as JSON and executing the full provisioning workflow.
    
    DESCRIPTION: Creates a new service using a 2-step workflow:
    1. POST to upload service to Routing Director
    2. POST to deploy the service using instance_id
    
    Args:
        service_json: Complete JSON configuration for the service (as string)
        service_type: Type of service to create ("l3vpn", "l2circuit", "evpn"). Default: "l2circuit"
    
    Returns:
        JSON string containing the complete workflow result including all steps and final status
    """
    result = _create_service_workflow(service_json, service_type)
    return json.dumps(result, indent=2)

def _fetch_all_instances(service_type: str = "l3vpn") -> Dict[str, Any]:
    """
    Internal function to fetch all Service Instances from Routing Director API and parse the specified service type.
    Each instance represents a deployed service with its current status, design information, and operational details.
    """
    try:
        # Validate service type
        if service_type not in SERVICE_TYPES:
            return {"error": f"Invalid service type: {service_type}. Available types: {list(SERVICE_TYPES.keys())}"}
        
        # Construct API endpoint
        endpoint = ENDPOINTS["get_instances"]
        api_path = endpoint.path.format(org_id=ORG_ID)
        
        # Make API call using synchronous function
        result = make_api_request_sync(api_path)
        
        # Check for errors
        if "error" in result:
            return {"error": f"Error fetching instances: {result['error']}"}
        
        # Get the appropriate parser for the service type
        parser_func = SERVICE_TYPES[service_type]["parser"]
        service_name = SERVICE_TYPES[service_type]["name"]
        
        logger.info(f"Parsing {service_name} services using {parser_func.__name__}")
        
        # Parse the data using the appropriate parser
        try:
            services_df, ref_data = parser_func(result)
        except Exception as parse_error:
            logger.error(f"Error parsing {service_name} data: {parse_error}")
            return {"error": f"Error parsing {service_name} data: {str(parse_error)}"}
        
        # Convert DataFrame to JSON-serializable format
        df_json = dataframe_to_json_serializable(services_df)
        
        return {
            "success": True,
            "data_type": "services",
            "service_type": service_type,
            "service_name": service_name,
            "total_services": len(services_df) if services_df is not None else 0,
            "data": df_json,
            "ref_data": ref_data
        }
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

@mcp.tool()
def fetch_all_instances(service_type: str = "l3vpn") -> str:
    """
    WHEN TO USE: Use this tool when you need to get a complete list of service instances for a specific service type. 
    This is the primary tool for instance discovery and should be called first before performing any instance-specific operations.
    
    DESCRIPTION: Fetches all Service Instances from Routing Director API and parses the specified service type.
    Each instance represents a deployed service with its current status, design information, and operational details.
    Supports L3VPN, L2 Circuit, and EVPN service types.
    
    Args:
        service_type: Type of service to parse ("l3vpn", "l2circuit", "evpn"). Default: "l3vpn"
    
    Returns:
        JSON string containing parsed service data for the specified type
    """
    result = _fetch_all_instances(service_type=service_type)
    logger.info("*"*60)
    logger.info(f"Result type: {type(result)}")
    if 'data' in result and result['data']:
        logger.info(f"Data type: {type(result['data'])}")
        logger.info(f"Data keys: {result['data'].keys() if isinstance(result['data'], dict) else 'Not a dict'}")
    logger.info("*"*60)
    return json.dumps(result, indent=2)

@mcp.tool()
def fetch_l3vpn_services() -> str:
    """
    WHEN TO USE: Use this tool specifically when you need L3VPN service instances.
    
    DESCRIPTION: Fetches and parses L3VPN service instances specifically.
    
    Returns:
        JSON string containing L3VPN service data
    """
    result = _fetch_all_instances(service_type="l3vpn")
    return json.dumps(result, indent=2)

@mcp.tool()
def fetch_l2circuit_services() -> str:
    """
    WHEN TO USE: Use this tool specifically when you need L2 Circuit service instances.
    
    DESCRIPTION: Fetches and parses L2 Circuit service instances specifically.
    
    Returns:
        JSON string containing L2 Circuit service data
    """
    result = _fetch_all_instances(service_type="l2circuit")
    return json.dumps(result, indent=2)

@mcp.tool()
def fetch_evpn_services() -> str:
    """
    WHEN TO USE: Use this tool specifically when you need EVPN service instances.
    
    DESCRIPTION: Fetches and parses L2VPN EVPN service instances specifically.
    
    Returns:
        JSON string containing EVPN service data
    """
    result = _fetch_all_instances(service_type="evpn")
    return json.dumps(result, indent=2)

def _fetch_all_orders() -> Dict[str, Any]:
    """
    Internal function to fetch all Service Orders from Routing Director API including complete history. 
    Orders represent operations (create, update, delete) performed on service instances. 
    Multiple orders per instance show the evolution of services over time.
    """
    try:
        # Construct API endpoint
        endpoint = ENDPOINTS["get_orders"]
        api_path = endpoint.path.format(org_id=ORG_ID)
        
        # Make API call using synchronous function
        result = make_api_request_sync(api_path)
        
        # Check for errors
        if "error" in result:
            return {"error": f"Error fetching orders: {result['error']}"}
        
        return {
            "success": True,
            "data_type": "orders",
            "count": len(result) if isinstance(result, list) else len(result.get("orders", [])),
            "data": result
        }
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

@mcp.tool()
def fetch_all_orders() -> str:
    """
    WHEN TO USE: Use this tool when you need to examine the order history, track service changes, or investigate 
    multiple operations performed on service instances. Essential for audit trails and operational analysis.
    
    DESCRIPTION: Fetches all Service Orders from Routing Director API including complete history. 
    Orders represent operations (create, update, delete) performed on service instances. 
    Multiple orders per instance show the evolution of services over time.
    
    Args:
        per_page: Number of items per page (default: 100, max recommended: 500)
        offset: Starting offset for pagination (default: 0)
        filter_query: Optional JQ filter to limit results (e.g., '.[] | select(.operation == "create")')
    
    Returns:
        JSON string containing order data
    """
    result = _fetch_all_orders()
    return json.dumps(result, indent=2)

def _get_instance_details(instance_name: str, service_type: str = "l3vpn") -> Dict[str, Any]:
    """
    Internal function to retrieve and display all available fields for a specific instance.
    First fetches all instances and then finds the matching one.
    """
    try:
        # First fetch all instances for the specified service type
        instances_result = _fetch_all_instances(service_type=service_type)
        
        if "error" in instances_result:
            return {"error": f"Failed to fetch instances: {instances_result['error']}"}
        
        instances_data = instances_result.get("data", {}).get("data", [])
        
        # Find the specific instance
        instance = next(
            (inst for inst in instances_data if inst.get("instance_name") == instance_name),
            None
        )
        
        if not instance:
            return {"error": f"Instance not found: '{instance_name}' in {service_type} services"}
        
        return {
            "success": True,
            "data_type": "instance_details",
            "service_type": service_type,
            "instance_name": instance_name,
            "data": instance
        }
        
    except Exception as e:
        return {"error": f"Error getting instance details: {str(e)}"}

@mcp.tool()
def get_instance_details(instance_name: str, service_type: str = "l3vpn") -> str:
    """
    WHEN TO USE: Use this tool when you need complete detailed information about a specific service instance.
    Perfect for deep-dive analysis, debugging, or getting full configuration details.
    
    DESCRIPTION: Retrieves and displays all available fields for a specific instance.
    Shows comprehensive details including timestamps, configuration, and status information.
    
    Args:
        instance_name: Exact name of the instance to get details for
        service_type: Type of service to search in ("l3vpn", "l2circuit", "evpn"). Default: "l3vpn"
    
    Returns:
        JSON string containing detailed information about the specified instance
    """
    result = _get_instance_details(instance_name, service_type)
    return json.dumps(result, indent=2)

def _get_api_endpoints() -> Dict[str, Any]:
    """
    Internal function that lists all available API endpoints with their descriptions, and usage guidelines.
    """
    endpoints_info = {
        "success": True,
        "data_type": "api_endpoints",
        "service_types": SERVICE_TYPES,
        "endpoints": {}
    }
    
    for name, endpoint in ENDPOINTS.items():
        endpoints_info["endpoints"][name] = {
            "name": name.replace('_', ' ').title(),
            "path": endpoint.path,
            "method": endpoint.method,
            "description": endpoint.description,
        }
    
    return endpoints_info

@mcp.tool()
def get_api_endpoints() -> str:
    """
    WHEN TO USE: Use this tool to understand what API endpoints are available and their usage patterns.
    Helpful for understanding the system capabilities and proper parameter usage.
    
    DESCRIPTION: Lists all available API endpoints with their descriptions, required parameters, service types, and usage guidelines.
    
    Returns:
        JSON string containing detailed information about available API endpoints and service types
    """
    result = _get_api_endpoints()
    return json.dumps(result, indent=2)

@mcp.tool()
def debug_service_generator() -> str:
    """
    WHEN TO USE: Use this tool to debug service generator issues and check system status.
    
    DESCRIPTION: Provides detailed information about enhanced service generator status, available data,
    CLI generator status, and can test basic functionality.
    
    Returns:
        JSON string with debug information and test results
    """
    debug_info = {
        "enhanced_service_generator_available": SERVICE_GENERATOR_AVAILABLE,
        "enhanced_service_generator_initialized": enhanced_service_generator is not None,
        "cli_generator_available": CLI_GENERATOR_AVAILABLE,
        "cli_generator_initialized": cli_generator is not None,
        "openai_available": openai_client is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check CLI generator status
    if cli_generator is not None:
        debug_info["cli_generator_ready"] = cli_generator.available
    
    if enhanced_service_generator is not None:
        try:
            # Test data availability
            customers_count = len(enhanced_service_generator.customers_data)
            devices_count = len(enhanced_service_generator.devices_data.get('devices', []))
            
            debug_info.update({
                "customers_loaded": customers_count,
                "devices_loaded": devices_count,
                "customers_sample": [c.get('name') for c in enhanced_service_generator.customers_data[:3]],
                "devices_sample": [d.get('hostname') for d in enhanced_service_generator.devices_data.get('devices', [])[:3]]
            })
            
            # Test the form-based workflow
            test_query = "create l2 circuit for SINET from PNH-ACX7024-A1 to TH-ACX7100-A6 with service name test-debug"
            try:
                test_results = enhanced_service_generator.process_service_request_with_forms(test_query)
                
                debug_info["test_form_workflow"] = {
                    "success": test_results.get("success", False),
                    "needs_form": test_results.get("needs_form", False),
                    "form_templates_count": len(test_results.get("form_templates", [])),
                    "validation_errors": test_results.get("validation_errors", []),
                    "parsed_request": test_results.get("parsed_request", {})
                }
                
            except Exception as e:
                debug_info["test_form_workflow"] = {
                    "success": False,
                    "error": str(e)
                }
        
        except Exception as e:
            debug_info["enhanced_generator_error"] = str(e)
    
    else:
        debug_info["enhanced_generator_error"] = "Enhanced service generator not initialized"
    
    return json.dumps(debug_info, indent=2)

if __name__ == "__main__":
    mcp.run()
    