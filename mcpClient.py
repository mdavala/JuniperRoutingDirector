import streamlit as st
import asyncio
import json
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Routing Director MCP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface and enhanced table styling
st.markdown("""
<style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
    }
    
    .gpt4-message {
        background-color: #e8f5e8;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    
    .form-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #ffc107;
    }
    
    .create-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #ffc107;
    }
    
    .delete-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #dc3545;
    }
    
    .confirmation-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #17a2b8;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
    }
    
    .connection-status {
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .gpt4-status {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .chat-input {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem 0;
        border-top: 1px solid #ddd;
    }
    
    .ai-hint {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    
    .form-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .form-header {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-weight: bold;
        text-align: center;
    }
    
    .form-field {
        margin-bottom: 15px;
        padding: 10px;
        background-color: white;
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }
    
    .form-field-label {
        font-weight: bold;
        color: #495057;
        margin-bottom: 5px;
        display: block;
    }
    
    .form-field-description {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 3px;
    }
    
    .form-actions {
        text-align: center;
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #dee2e6;
    }
    
    .delete-confirmation-panel {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
        border-left: 4px solid #721c24;
    }
    
    .delete-service-details {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #dc3545;
    }
    
    .delete-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #ffc107;
    }
    
    .dataframe-container {
        margin: 10px 0;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .table-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px 8px 0 0;
        margin-bottom: 10px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .table-actions {
        display: flex;
        gap: 10px;
        margin-bottom: 10px;
    }
    
    .metric-card {
        background: white;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 2px;
    }
    
    .data-quality-indicator {
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
    
    .quality-excellent {
        background-color: #d4edda;
        color: #155724;
    }
    
    .quality-good {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    .quality-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .quality-poor {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .workflow-step {
        background-color: #2a2e35;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .workflow-step-success {
        border-left: 4px solid #28a745;
        background-color: #d4edda;
    }
    
    .workflow-step-failed {
        border-left: 4px solid #dc3545;
        background-color: #f8d7da;
    }
    
    .workflow-step-pending {
        border-left: 4px solid #ffc107;
        background-color: #fff3cd;
    }
    
    .service-config-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .service-config-header {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .config-detail {
        background-color: #f8f9fa;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 5px 0;
        border-left: 3px solid #28a745;
    }
    
    .confirmation-panel {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
    }
    
    .confirmation-buttons {
        display: flex;
        gap: 15px;
        justify-content: center;
        margin-top: 15px;
    }
    
    .btn-confirm {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .btn-cancel {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    
    .execution-result {
        margin: 15px 0;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .result-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .result-partial {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    
    .result-failed {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    .cli-config-container {
        background-color: #1e1e1e;
        border: 1px solid #444;
        border-radius: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .cli-config-header {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 8px 15px;
        font-size: 0.9rem;
        font-weight: bold;
        border-bottom: 1px solid #444;
    }
    
    .cli-config-content {
        padding: 15px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.4;
        color: #f8f8f2;
        background-color: #1e1e1e;
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .device-tab {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        margin: 5px 0;
    }
    
    .device-tab-header {
        background-color: #e9ecef;
        padding: 8px 12px;
        font-weight: bold;
        border-bottom: 1px solid #dee2e6;
    }
    
    .cli-preview {
        background-color: #263238;
        color: #eeffff;
        padding: 12px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.3;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
    }
    
    .junos-command {
        color: #66bb6a;
    }
    
    .junos-parameter {
        color: #42a5f5;
    }
    
    .junos-value {
        color: #ffa726;
    }
    
    .collapsible-section {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 10px 0;
        background-color: white;
    }
    
    .collapsible-header {
        background-color: #f8f9fa;
        padding: 12px 15px;
        border-bottom: 1px solid #dee2e6;
        cursor: pointer;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .collapsible-header:hover {
        background-color: #e9ecef;
    }
    
    .collapsible-content {
        padding: 15px;
        border-top: 1px solid #dee2e6;
    }
    
    .json-preview {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    .expand-icon {
        transition: transform 0.3s ease;
    }
    
    .expand-icon.expanded {
        transform: rotate(90deg);
    }
</style>
""", unsafe_allow_html=True)

class MCPClient:
    def __init__(self):
        self.server_script_path = "mcpServer.py"
        self.connected = False
        self.available_tools = []
        self.gpt4_enabled = False
    
    async def connect(self, server_script_path="mcpServer.py"):
        """Test connection to the MCP server"""
        try:
            self.server_script_path = server_script_path
            
            # Test connection by listing available tools
            tools = await self.get_available_tools()
            if tools:
                self.connected = True
                self.available_tools = tools
                
                # Check if GPT-4 analyze_query tool is available
                self.gpt4_enabled = any(tool.get('name') == 'analyze_query' for tool in tools)
                
                return True
            else:
                self.connected = False
                return False
        except Exception as e:
            st.error(f"Failed to connect to MCP server: {str(e)}")
            self.connected = False
            return False
    
    async def get_available_tools(self):
        """Get list of available tools from the server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path],
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return [tool.dict() for tool in tools_result.tools]
        except Exception as e:
            print(f"Error getting tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path],
            )
            
            # Create connection and session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)
                    
                    if result.isError:
                        error_msg = result.content[0].text if result.content else 'Unknown error'
                        return f"❌ Tool error: {error_msg}"
                    
                    return result.content[0].text if result.content else "No response"
            
        except Exception as e:
            return f"❌ Error calling tool: {str(e)}"
    
    async def send_message(self, message):
        """Send a message to the MCP server using GPT-4 intelligent routing"""
        if not self.connected:
            raise Exception("Not connected to MCP server")
        
        try:
            # Use GPT-4 powered analysis if available
            if self.gpt4_enabled:
                response = await self.handle_gpt4_query(message)
            else:
                # Fallback to legacy routing
                response = await self.handle_legacy_query(message)
            
            return response
        except Exception as e:
            raise Exception(f"Message processing failed: {str(e)}")
    
    async def handle_gpt4_query(self, message):
        """Handle queries using GPT-4 powered analyze_query tool"""
        try:
            # Use the analyze_query tool which leverages GPT-4 for intelligent routing
            response = await self.call_tool("analyze_query", {"user_query": message})
            return response
        except Exception as e:
            return f"❌ Error with GPT-4 analysis: {str(e)}"
    
    async def execute_service_creation(self, configurations, confirm=True):
        """Execute service creation with user confirmation"""
        try:
            response = await self.call_tool("execute_service_creation", {
                "configurations": configurations,
                "confirm": confirm
            })
            return response
        except Exception as e:
            return f"❌ Error executing service creation: {str(e)}"
    
    async def confirm_delete_service(self, instance_name, service_type="l2circuit"):
        """Execute confirmed service deletion"""
        try:
            response = await self.call_tool("confirm_delete_service", {
                "instance_name": instance_name,
                "service_type": service_type
            })
            return response
        except Exception as e:
            return f"❌ Error executing service deletion: {str(e)}"
    
    async def submit_service_forms(self, form_data):
        """Submit filled service forms to the server"""
        try:
            response = await self.call_tool("submit_service_forms", {
                "form_data": json.dumps(form_data)
            })
            return response
        except Exception as e:
            return f"❌ Error submitting forms: {str(e)}"
    
    async def handle_legacy_query(self, message):
        """Fallback method for queries when GPT-4 is not available"""
        message_lower = message.lower()
        
        try:
            # Check what tools are available and route accordingly
            tool_names = [tool.get('name', '') for tool in self.available_tools]
            
            # Basic routing logic for fallback
            if any(word in message_lower for word in ['create', 'provision', 'deploy', 'add', 'new']):
                return "Please use the GPT-4 powered intelligent service creation. Your query will be analyzed automatically."
            elif any(word in message_lower for word in ['instances', 'services', 'list', 'show all']):
                return await self.call_tool("fetch_all_instances", {})
            elif any(word in message_lower for word in ['orders', 'history', 'operations']):
                return await self.call_tool("fetch_all_orders", {})
            elif any(word in message_lower for word in ['search', 'find', 'look for']):
                # Extract search term (simple extraction)
                words = message.split()
                search_term = ' '.join(words[1:]) if len(words) > 1 else ""
                return await self.call_tool("search_instances", {"search_term": search_term})
            elif any(word in message_lower for word in ['help', 'endpoints', 'api']):
                return await self.call_tool("get_api_endpoints", {})
            else:
                # Default to fetch instances
                return await self.call_tool("fetch_all_instances", {})
                
        except Exception as e:
            return f"❌ Error processing your request: {str(e)}"

def analyze_dataframe_quality(df):
    """Analyze dataframe quality and return insights"""
    if df is None or df.empty:
        return {"quality": "poor", "insights": ["No data available"]}
    
    insights = []
    quality_score = 100
    
    try:
        # Check for missing values
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 20:
            insights.append(f"High missing data: {missing_percentage:.1f}%")
            quality_score -= 30
        elif missing_percentage > 10:
            insights.append(f"Some missing data: {missing_percentage:.1f}%")
            quality_score -= 15
        elif missing_percentage > 0:
            insights.append(f"Minimal missing data: {missing_percentage:.1f}%")
            quality_score -= 5
        
        # Check for duplicate rows - handle unhashable types
        try:
            # Create a copy of the dataframe and convert any list/dict columns to strings
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    # Convert any unhashable types to strings
                    df_copy[col] = df_copy[col].apply(
                        lambda x: str(x) if isinstance(x, (list, dict, set)) else x
                    )
            
            duplicate_count = df_copy.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100
            
            if duplicate_percentage > 10:
                insights.append(f"High duplicates: {duplicate_percentage:.1f}%")
                quality_score -= 20
            elif duplicate_percentage > 0:
                insights.append(f"Some duplicates: {duplicate_percentage:.1f}%")
                quality_score -= 10
        except Exception as e:
            print(f"DEBUG: Could not analyze duplicates: {e}")
            insights.append("Duplicate analysis skipped (complex data types)")
            quality_score -= 5
        
        # Data type analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            insights.append(f"Numeric columns: {len(numeric_cols)}")
        if len(text_cols) > 0:
            insights.append(f"Text columns: {len(text_cols)}")
        
        # Check for list/unhashable columns
        list_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(5)
                if any(isinstance(val, (list, dict, set)) for val in sample_values):
                    list_cols.append(col)
        
        if list_cols:
            insights.append(f"Complex data columns: {len(list_cols)}")
            quality_score -= 5
        
    except Exception as e:
        print(f"DEBUG: Error in analyze_dataframe_quality: {e}")
        insights.append("Analysis partially failed")
        quality_score = 50
    
    # Determine quality level
    if quality_score >= 90:
        quality = "excellent"
    elif quality_score >= 75:
        quality = "good"
    elif quality_score >= 60:
        quality = "warning"
    else:
        quality = "poor"
    
    return {"quality": quality, "insights": insights, "score": quality_score}

def create_downloadable_csv(df):
    """Create downloadable CSV buffer"""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def extract_l3vpn_services_from_ref_data(ref_data):
    """Extract L3VPN service information from the complex ref_data structure and convert to DataFrame"""
    services_list = []
    print(f"DEBUG: extract_l3vpn_services_from_ref_data called with keys: {list(ref_data.keys()) if isinstance(ref_data, dict) else 'Not a dict'}")
    
    for vpn_id, vpn_data in ref_data.items():
        print(f"DEBUG: Processing L3VPN ID: {vpn_id}")
        
        try:
            # Navigate through the nested structure for L3VPN
            if 'l3vpn_ntw' in vpn_data and 'vpn_services' in vpn_data['l3vpn_ntw']:
                vpn_services = vpn_data['l3vpn_ntw']['vpn_services']
                
                if 'vpn_service' in vpn_services:
                    for idx, service in enumerate(vpn_services['vpn_service']):
                        # Extract basic L3VPN service information
                        service_info = {
                            'Service_Type': 'L3VPN',
                            'VPN_ID': str(vpn_id),
                            'Service_VPN_ID': str(service.get('vpn_id', '')),
                            'VPN_Name': str(service.get('vpn_name', '')),
                            'Customer_Name': str(service.get('customer_name', '')),
                            'Description': str(service.get('vpn_description', '')),
                            'Node_Count': 0,
                            'Network_Access_Count': 0,
                            'Node_IDs': '',
                            'Site_IDs': '',
                            'RD_Values': ''
                        }
                        
                        # Extract node information
                        node_ids = []
                        site_ids = []
                        rd_values = []
                        
                        if 'vpn_nodes' in service and 'vpn_node' in service['vpn_nodes']:
                            nodes = service['vpn_nodes']['vpn_node']
                            service_info['Node_Count'] = len(nodes)
                            
                            for node in nodes:
                                node_ids.append(str(node.get('ne_id', '')))
                                site_ids.append(str(node.get('site_id', '')))
                                rd_values.append(str(node.get('rd', '')))
                                
                                # Count network accesses
                                if 'vpn_network_accesses' in node and 'vpn_network_access' in node['vpn_network_accesses']:
                                    access_count = len(node['vpn_network_accesses']['vpn_network_access'])
                                    service_info['Network_Access_Count'] += access_count
                        
                        # Convert lists to comma-separated strings for better display
                        service_info['Node_IDs'] = ', '.join(node_ids[:3]) + ('...' if len(node_ids) > 3 else '')
                        service_info['Site_IDs'] = ', '.join(site_ids[:3]) + ('...' if len(site_ids) > 3 else '')
                        service_info['RD_Values'] = ', '.join(rd_values[:3]) + ('...' if len(rd_values) > 3 else '')
                        
                        # Ensure all values are strings (not lists or other unhashable types)
                        for key, value in service_info.items():
                            if isinstance(value, (list, dict, set)):
                                service_info[key] = str(value)
                            elif value is None:
                                service_info[key] = ''
                            else:
                                service_info[key] = str(value)
                        
                        services_list.append(service_info)
                        
        except Exception as e:
            print(f"DEBUG: Error extracting L3VPN service data for VPN {vpn_id}: {e}")
            continue
    
    print(f"DEBUG: Final L3VPN services_list length: {len(services_list)}")
    return services_list

def extract_l2circuit_services_from_ref_data(ref_data):
    """Extract L2 Circuit service information from the ref_data structure and convert to DataFrame"""
    services_list = []
    print(f"DEBUG: extract_l2circuit_services_from_ref_data called with keys: {list(ref_data.keys()) if isinstance(ref_data, dict) else 'Not a dict'}")
    
    try:
        # Handle different possible structures for L2 circuits
        for circuit_id, circuit_data in ref_data.items():
            print(f"DEBUG: Processing L2 Circuit ID: {circuit_id}")
            
            try:
                # Common L2 circuit structure patterns
                service_info = {
                    'Service_Type': 'L2 Circuit',
                    'Circuit_ID': str(circuit_id),
                    'Circuit_Name': '',
                    'Customer_Name': '',
                    'Description': '',
                    'Endpoint_Count': 0,
                    'Status': '',
                    'Bandwidth': '',
                    'VLAN_ID': '',
                    'Endpoints': ''
                }
                
                # Try different possible data structures
                if isinstance(circuit_data, dict):
                    # Pattern 1: Direct circuit information
                    service_info['Circuit_Name'] = str(circuit_data.get('circuit_name', circuit_data.get('name', '')))
                    service_info['Customer_Name'] = str(circuit_data.get('customer_name', circuit_data.get('customer', '')))
                    service_info['Description'] = str(circuit_data.get('description', circuit_data.get('desc', '')))
                    service_info['Status'] = str(circuit_data.get('status', circuit_data.get('admin_status', '')))
                    service_info['Bandwidth'] = str(circuit_data.get('bandwidth', circuit_data.get('speed', '')))
                    service_info['VLAN_ID'] = str(circuit_data.get('vlan_id', circuit_data.get('vlan', '')))
                    
                    # Extract endpoints
                    endpoints = []
                    if 'endpoints' in circuit_data:
                        endpoints = circuit_data['endpoints']
                    elif 'l2_circuit_endpoints' in circuit_data:
                        endpoints = circuit_data['l2_circuit_endpoints']
                    elif 'ports' in circuit_data:
                        endpoints = circuit_data['ports']
                    
                    if isinstance(endpoints, list):
                        service_info['Endpoint_Count'] = len(endpoints)
                        endpoint_names = []
                        for endpoint in endpoints[:3]:  # Show first 3 endpoints
                            if isinstance(endpoint, dict):
                                endpoint_name = endpoint.get('interface', endpoint.get('port', endpoint.get('name', str(endpoint))))
                                endpoint_names.append(str(endpoint_name))
                            else:
                                endpoint_names.append(str(endpoint))
                        
                        service_info['Endpoints'] = ', '.join(endpoint_names)
                        if len(endpoints) > 3:
                            service_info['Endpoints'] += '...'
                    else:
                        service_info['Endpoints'] = ''
                    
                    # Pattern 2: Nested service structure
                    if 'l2_circuit' in circuit_data:
                        l2_data = circuit_data['l2_circuit']
                        service_info['Circuit_Name'] = str(l2_data.get('circuit_name', service_info['Circuit_Name']))
                        service_info['Customer_Name'] = str(l2_data.get('customer_name', service_info['Customer_Name']))
                        service_info['Description'] = str(l2_data.get('description', service_info['Description']))
                    
                    # Pattern 3: Check for circuit services array
                    if 'circuit_services' in circuit_data and 'circuit_service' in circuit_data['circuit_services']:
                        for circuit_service in circuit_data['circuit_services']['circuit_service']:
                            sub_service_info = service_info.copy()
                            sub_service_info['Circuit_Name'] = str(circuit_service.get('circuit_name', ''))
                            sub_service_info['Customer_Name'] = str(circuit_service.get('customer_name', ''))
                            sub_service_info['Description'] = str(circuit_service.get('description', ''))
                            services_list.append(sub_service_info)
                        continue
                
                # Ensure all values are strings (not lists or other unhashable types)
                for key, value in service_info.items():
                    if isinstance(value, (list, dict, set)):
                        service_info[key] = str(value)
                    elif value is None:
                        service_info[key] = ''
                    else:
                        service_info[key] = str(value)
                
                services_list.append(service_info)
                
            except Exception as e:
                print(f"DEBUG: Error extracting L2 circuit data for {circuit_id}: {e}")
                continue
                
    except Exception as e:
        print(f"DEBUG: Error in extract_l2circuit_services_from_ref_data: {e}")
    
    print(f"DEBUG: Final L2 Circuit services_list length: {len(services_list)}")
    return services_list

def extract_evpn_services_from_ref_data(ref_data):
    """Extract EVPN service information from the ref_data structure and convert to DataFrame"""
    services_list = []
    print(f"DEBUG: extract_evpn_services_from_ref_data called with keys: {list(ref_data.keys()) if isinstance(ref_data, dict) else 'Not a dict'}")
    
    try:
        for evpn_id, evpn_data in ref_data.items():
            print(f"DEBUG: Processing EVPN ID: {evpn_id}")
            
            try:
                # Common EVPN structure patterns
                service_info = {
                    'Service_Type': 'EVPN',
                    'EVPN_ID': str(evpn_id),
                    'EVPN_Name': '',
                    'Customer_Name': '',
                    'Description': '',
                    'VNI': '',
                    'Route_Target': '',
                    'Bridge_Domain': '',
                    'Status': '',
                    'Endpoint_Count': 0,
                    'MAC_VRF_Count': 0,
                    'Endpoints': ''
                }
                
                if isinstance(evpn_data, dict):
                    # Pattern 1: Direct EVPN information
                    service_info['EVPN_Name'] = str(evpn_data.get('evpn_name', evpn_data.get('name', '')))
                    service_info['Customer_Name'] = str(evpn_data.get('customer_name', evpn_data.get('customer', '')))
                    service_info['Description'] = str(evpn_data.get('description', evpn_data.get('desc', '')))
                    service_info['Status'] = str(evpn_data.get('status', evpn_data.get('admin_status', '')))
                    service_info['VNI'] = str(evpn_data.get('vni', evpn_data.get('vxlan_id', '')))
                    service_info['Route_Target'] = str(evpn_data.get('route_target', evpn_data.get('rt', '')))
                    service_info['Bridge_Domain'] = str(evpn_data.get('bridge_domain', evpn_data.get('bd', '')))
                    
                    # Pattern 2: L2VPN EVPN nested structure
                    if 'l2vpn_evpn' in evpn_data:
                        evpn_service_data = evpn_data['l2vpn_evpn']
                        service_info['EVPN_Name'] = str(evpn_service_data.get('evpn_name', service_info['EVPN_Name']))
                        service_info['Customer_Name'] = str(evpn_service_data.get('customer_name', service_info['Customer_Name']))
                        service_info['Description'] = str(evpn_service_data.get('description', service_info['Description']))
                        
                        # Extract EVPN instances
                        if 'evpn_instances' in evpn_service_data and 'evpn_instance' in evpn_service_data['evpn_instances']:
                            instances = evpn_service_data['evpn_instances']['evpn_instance']
                            if isinstance(instances, list):
                                service_info['Endpoint_Count'] = len(instances)
                                
                                # Process instances for additional details
                                for instance in instances:
                                    if isinstance(instance, dict):
                                        if not service_info['VNI'] and 'vni' in instance:
                                            service_info['VNI'] = str(instance['vni'])
                                        if not service_info['Route_Target'] and 'route_target' in instance:
                                            service_info['Route_Target'] = str(instance['route_target'])
                    
                    # Pattern 3: EVPN services array
                    if 'evpn_services' in evpn_data and 'evpn_service' in evpn_data['evpn_services']:
                        for evpn_service in evpn_data['evpn_services']['evpn_service']:
                            sub_service_info = service_info.copy()
                            sub_service_info['EVPN_Name'] = str(evpn_service.get('evpn_name', ''))
                            sub_service_info['Customer_Name'] = str(evpn_service.get('customer_name', ''))
                            sub_service_info['Description'] = str(evpn_service.get('description', ''))
                            sub_service_info['VNI'] = str(evpn_service.get('vni', ''))
                            
                            # Ensure all values are strings
                            for key, value in sub_service_info.items():
                                if isinstance(value, (list, dict, set)):
                                    sub_service_info[key] = str(value)
                                elif value is None:
                                    sub_service_info[key] = ''
                                else:
                                    sub_service_info[key] = str(value)
                            
                            services_list.append(sub_service_info)
                        continue
                    
                    # Extract MAC VRFs if present
                    if 'mac_vrfs' in evpn_data:
                        mac_vrfs = evpn_data['mac_vrfs']
                        if isinstance(mac_vrfs, list):
                            service_info['MAC_VRF_Count'] = len(mac_vrfs)
                        elif isinstance(mac_vrfs, dict) and 'mac_vrf' in mac_vrfs:
                            mac_vrf_list = mac_vrfs['mac_vrf']
                            if isinstance(mac_vrf_list, list):
                                service_info['MAC_VRF_Count'] = len(mac_vrf_list)
                    
                    # Extract endpoints/interfaces
                    endpoints = []
                    if 'endpoints' in evpn_data:
                        endpoints = evpn_data['endpoints']
                    elif 'interfaces' in evpn_data:
                        endpoints = evpn_data['interfaces']
                    elif 'evpn_interfaces' in evpn_data:
                        endpoints = evpn_data['evpn_interfaces']
                    
                    if isinstance(endpoints, list) and endpoints:
                        endpoint_names = []
                        for endpoint in endpoints[:3]:  # Show first 3 endpoints
                            if isinstance(endpoint, dict):
                                endpoint_name = endpoint.get('interface', endpoint.get('name', str(endpoint)))
                                endpoint_names.append(str(endpoint_name))
                            else:
                                endpoint_names.append(str(endpoint))
                        
                        service_info['Endpoints'] = ', '.join(endpoint_names)
                        if len(endpoints) > 3:
                            service_info['Endpoints'] += '...'
                    else:
                        service_info['Endpoints'] = ''
                
                # Ensure all values are strings (not lists or other unhashable types)
                for key, value in service_info.items():
                    if isinstance(value, (list, dict, set)):
                        service_info[key] = str(value)
                    elif value is None:
                        service_info[key] = ''
                    else:
                        service_info[key] = str(value)
                
                services_list.append(service_info)
                
            except Exception as e:
                print(f"DEBUG: Error extracting EVPN service data for {evpn_id}: {e}")
                continue
                
    except Exception as e:
        print(f"DEBUG: Error in extract_evpn_services_from_ref_data: {e}")
    
    print(f"DEBUG: Final EVPN services_list length: {len(services_list)}")
    return services_list

def display_service_forms(form_templates, metadata=None):
    """Display collapsible forms for service configuration with unique keys for multiple services"""
    if not form_templates:
        st.error("❌ No form templates provided")
        return None
    
    st.markdown("---")
    st.markdown("### 📋 Service Configuration Forms")
    
    service_type = form_templates[0]["fields"][0]["value"] if form_templates else "l2circuit"
    total_services = len(form_templates)
    
    # Enhanced message for default values approval
    service_display_name = "L2 Circuit" if service_type == "l2circuit" else "EVPN ELAN" if service_type == "evpn" else service_type.upper()
    
    st.markdown(f"""
    <div class="confirmation-panel">
        <h4>📝 Review and Confirm Configuration for {total_services} {service_display_name} service(s)</h4>
        <p><strong>🔧 Default Values Pre-populated:</strong> The system has filled in default values. 
        Please review and modify any values as needed before proceeding.</p>
        <p><strong>📁 Collapsible Forms:</strong> Each service has its own expandable section below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for form data if not exists
    if 'service_form_data' not in st.session_state:
        st.session_state.service_form_data = {}
    
    filled_forms = []
    
    # Create individual collapsible forms for each service
    for i, form_template in enumerate(form_templates):
        service_index = form_template.get("service_index", i + 1)
        form_id = form_template.get("form_id", f"form_{i}")
        
        # Extract service name for the expander title
        service_name = "Unknown Service"
        for field in form_template.get("fields", []):
            if field["field_name"] == "service_name":
                service_name = field["value"]
                break
        
        # Create collapsible expander for each service
        with st.expander(f"🛠️ Service {service_index}: {service_name}", expanded=(i == 0)):  # First one expanded by default
            
            # Use unique form key for each service
            unique_form_key = f"service_config_form_{service_index}_{i}"
            
            with st.form(unique_form_key):
                st.markdown(f"""
                <div class="form-container">
                    <div class="form-header">
                        ⚙️ Configuration for {service_display_name} Service {service_index}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create form fields
                form_data = {
                    "service_index": service_index,
                    "form_id": form_id,
                    "fields": []
                }
                
                # Categorize fields based on service type
                if service_type == "l2circuit":
                    basic_fields = ["service_type", "customer_name", "service_name", "source_node", "dest_node"]
                    network_fields = ["source_peer_addr", "dest_peer_addr", "vc_id", "vlan_id"]
                    port_fields = ["source_port_id", "dest_port_id", "source_port_access_id", "dest_port_access_id"]
                elif service_type == "evpn":
                    basic_fields = ["service_type", "customer_name", "service_name", "source_node", "dest_node", "evpn_type"]
                    network_fields = ["vpn_id", "parent_service_id", "route_target", "source_rd", "dest_rd", "cvlan_id"]
                    port_fields = ["source_port_id", "dest_port_id", "source_network_access_id", "dest_network_access_id", "source_site_id", "dest_site_id"]
                    performance_fields = ["port_speed", "mac_limit"]
                else:
                    # Fallback for unknown service types
                    basic_fields = ["service_type", "customer_name", "service_name", "source_node", "dest_node"]
                    network_fields = []
                    port_fields = []
                    performance_fields = []
                
                # Basic Service Information Section
                st.markdown("#### 📋 Basic Service Information")
                basic_col1, basic_col2 = st.columns(2)
                
                field_counter = 0
                for field in form_template.get("fields", []):
                    if field["field_name"] in basic_fields:
                        field_name = field["field_name"]
                        display_name = field["display_name"]
                        field_type = field["type"]
                        field_value = field["value"]
                        field_required = field.get("required", False)
                        field_description = field.get("description", "")
                        
                        # Use unique widget key including service index
                        widget_key = f"{unique_form_key}_{field_name}"
                        
                        current_col = basic_col1 if field_counter % 2 == 0 else basic_col2
                        
                        with current_col:
                            st.markdown(f"**{display_name}{'*' if field_required else ''}**")
                            if field_description:
                                st.caption(field_description)
                            
                            if field_type == "select":
                                options = field.get("options", [])
                                if field_value and field_value in options:
                                    default_index = options.index(field_value)
                                else:
                                    default_index = 0
                                
                                selected_value = st.selectbox(
                                    f"Select {display_name}",
                                    options=options,
                                    index=default_index,
                                    key=widget_key,
                                    label_visibility="collapsed"
                                )
                            elif field_type == "number":
                                selected_value = st.number_input(
                                    f"Enter {display_name}",
                                    value=int(field_value) if str(field_value).isdigit() else 0,
                                    key=widget_key,
                                    label_visibility="collapsed"
                                )
                            else:  # text
                                selected_value = st.text_input(
                                    f"Enter {display_name}",
                                    value=field_value,
                                    key=widget_key,
                                    label_visibility="collapsed"
                                )
                            
                            form_data["fields"].append({
                                "field_name": field_name,
                                "display_name": display_name,
                                "type": field_type,
                                "value": str(selected_value),
                                "required": field_required,
                                "description": field_description
                            })
                        
                        field_counter += 1
                
                # Network Configuration Section
                if network_fields:
                    if service_type == "l2circuit":
                        st.markdown("#### 🌐 Network Configuration (Default Values)")
                        st.info("💡 These are system defaults that can be customized:")
                    elif service_type == "evpn":
                        st.markdown("#### 🌐 VPN & Routing Configuration (Default Values)")
                        st.info("💡 BGP EVPN routing configuration - modify as needed:")
                    
                    network_col1, network_col2 = st.columns(2)
                    
                    # Track EVPN type for conditional VLAN display
                    current_evpn_type = "untagged"
                    
                    for field in form_template.get("fields", []):
                        if field["field_name"] in network_fields:
                            field_name = field["field_name"]
                            display_name = field["display_name"]
                            field_type = field["type"]
                            field_value = field["value"]
                            field_required = field.get("required", False)
                            field_description = field.get("description", "")
                            
                            # Use unique widget key including service index
                            widget_key = f"{unique_form_key}_{field_name}"
                            
                            # Special handling for EVPN type selection
                            if field_name == "evpn_type":
                                current_col = network_col1 if field_counter % 2 == 0 else network_col2
                                with current_col:
                                    st.markdown(f"**🔧 {display_name}{'*' if field_required else ''}**")
                                    if field_description:
                                        st.caption(f"⚙️ {field_description}")
                                    
                                    options = field.get("options", ["untagged", "tagged"])
                                    if field_value and field_value in options:
                                        default_index = options.index(field_value)
                                    else:
                                        default_index = 0
                                    
                                    selected_value = st.selectbox(
                                        f"Select {display_name}",
                                        options=options,
                                        index=default_index,
                                        key=widget_key,
                                        label_visibility="collapsed",
                                        help="Choose 'tagged' for VLAN encapsulation or 'untagged' for plain Ethernet"
                                    )
                                    current_evpn_type = selected_value
                                    
                                    form_data["fields"].append({
                                        "field_name": field_name,
                                        "display_name": display_name,
                                        "type": field_type,
                                        "value": str(selected_value),
                                        "required": field_required,
                                        "description": field_description
                                    })
                            
                            # Special handling for CVLAN ID (conditional on EVPN type)
                            elif field_name == "cvlan_id":
                                current_col = network_col1 if field_counter % 2 == 0 else network_col2
                                
                                # Only show CVLAN ID if tagged is selected
                                evpn_type_key = f"{unique_form_key}_evpn_type"
                                if evpn_type_key in st.session_state:
                                    current_evpn_type = st.session_state[evpn_type_key]
                                
                                if current_evpn_type == "tagged":
                                    with current_col:
                                        st.markdown(f"**🏷️ {display_name}{'*' if current_evpn_type == 'tagged' else ''}**")
                                        st.caption(f"🔢 {field_description}")
                                        
                                        if field_type == "number":
                                            selected_value = st.number_input(
                                                f"Enter {display_name}",
                                                value=int(field_value) if str(field_value).isdigit() else 1000,
                                                min_value=1,
                                                max_value=4094,
                                                key=widget_key,
                                                label_visibility="collapsed",
                                                help=f"VLAN ID for tagged EVPN (1-4094). Default: {field_value}"
                                            )
                                        else:
                                            selected_value = st.text_input(
                                                f"Enter {display_name}",
                                                value=field_value,
                                                key=widget_key,
                                                label_visibility="collapsed",
                                                help=f"Default: {field_value}"
                                            )
                                        
                                        form_data["fields"].append({
                                            "field_name": field_name,
                                            "display_name": display_name,
                                            "type": field_type,
                                            "value": str(selected_value),
                                            "required": current_evpn_type == "tagged",
                                            "description": field_description
                                        })
                                else:
                                    # For untagged, add empty CVLAN ID
                                    form_data["fields"].append({
                                        "field_name": field_name,
                                        "display_name": display_name,
                                        "type": field_type,
                                        "value": "",
                                        "required": False,
                                        "description": field_description
                                    })
                            
                            # Regular network fields
                            else:
                                current_col = network_col1 if field_counter % 2 == 0 else network_col2
                                
                                with current_col:
                                    st.markdown(f"**🔧 {display_name}{'*' if field_required else ''}**")
                                    if field_description:
                                        st.caption(f"⚙️ {field_description}")
                                    
                                    if field_type == "number":
                                        selected_value = st.number_input(
                                            f"Enter {display_name}",
                                            value=int(field_value) if str(field_value).isdigit() else 0,
                                            key=widget_key,
                                            label_visibility="collapsed",
                                            help=f"Default: {field_value}"
                                        )
                                    else:  # text
                                        selected_value = st.text_input(
                                            f"Enter {display_name}",
                                            value=field_value,
                                            key=widget_key,
                                            label_visibility="collapsed",
                                            help=f"Default: {field_value}"
                                        )
                                    
                                    form_data["fields"].append({
                                        "field_name": field_name,
                                        "display_name": display_name,
                                        "type": field_type,
                                        "value": str(selected_value),
                                        "required": field_required,
                                        "description": field_description
                                    })
                            
                            field_counter += 1
                
                # Port Configuration Section
                if port_fields:
                    if service_type == "l2circuit":
                        st.markdown("#### 🔌 Port Configuration (Default Values)")
                        st.info("💡 Default port settings - modify if needed:")
                    elif service_type == "evpn":
                        st.markdown("#### 🔌 Interface & Site Configuration (Default Values)")
                        st.info("💡 Interface and site identifiers - customize as required:")
                    
                    port_col1, port_col2 = st.columns(2)
                    
                    for field in form_template.get("fields", []):
                        if field["field_name"] in port_fields:
                            field_name = field["field_name"]
                            display_name = field["display_name"]
                            field_type = field["type"]
                            field_value = field["value"]
                            field_required = field.get("required", False)
                            field_description = field.get("description", "")
                            
                            # Use unique widget key including service index
                            widget_key = f"{unique_form_key}_{field_name}"
                            
                            current_col = port_col1 if field_counter % 2 == 0 else port_col2
                            
                            with current_col:
                                st.markdown(f"**🔌 {display_name}{'*' if field_required else ''}**")
                                if field_description:
                                    st.caption(f"🔧 {field_description}")
                                
                                selected_value = st.text_input(
                                    f"Enter {display_name}",
                                    value=field_value,
                                    key=widget_key,
                                    label_visibility="collapsed",
                                    help=f"Default: {field_value}"
                                )
                                
                                form_data["fields"].append({
                                    "field_name": field_name,
                                    "display_name": display_name,
                                    "type": field_type,
                                    "value": str(selected_value),
                                    "required": field_required,
                                    "description": field_description
                                })
                            
                            field_counter += 1
                
                # Performance Configuration Section (EVPN only)
                if service_type == "evpn" and "performance_fields" in locals() and performance_fields:
                    st.markdown("#### ⚡ Performance Configuration (Default Values)")
                    st.info("💡 Performance and capacity settings:")
                    
                    perf_col1, perf_col2 = st.columns(2)
                    
                    for field in form_template.get("fields", []):
                        if field["field_name"] in performance_fields:
                            field_name = field["field_name"]
                            display_name = field["display_name"]
                            field_type = field["type"]
                            field_value = field["value"]
                            field_required = field.get("required", False)
                            field_description = field.get("description", "")
                            
                            # Use unique widget key including service index
                            widget_key = f"{unique_form_key}_{field_name}"
                            
                            current_col = perf_col1 if field_counter % 2 == 0 else perf_col2
                            
                            with current_col:
                                st.markdown(f"**⚡ {display_name}{'*' if field_required else ''}**")
                                if field_description:
                                    st.caption(f"📊 {field_description}")
                                
                                if field_type == "number":
                                    selected_value = st.number_input(
                                        f"Enter {display_name}",
                                        value=int(field_value) if str(field_value).isdigit() else 0,
                                        key=widget_key,
                                        label_visibility="collapsed",
                                        help=f"Default: {field_value}"
                                    )
                                else:  # text
                                    selected_value = st.text_input(
                                        f"Enter {display_name}",
                                        value=field_value,
                                        key=widget_key,
                                        label_visibility="collapsed",
                                        help=f"Default: {field_value}"
                                    )
                                
                                form_data["fields"].append({
                                    "field_name": field_name,
                                    "display_name": display_name,
                                    "type": field_type,
                                    "value": str(selected_value),
                                    "required": field_required,
                                    "description": field_description
                                })
                            
                            field_counter += 1
                
                # Individual form submit button
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    service_submitted = st.form_submit_button(
                        f"✅ Save Service {service_index}",
                        use_container_width=True
                    )
                
                with col2:
                    service_reset = st.form_submit_button(
                        f"🔄 Reset Service {service_index}",
                        use_container_width=True
                    )
                
                # Handle individual service form submission
                if service_submitted:
                    # Validate this service's required fields
                    validation_errors = []
                    for field in form_data["fields"]:
                        if field["required"] and not field["value"].strip():
                            validation_errors.append(f"Service {service_index}: {field['display_name']} is required")
                    
                    if validation_errors:
                        st.error("❌ Please fill in all required fields:")
                        for error in validation_errors:
                            st.error(f"• {error}")
                    else:
                        # Store this service's data
                        st.session_state.service_form_data[f"service_{service_index}"] = form_data
                        st.success(f"✅ Service {service_index} configuration saved!")
                
                elif service_reset:
                    # Clear this service's data
                    if f"service_{service_index}" in st.session_state.service_form_data:
                        del st.session_state.service_form_data[f"service_{service_index}"]
                    st.info(f"🔄 Service {service_index} configuration reset to defaults")
                    st.rerun()
                
                # Always add current form data to filled_forms for submission
                filled_forms.append(form_data)
    
    # Global submission section outside all expanders
    st.markdown("---")
    st.markdown("### 🚀 Submit All Configurations")
    
    # Show saved services summary
    saved_services = list(st.session_state.service_form_data.keys())
    if saved_services:
        st.success(f"✅ Saved configurations: {len(saved_services)} of {total_services} services")
        for service_key in saved_services:
            service_data = st.session_state.service_form_data[service_key]
            service_name = "Unknown"
            for field in service_data["fields"]:
                if field["field_name"] == "service_name":
                    service_name = field["value"]
                    break
            st.info(f"📋 Service {service_data['service_index']}: {service_name}")
    else:
        st.warning("⚠️ No services saved yet. Please save individual service configurations above.")
    
    # Configuration summary based on service type
    if service_type == "l2circuit":
        st.markdown("""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; color: #155724;">
        <h5 style="color: #155724; margin-bottom: 10px;">📋 L2 Circuit Default Values Applied</h5>
        <p style="color: #155724; margin-bottom: 8px;"><strong>🔧 Network:</strong> Peer addresses (10.40.40.6, 10.40.40.1), VC ID (100)</p>
        <p style="color: #155724; margin-bottom: 8px;"><strong>🔌 Ports:</strong> Source/dest ports (et-0/0/6, et-0/0/8), Access IDs (111)</p>
        <p style="color: #155724; margin-bottom: 8px;"><strong>🎲 VLAN:</strong> Random ID (1000-1100 range)</p>
        <p style="color: #155724; margin-bottom: 0px;"><strong>✏️ Modification:</strong> All values can be changed in forms above</p>
        </div>
        """, unsafe_allow_html=True)
    elif service_type == "evpn":
        st.markdown("""
        <div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #0c5460; color: #0c5460;">
        <h5 style="color: #0c5460; margin-bottom: 10px;">📋 EVPN ELAN Default Values Applied</h5>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>🌐 VPN:</strong> VPN ID (001), Route Target (0:7777:2)</p>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>🔀 Routing:</strong> Source RD (0:1234:11), Dest RD (0:1234:10)</p>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>🔌 Interfaces:</strong> et-0/0/5.0, Network Access IDs (PNH/TH-underlay-link1)</p>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>📍 Sites:</strong> PNH-site1, TH-site2</p>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>⚡ Performance:</strong> Port Speed (10000 Mbps), MAC Limit (1000)</p>
        <p style="color: #0c5460; margin-bottom: 8px;"><strong>🏷️ VLAN Type:</strong> Choose "tagged" for VLAN encapsulation or "untagged" for plain Ethernet</p>
        <p style="color: #0c5460; margin-bottom: 0px;"><strong>✏️ Modification:</strong> All values can be changed in forms above</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Global submit buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Submit All Services", key="submit_all_services"):
            # Check if all services are saved
            if len(saved_services) == total_services:
                # Prepare final submission data from saved services
                final_forms = []
                for service_key in sorted(saved_services):
                    final_forms.append(st.session_state.service_form_data[service_key])
                
                st.session_state.pending_form_submission = final_forms
                # Clear saved form data
                st.session_state.service_form_data = {}
                st.success("🚀 All services submitted for configuration generation!")
                st.rerun()
            else:
                st.error(f"❌ Please save all {total_services} service configurations first. Currently saved: {len(saved_services)}")
    
    with col2:
        if st.button("❌ Cancel All", key="cancel_all_services"):
            # Clear all saved data
            st.session_state.service_form_data = {}
            st.info("❌ All configurations cancelled")
            st.rerun()
    
    with col3:
        st.caption(f"💡 Save each service configuration individually, then submit all {total_services} services together.")
    
    return filled_forms

def display_delete_confirmation(delete_data, metadata=None):
    """Display delete service confirmation with service details and collapsible JSON"""
    if not delete_data:
        st.error("❌ Invalid delete confirmation data")
        return
    
    # Header
    service_type = delete_data.get('service_type', 'Unknown')
    service_name = delete_data.get('service_name', 'Unknown Service')
    instance_name = delete_data.get('instance_name', 'Unknown')
    customer_name = delete_data.get('customer_name', 'Unknown')
    source_node = delete_data.get('source_node', 'Unknown')
    dest_node = delete_data.get('dest_node', 'Unknown')
    
    st.markdown(f"""
    <div class="delete-confirmation-panel">
        <h3>⚠️ Confirm Service Deletion</h3>
        <p><strong>You are about to permanently delete the following service:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Service Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Service Type:** {service_name}")
    with col2:
        st.markdown(f"**Instance Name:** {instance_name}")
    with col3:
        st.markdown(f"**Customer:** {customer_name}")
    with col4:
        st.markdown(f"**Route:** {source_node} → {dest_node}")
    
    # Service Details
    st.markdown("### 📋 Service Details")
    
    service_details = delete_data.get('service_details', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Customer ID:** {service_details.get('customer_id', 'Unknown')[:8]}...")
        st.markdown(f"**Current Status:** {service_details.get('instance_status', 'Unknown')}")
    with col2:
        st.markdown(f"**Design ID:** {service_details.get('design_id', 'Unknown')}")
        st.markdown(f"**Design Version:** {service_details.get('design_version', 'Unknown')}")
    with col3:
        created_time = service_details.get('created_time', 'Unknown')
        updated_time = service_details.get('updated_time', 'Unknown')
        if created_time != 'Unknown':
            try:
                created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                st.markdown(f"**Created:** {created_dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.markdown(f"**Created:** {created_time}")
        if updated_time != 'Unknown':
            try:
                updated_dt = datetime.fromisoformat(updated_time.replace('Z', '+00:00'))
                st.markdown(f"**Updated:** {updated_dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.markdown(f"**Updated:** {updated_time}")
    
    # Collapsible JSON Configuration
    with st.expander(f"📄 Complete Service JSON Configuration for {instance_name}", expanded=False):
        delete_config = delete_data.get('delete_config', {})
        if delete_config:
            st.markdown("**Current Service Configuration:**")
            st.code(json.dumps(delete_config, indent=2), language="json")
            
            # Download button for JSON
            json_str = json.dumps(delete_config, indent=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label=f"📄 Download Service JSON",
                data=json_str,
                file_name=f"{instance_name}_service_config_{timestamp}.json",
                mime="application/json",
                key=f"json_dl_delete_{instance_name}_{timestamp}",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Service configuration data not available")
    
    # Warning
    warning_message = delete_data.get('warning', f"This will DELETE service '{instance_name}'. This action cannot be undone.")
    st.markdown(f"""
    <div class="delete-warning">
        <h4>⚠️ Warning</h4>
        <p>{warning_message}</p>
        <p><strong>This action is permanent and cannot be reversed!</strong></p>
        <p><strong>Service Type:</strong> {service_name}</p>
        <p><strong>Service Name:</strong> {instance_name}</p>
        <p><strong>Customer:</strong> {customer_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return delete_data

def debug_session_state():
    """Debug function to show session state contents"""
    with st.expander("🐛 Debug Session State", expanded=False):
        st.write("**Current Session State Keys:**")
        for key in st.session_state:
            if "form" in key.lower() or "pending" in key.lower():
                st.write(f"- {key}: {type(st.session_state[key])}")
        
        st.write("**All Session State:**")
        st.json({k: str(v)[:200] for k, v in st.session_state.items() if not k.startswith('_')})

def display_service_configurations(config_data, metadata=None):
    """Display service configurations with JUNOS CLI in collapsible format"""
    if not config_data or "generated_configs" not in config_data:
        st.error("❌ Invalid configuration data")
        return
    
    # Header
    service_type = config_data.get('service_type', 'Unknown')
    service_name = config_data.get('service_name', 'Unknown Service')
    total_services = config_data.get('total_services', 0)
    customer_name = config_data.get('customer_name', 'Unknown')
    source_node = config_data.get('source_node', 'Unknown')
    dest_node = config_data.get('dest_node', 'Unknown')
    
    # Generate a unique session identifier for this configuration display
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    st.markdown(f"""
    <div class="confirmation-panel">
        <h3>🚀 Service Configuration Ready</h3>
        <p>The following {service_name} service(s) have been configured and are ready for deployment:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Service Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"**Service Type:** {service_name}")
    with col2:
        st.markdown(f"**Total Services:** {total_services}")
    with col3:
        st.markdown(f"**Customer:** {customer_name}")
    with col4:
        st.markdown(f"**Route:** {source_node} → {dest_node}")
    
    # Configuration Details with CLI in collapsible sections
    st.markdown("### 📋 Configuration Details")
    
    for i, config in enumerate(config_data.get('generated_configs', [])):
        instance_id = config.get('instance_id', f'service_{i}')
        customer_id = config.get('customer_id', 'Unknown')
        vlan_id = config.get('vlan_id', 'N/A')
        
        # Service overview
        st.markdown(f"""
        <div class="service-config-card">
            <div class="service-config-header">
                📊 Service {i+1} of {total_services}: {instance_id}
            </div>
        """, unsafe_allow_html=True)
        
        # Service metadata in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Instance ID:** {instance_id}")
        with col2:
            st.markdown(f"**Customer ID:** {customer_id[:8]}...")
        with col3:
            if vlan_id and vlan_id != 'N/A':
                st.markdown(f"**VLAN ID:** {vlan_id}")
        
        # JSON Configuration (Collapsible)
        with st.expander(f"📄 JSON Configuration for {instance_id}", expanded=False):
            # Find the corresponding full JSON config
            full_configs = config_data.get('configurations', [])
            full_config = full_configs[i] if i < len(full_configs) else {}
            
            if full_config:
                st.code(json.dumps(full_config, indent=2), language="json")
                
                # Download button for JSON
                json_str = json.dumps(full_config, indent=2)
                st.download_button(
                    label=f"📄 Download JSON Configuration",
                    data=json_str,
                    file_name=f"{instance_id}_config.json",
                    mime="application/json",
                    key=f"json_dl_{instance_id}_{i}_{session_timestamp}",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Full JSON configuration not available")
        
        # JUNOS CLI Configuration (Collapsible)
        cli_configs = config.get('cli_configs', {})
        
        if cli_configs and not cli_configs.get('error') and not cli_configs.get('info'):
            with st.expander(f"🖥️ JUNOS CLI Configuration for {instance_id}", expanded=False):
                
                # Create tabs for each device if multiple devices
                if len(cli_configs) > 1:
                    device_tabs = st.tabs([f"📟 {device}" for device in cli_configs.keys()])
                    for device_idx, (tab, (device, cli_config)) in enumerate(zip(device_tabs, cli_configs.items())):
                        with tab:
                            st.markdown(f"**Device:** {device}")
                            st.code(cli_config, language="bash")
                            
                            # Create unique key using instance_id, device, and indices
                            safe_device = device.replace('-', '_').replace('.', '_')
                            unique_key = f"cli_dl_{instance_id}_{safe_device}_{i}_{device_idx}_{session_timestamp}"
                            
                            # Download button for each device CLI
                            st.download_button(
                                label=f"📄 Download CLI for {device}",
                                data=cli_config,
                                file_name=f"{instance_id}_{device}_config.txt",
                                mime="text/plain",
                                key=unique_key,
                                use_container_width=True
                            )
                else:
                    # Single device - display directly
                    device, cli_config = next(iter(cli_configs.items()))
                    st.markdown(f"**Target Device:** {device}")
                    st.code(cli_config, language="bash")
                    
                    # Create unique key for single device download
                    safe_device = device.replace('-', '_').replace('.', '_')
                    unique_key = f"cli_single_{instance_id}_{safe_device}_{i}_{session_timestamp}"
                    
                    # Download button
                    st.download_button(
                        label=f"📄 Download CLI Configuration",
                        data=cli_config,
                        file_name=f"{instance_id}_{device}_config.txt",
                        mime="text/plain",
                        key=unique_key,
                        use_container_width=True
                    )
        
        elif cli_configs.get('error'):
            with st.expander(f"⚠️ CLI Configuration Issue for {instance_id}", expanded=False):
                st.error(f"❌ CLI Generation Error: {cli_configs['error']}")
                st.info("💡 The service can still be created, but CLI preview is not available.")
        
        elif cli_configs.get('info'):
            with st.expander(f"ℹ️ CLI Configuration Info for {instance_id}", expanded=False):
                st.info(f"ℹ️ {cli_configs['info']}")
                st.caption("💡 CLI generation requires OpenAI API configuration.")
        
        else:
            with st.expander(f"⚠️ CLI Configuration for {instance_id}", expanded=False):
                st.warning("⚠️ No CLI configuration available for preview.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Summary section
    st.markdown("### 🎯 Deployment Summary")
    
    # Count devices and services
    total_devices = set()
    total_cli_configs = 0
    
    for config in config_data.get('generated_configs', []):
        cli_configs = config.get('cli_configs', {})
        if cli_configs and not cli_configs.get('error') and not cli_configs.get('info'):
            total_devices.update(cli_configs.keys())
            total_cli_configs += len(cli_configs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Services to Create", total_services)
    with col2:
        st.metric("Devices to Configure", len(total_devices))
    with col3:
        st.metric("CLI Configurations", total_cli_configs)
    
    if total_devices:
        st.markdown("**Devices to be configured:**")
        device_list = ", ".join(sorted(total_devices))
        st.markdown(f"📟 {device_list}")
    
    return config_data

def display_execution_results(results_data, metadata=None):
    """Display service creation execution results"""
    if not results_data:
        st.error("❌ Invalid results data")
        return
    
    # Header with overall status
    success = results_data.get('success', False)
    total_services = results_data.get('total_services', 0)
    successful_count = results_data.get('successful_count', 0)
    failed_count = results_data.get('failed_count', 0)
    
    if success:
        result_class = "result-success"
        status_icon = "🎉"
        status_text = "All Services Created Successfully!"
    elif successful_count > 0:
        result_class = "result-partial"
        status_icon = "⚠️"
        status_text = "Partial Success - Some Services Failed"
    else:
        result_class = "result-failed"
        status_icon = "❌"
        status_text = "Service Creation Failed"
    
    st.markdown(f"""
    <div class="execution-result {result_class}">
        <h3>{status_icon} {status_text}</h3>
        <p><strong>Total Services:</strong> {total_services} | <strong>Successful:</strong> {successful_count} | <strong>Failed:</strong> {failed_count}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    summary = results_data.get('summary', {})
    if summary.get('successful_services'):
        st.markdown("### ✅ Successfully Created Services:")
        for service in summary['successful_services']:
            st.markdown(f"- **{service}** - Active and ready for use")
    
    if summary.get('failed_services'):
        st.markdown("### ❌ Failed Services:")
        for service in summary['failed_services']:
            st.markdown(f"- **{service}** - Check detailed results below")
    
    # Detailed Results
    if st.expander("📊 Detailed Execution Results", expanded=False):
        for result in results_data.get('service_results', []):
            service_result = result.get('result', {})
            instance_id = result.get('instance_id', 'Unknown')
            
            if service_result.get('success', False):
                final_status = service_result.get('final_status', 'unknown')
                st.success(f"✅ **{instance_id}** - Status: {final_status}")
            else:
                st.error(f"❌ **{instance_id}** - Creation failed")
            
            # Show workflow steps if available
            steps = service_result.get('steps', [])
            if steps:
                with st.expander(f"Workflow Details for {instance_id}", expanded=False):
                    for step in steps:
                        step_status = step.get('status', 'unknown')
                        step_action = step.get('action', 'unknown')
                        
                        if step_status == 'success':
                            st.info(f"✅ Step {step.get('step', 0)}: {step_action}")
                        else:
                            st.error(f"❌ Step {step.get('step', 0)}: {step_action} - {step.get('error', 'Unknown error')}")

def display_workflow_results(workflow_data, metadata=None):
    """Display service creation workflow results"""
    if not workflow_data or "steps" not in workflow_data:
        st.error("❌ Invalid workflow data")
        return
    
    # Header
    service_type = workflow_data.get('service_type', 'Unknown')
    customer_id = workflow_data.get('customer_id', 'Unknown')
    instance_id = workflow_data.get('instance_id', 'Unknown')
    
    st.markdown(f"""
    <div class="table-header">
        <div>🚀 Service Creation Workflow - {service_type.upper()}</div>
        <div>Instance: {instance_id}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Service Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Service Type:** {service_type.upper()}")
    with col2:
        st.markdown(f"**Customer ID:** {customer_id}")
    with col3:
        st.markdown(f"**Instance ID:** {instance_id}")
    
    overall_success = workflow_data.get('success', False)
    
    # Final Status
    st.markdown("### 🎯 Final Result")
    
    if overall_success:
        final_status = workflow_data.get('final_status', 'unknown')
        provisioning_status = workflow_data.get('provisioning_status', 'unknown')
        
        if final_status == 'active':
            st.success(f"🎉 Service {instance_id} has been successfully created and is now ACTIVE!")
        elif provisioning_status == 'pending':
            st.warning(f"⏳ Service {instance_id} has been created but is still provisioning. Current status: {final_status}")
        else:
            st.info(f"ℹ️ Service {instance_id} has been created. Current status: {final_status}")
    else:
        st.error("❌ Service creation failed. Please check the workflow steps above for details.")

def parse_response_for_dataframe(response_text):
    """
    Enhanced function to parse the response and extract dataframe data for all service types,
    including service configurations, execution results, and deletion confirmations
    Returns: (has_dataframe, dataframe, remaining_text, data_metadata, workflow_data, config_data, execution_data, form_data, delete_data)
    """
    # First, try to handle truncated JSON by attempting to fix it
    if response_text.strip().endswith(',') or not response_text.strip().endswith('}'):
        print("DEBUG: Detected potentially truncated JSON, attempting to fix...")
        # Try to find the last complete object and close it properly
        lines = response_text.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.endswith('}') or line.endswith(']'):
                # Try to parse up to this line
                truncated_response = '\n'.join(lines[:i+1])
                # Add closing braces if needed
                brace_count = truncated_response.count('{') - truncated_response.count('}')
                bracket_count = truncated_response.count('[') - truncated_response.count(']')
                
                if brace_count > 0 or bracket_count > 0:
                    truncated_response += '}' * brace_count + ']' * bracket_count
                
                try:
                    test_data = json.loads(truncated_response)
                    print("DEBUG: Successfully fixed truncated JSON!")
                    response_text = truncated_response
                    break
                except:
                    continue
    
    try:
        # Try to parse as JSON
        data = json.loads(response_text)
        print(f"DEBUG: Parsed JSON successfully. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Check if this looks like a dataframe response
        if isinstance(data, dict):
            # Check for deletion confirmation
            if data.get('action_required') == 'delete_confirmation':
                print("DEBUG: Detected deletion confirmation requirement")
                
                service_type = data.get('service_type', 'unknown')
                service_name = data.get('service_name', 'Unknown Service')
                instance_name = data.get('instance_name', 'Unknown')
                
                clean_response = f"🤖 **Service Deletion Confirmation Required**\n\n⚠️ **Service Found:** {service_name} service '{instance_name}'\n🔍 **Details:** Review service configuration and confirm deletion\n⚠️ **Warning:** This action is permanent and cannot be undone"
                
                return False, None, clean_response, None, None, None, None, None, data
            
            # Check for form input requirement
            if data.get('action_required') == 'form_input':
                print("DEBUG: Detected form input requirement")
                
                service_type = data.get('service_type', 'unknown')
                message = data.get('message', 'Form input required')
                form_templates = data.get('form_templates', [])
                
                clean_response = f"🤖 **Configuration Forms Required**\n\n📋 {message}\n🛠️ **Service Type:** {service_type.upper()}\n📝 **Forms:** {len(form_templates)} service(s) need detailed configuration"
                
                return False, None, clean_response, None, None, None, None, data, None
            
            # Check for service configuration (user confirmation required)
            if data.get('action_required') == 'user_confirmation':
                print("DEBUG: Detected service configuration for user confirmation")
                
                service_type = data.get('service_type', 'unknown')
                service_name = data.get('service_name', 'Unknown Service')
                total_services = data.get('total_services', 0)
                
                clean_response = f"🤖 **Configuration Generated Successfully!**\n\n✅ Generated {total_services} {service_name} configuration(s)\n📊 Ready for deployment\n⚠️ **User confirmation required before proceeding**"
                
                return False, None, clean_response, None, None, data, None, None, None
            
            # Check for execution results
            if 'service_results' in data and 'total_services' in data:
                print("DEBUG: Detected service execution results")
                
                total_services = data.get('total_services', 0)
                successful_count = data.get('successful_count', 0)
                failed_count = data.get('failed_count', 0)
                
                if successful_count == total_services:
                    clean_response = f"🎉 **All Services Created Successfully!**\n\n✅ {successful_count}/{total_services} services are now active\n📊 Check detailed results below"
                elif successful_count > 0:
                    clean_response = f"⚠️ **Partial Success**\n\n✅ {successful_count}/{total_services} services created successfully\n❌ {failed_count} services failed\n📊 Check detailed results below"
                else:
                    clean_response = f"❌ **Service Creation Failed**\n\n❌ All {total_services} services failed to create\n📊 Check detailed results below"
                
                return False, None, clean_response, None, None, None, data, None, None
            
            # Check for workflow result (service creation or deletion)
            if data.get('success') is not None and 'steps' in data and ('service_type' in data or 'instance_name' in data):
                print("DEBUG: Detected service workflow result")
                
                # Format workflow response
                service_type = data.get('service_type', 'unknown')
                instance_id = data.get('instance_id', data.get('instance_name', 'unknown'))
                success = data.get('success', False)
                
                if success:
                    final_status = data.get('final_status', 'unknown')
                    if final_status == 'active':
                        clean_response = f"🎉 **Service Operation Successful!**\n\n✅ {service_type.upper()} service '{instance_id}' operation completed successfully!"
                    elif final_status == 'deleted':
                        clean_response = f"🗑️ **Service Deletion Successful!**\n\n✅ {service_type.upper()} service '{instance_id}' has been permanently deleted!"
                    else:
                        clean_response = f"⏳ **Service Operation In Progress...**\n\n✅ {service_type.upper()} service '{instance_id}' operation initiated.\n📊 Current status: **{final_status.upper()}**"
                else:
                    operation_type = "deletion" if any(step.get('action', '').startswith('delete') for step in data.get('steps', [])) else "creation"
                    clean_response = f"❌ **Service {operation_type.title()} Failed**\n\n{service_type.upper()} service '{instance_id}' {operation_type} could not be completed. Check the workflow details below."
                
                return False, None, clean_response, None, data, None, None, None, None
            
            # Check for GPT-4 analysis result format with ref_data
            if 'result' in data and isinstance(data['result'], dict):
                result_data = data['result']
                print(f"DEBUG: Found result data. Keys: {list(result_data.keys())}")
                print(f"DEBUG: data_type = {result_data.get('data_type')}")
                print(f"DEBUG: service_type = {result_data.get('service_type')}")
                print(f"DEBUG: has ref_data = {'ref_data' in result_data}")
                print(f"DEBUG: has data = {'data' in result_data}")
                print(f"DEBUG: action_required = {result_data.get('action_required')}")
                
                # Check for deletion confirmation from GPT-4 result
                if result_data.get('action_required') == 'delete_confirmation':
                    print("DEBUG: GPT-4 result requires deletion confirmation")
                    
                    service_type = result_data.get('service_type', 'l2circuit')
                    service_name = result_data.get('service_name', 'Unknown Service')
                    instance_name = result_data.get('instance_name', 'Unknown')
                    
                    clean_response = f"""🤖 **Routing Director Analysis:** {data.get('gpt4_analysis', {}).get('reasoning', 'Service deletion request analyzed.')}

⚠️ **Action Required:** Deletion Confirmation
🛠️ **Service Type:** {service_type.upper()}
🗑️ **Service Found:** {service_name} service '{instance_name}'
⚠️ **Warning:** This action is permanent and cannot be undone"""
                    
                    return False, None, clean_response, result_data, None, None, None, None, result_data
                
                # Check for form input requirement from GPT-4 result
                if result_data.get('action_required') == 'form_input':
                    print("DEBUG: GPT-4 result requires form input")
                    
                    service_type = result_data.get('service_type', 'l2circuit')
                    message = result_data.get('message', 'Form input required')
                    form_templates = result_data.get('form_templates', [])
                    
                    clean_response = f"""🤖 **Routing Director Analysis:** {data.get('gpt4_analysis', {}).get('reasoning', 'Service creation request analyzed.')}

📋 **Action Required:** Form Input
🛠️ **Service Type:** {service_type.upper()}
📝 **Message:** {message}
🔧 **Forms:** {len(form_templates)} service(s) need detailed configuration"""
                    
                    return False, None, clean_response, result_data, None, None, None, result_data, None
                
                # Check for service configuration confirmation
                if result_data.get('action_required') == 'user_confirmation':
                    print("DEBUG: GPT-4 generated service configurations for confirmation")
                    
                    service_type = result_data.get('service_type', 'l2circuit')
                    service_name = result_data.get('service_name', service_type.upper())
                    total_services = result_data.get('total_services', 0)
                    
                    clean_response = f"""🤖 **Routing Director Analysis:** {data.get('gpt4_analysis', {}).get('reasoning', 'Service configurations generated successfully.')}

✅ **Result:** Generated {total_services} {service_name} configuration(s)
📊 **Configurations Ready:** Please review and confirm below
🛠️ **Service Type:** {service_type.upper()}
⚠️ **Action Required:** User confirmation needed to proceed"""
                    
                    return False, None, clean_response, result_data, None, result_data, None, None, None
                
                # Check for validation errors
                if 'validation_errors' in result_data:
                    validation_errors = result_data['validation_errors']
                    error_list = '\n'.join([f"• {error}" for error in validation_errors])
                    
                    clean_response = f"""🤖 **Routing Director:** {data.get('gpt4_analysis', {}).get('reasoning', 'Request analyzed but validation failed.')}

❌ **Validation Errors:**
{error_list}

💡 **Suggestions:**
• Ensure customer name is valid and active
• Check that source and destination nodes exist
• Verify the service type is supported"""
                    
                    return False, None, clean_response, result_data, None, None, None, None, None
                
                # Check for file upload requirement
                if result_data.get('action_required') == 'file_upload':
                    service_type = result_data.get('service_type', 'l2circuit')
                    service_name = result_data.get('service_name', service_type.upper())
                    instructions = result_data.get('instructions', f'Please upload a JSON file for {service_name} configuration.')
                    message = result_data.get('message', 'File upload required.')
                    
                    # Return file upload response
                    file_upload_response = f"""🤖 **Routing Director Analysis:** {data.get('gpt4_analysis', {}).get('reasoning', 'Service creation requested.')}

📁 **Action Required:** File Upload
🛠️ **Service Type:** {service_name}
📋 **Instructions:** {instructions}

Please upload your JSON configuration file using the file uploader below."""
                    
                    return False, None, file_upload_response, result_data, None, None, None, None, None
                
                # Check if this is a services response
                if result_data.get('data_type') == 'services':
                    service_type = result_data.get('service_type', 'l3vpn')
                    print(f"DEBUG: Detected service type: {service_type}")
                    
                    # First try to use the DataFrame from the 'data' field if it exists and is populated
                    if 'data' in result_data and result_data['data']:
                        df_data = result_data['data']
                        print(f"DEBUG: Found data field with keys: {list(df_data.keys()) if isinstance(df_data, dict) else 'Not a dict'}")
                        
                        if isinstance(df_data, dict) and 'data' in df_data and df_data['data']:
                            try:
                                df = pd.DataFrame(df_data['data'])
                                print(f"DEBUG: Successfully created DataFrame from data field with shape: {df.shape}")
                                print(f"DEBUG: DataFrame columns: {list(df.columns)}")
                                print(f"DEBUG: Sample data: {df.head(1).to_dict() if not df.empty else 'Empty DataFrame'}")
                                
                                if not df.empty:
                                    # Extract metadata
                                    metadata = {
                                        'data_type': result_data.get('data_type'),
                                        'service_type': service_type,
                                        'service_name': result_data.get('service_name', service_type.upper()),
                                        'source': 'Routing Director MCP Analysis',
                                        'timestamp': datetime.now().isoformat(),
                                        'gpt4_analysis': data.get('gpt4_analysis', {}),
                                        'original_record_count': result_data.get('total_services', len(df))
                                    }
                                    
                                    # Format the remaining response
                                    gpt4_reasoning = data.get('gpt4_analysis', {}).get('reasoning', 'Successfully analyzed your request.')
                                    recommended_tool = data.get('gpt4_analysis', {}).get('recommended_tool', 'fetch_all_instances')
                                    service_name = metadata['service_name']
                                    
                                    clean_response = f"""🤖 **Routing Director Analysis:** {gpt4_reasoning}

✅ **Result:** Found {len(df)} {service_name} service(s) 
📊 **Tool Used:** {recommended_tool}
🔍 **Data Source:** {metadata['source']}
🛠️ **Service Type:** {service_type.upper()}"""
                                    
                                    return True, df, clean_response, metadata, None, None, None, None, None
                            except Exception as e:
                                print(f"DEBUG: Error creating DataFrame from data field: {e}")
                    
                    # If data field doesn't work, try ref_data extraction (mainly for L3VPN)
                    if 'ref_data' in result_data and result_data['ref_data']:
                        ref_data = result_data['ref_data']
                        print(f"DEBUG: Trying ref_data extraction. ref_data keys: {list(ref_data.keys()) if isinstance(ref_data, dict) else 'Not a dict'}")
                        
                        if ref_data and isinstance(ref_data, dict):
                            try:
                                # Route to appropriate extraction function based on service type
                                services_list = []
                                if service_type == 'l3vpn':
                                    services_list = extract_l3vpn_services_from_ref_data(ref_data)
                                elif service_type == 'l2circuit':
                                    services_list = extract_l2circuit_services_from_ref_data(ref_data)
                                elif service_type == 'evpn':
                                    services_list = extract_evpn_services_from_ref_data(ref_data)
                                else:
                                    # Fallback: try all extraction methods
                                    print(f"DEBUG: Unknown service type '{service_type}', trying all extraction methods")
                                    services_list = extract_l3vpn_services_from_ref_data(ref_data)
                                    if not services_list:
                                        services_list = extract_l2circuit_services_from_ref_data(ref_data)
                                    if not services_list:
                                        services_list = extract_evpn_services_from_ref_data(ref_data)
                                
                                print(f"DEBUG: Extracted {len(services_list)} services from ref_data for type {service_type}")
                                
                                if services_list:
                                    df = pd.DataFrame(services_list)
                                    print(f"DEBUG: Created DataFrame from ref_data with shape: {df.shape}")
                                    
                                    # Extract metadata
                                    metadata = {
                                        'data_type': result_data.get('data_type'),
                                        'service_type': service_type,
                                        'service_name': result_data.get('service_name', service_type.upper()),
                                        'source': 'Routing Director MCP Analysis (ref_data)',
                                        'timestamp': datetime.now().isoformat(),
                                        'gpt4_analysis': data.get('gpt4_analysis', {}),
                                        'original_record_count': result_data.get('total_services', len(df))
                                    }
                                    
                                    # Format the remaining response without the raw dataframe
                                    gpt4_reasoning = data.get('gpt4_analysis', {}).get('reasoning', 'Successfully analyzed your request.')
                                    recommended_tool = data.get('gpt4_analysis', {}).get('recommended_tool', 'fetch_all_instances')
                                    service_name = metadata['service_name']
                                    
                                    clean_response = f"""🤖 **Routing Director Analysis:** {gpt4_reasoning}

✅ **Result:** Found {len(df)} {service_name} service(s) 
📊 **Tool Used:** {recommended_tool}
🔍 **Data Source:** {metadata['source']}
🛠️ **Service Type:** {service_type.upper()}"""
                                    
                                    return True, df, clean_response, metadata, None, None, None, None, None
                                else:
                                    print(f"DEBUG: No services extracted from ref_data for service type {service_type}")
                            except Exception as e:
                                print(f"DEBUG: Error parsing {service_type} services from ref_data: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # If both data and ref_data don't work, provide debugging info
                    print(f"DEBUG: Unable to extract DataFrame for service type {service_type}")
                    print(f"DEBUG: result_data structure: {json.dumps(result_data, indent=2, default=str)[:1000]}...")
            
            # Check for direct dataframe response format
            elif data.get('data_type') == 'services' and 'data' in data:
                df_data = data['data']
                if isinstance(df_data, dict) and 'data' in df_data:
                    try:
                        df = pd.DataFrame(df_data['data'])
                        
                        # Extract metadata
                        metadata = {
                            'data_type': data.get('data_type'),
                            'service_type': data.get('service_type', 'unknown'),
                            'service_name': data.get('service_name', 'Services'),
                            'source': 'Direct API',
                            'timestamp': datetime.now().isoformat(),
                            'ref_data': data.get('ref_data', {})
                        }
                        
                        # Format the remaining response
                        clean_response = f"✅ Successfully retrieved {len(df)} service records."
                        
                        return True, df, clean_response, metadata, None, None, None, None, None
                    except Exception as e:
                        print(f"Error parsing direct dataframe: {e}")
            
            # Check for list of dictionaries (common API response format)
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                try:
                    df = pd.DataFrame(data)
                    metadata = {
                        'data_type': 'list_records',
                        'service_type': 'unknown',
                        'service_name': 'Records',
                        'source': 'List API Response',
                        'timestamp': datetime.now().isoformat()
                    }
                    clean_response = f"✅ Successfully parsed {len(df)} records into tabular format."
                    return True, df, clean_response, metadata, None, None, None, None, None
                except Exception as e:
                    print(f"Error parsing list to dataframe: {e}")
        
        # Check if it's a list directly
        elif isinstance(data, list) and len(data) > 0:
            try:
                if isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    metadata = {
                        'data_type': 'list_records',
                        'service_type': 'unknown',
                        'service_name': 'Records',
                        'source': 'Direct List',
                        'timestamp': datetime.now().isoformat()
                    }
                    clean_response = f"✅ Successfully parsed {len(df)} records into tabular format."
                    return True, df, clean_response, metadata, None, None, None, None, None
            except Exception as e:
                print(f"Error parsing direct list: {e}")
        
        print("DEBUG: No matching condition found for dataframe parsing")
        return False, None, response_text, None, None, None, None, None, None
        
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decode error: {e}")
        print(f"DEBUG: Response length: {len(response_text)}")
        print(f"DEBUG: Response ends with: '{response_text[-50:]}'")
        
        # Try to detect CSV-like data in plain text
        lines = response_text.strip().split('\n')
        if len(lines) > 1 and ',' in lines[0]:
            try:
                # Attempt to parse as CSV
                df = pd.read_csv(io.StringIO(response_text))
                if not df.empty:
                    metadata = {
                        'data_type': 'csv_text',
                        'service_type': 'unknown',
                        'service_name': 'CSV Data',
                        'source': 'Text CSV',
                        'timestamp': datetime.now().isoformat()
                    }
                    clean_response = f"✅ Successfully parsed CSV text into {len(df)} records."
                    return True, df, clean_response, metadata, None, None, None, None, None
            except Exception as e:
                print(f"Error parsing CSV text: {e}")
        
        return False, None, response_text, None, None, None, None, None, None
    except Exception as e:
        print(f"DEBUG: Unexpected error in parse_response_for_dataframe: {e}")
        import traceback
        traceback.print_exc()
        return False, None, response_text, None, None, None, None, None, None

def display_enhanced_dataframe(df, metadata=None, chat_index=None):
    """Display dataframe with enhanced features"""
    if df is None or df.empty:
        st.warning("📊 No data to display")
        return
    
    # Unique key for this dataframe instance with timestamp
    timestamp_key = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_key = f"df_{chat_index}_{timestamp_key}" if chat_index is not None else f"df_{timestamp_key}"
    
    # Container for the enhanced table
    with st.container():
        # Get service type info for display
        service_type = metadata.get('service_type', 'unknown') if metadata else 'unknown'
        service_name = metadata.get('service_name', 'Data') if metadata else 'Data'
        
        # Table Header with metadata
        st.markdown(f"""
        <div class="table-header">
            <div>📊 {service_name} Table ({len(df)} rows × {len(df.columns)} columns)</div>
            <div>{metadata.get('source', 'Unknown Source') if metadata else 'Data Table'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data quality analysis
        quality_info = analyze_dataframe_quality(df)
        quality_class = f"quality-{quality_info['quality']}"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="data-quality-indicator {quality_class}">
                Data Quality: {quality_info['quality'].title()} ({quality_info['score']:.0f}%)
            </div>
            """, unsafe_allow_html=True)
            
            if quality_info['insights']:
                st.caption("💡 " + " • ".join(quality_info['insights']))
        
        with col2:
            st.markdown("**📊 Actions:**")
        
        with col3:
            # CSV Export button with unique key
            csv_data = create_downloadable_csv(df)
            filename = f"{service_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key=f"csv_{unique_key}",
                use_container_width=True
            )
        
        # Display summary metrics
        st.markdown("### 📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_count = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_count}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024  # KB
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{memory_usage:.1f}</div>
                <div class="metric-label">Memory (KB)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display service type specific info if available
        if metadata and service_type != 'unknown':
            st.markdown(f"### 🛠️ {service_name} Details")
            info_cols = st.columns(3)
            
            with info_cols[0]:
                st.markdown(f"**Service Type:** {service_type.upper()}")
            
            with info_cols[1]:
                if 'timestamp' in metadata:
                    timestamp = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                    st.markdown(f"**Last Updated:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            with info_cols[2]:
                if 'gpt4_analysis' in metadata and metadata['gpt4_analysis']:
                    st.markdown("**Analysis:** GPT-4 Powered ✨")
        
        # Main dataframe display
        st.markdown("### 📋 Data Table")
        
        # Configure dataframe display
        height = min(600, len(df) * 35 + 100)
        
        if not df.empty:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=height,
                key=f"main_df_{unique_key}"
            )
        else:
            st.warning("🔍 No data to display")

def initialize_session_state():
    """Initialize session state variables with better management"""
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = MCPClient()
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pending_confirmation' not in st.session_state:
        st.session_state.pending_confirmation = None
    if 'pending_forms' not in st.session_state:
        st.session_state.pending_forms = None
    if 'pending_form_submission' not in st.session_state:
        st.session_state.pending_form_submission = None
    if 'pending_delete_confirmation' not in st.session_state:
        st.session_state.pending_delete_confirmation = None
    
    # Clear any old form states on restart
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith('form_state_') and key != 'form_state_current']
    for key in keys_to_remove:
        del st.session_state[key]

if st.sidebar.checkbox("🐛 Debug Mode"):
    debug_session_state()

def add_to_chat_history(user_message, assistant_response, is_gpt4=False, dataframe=None, metadata=None, workflow_data=None, config_data=None, execution_data=None, form_data=None, delete_data=None, is_create=False, is_form=False, is_delete=False):
    """Add a new message pair to chat history"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.chat_history.append({
        'timestamp': timestamp,
        'user': user_message,
        'assistant': assistant_response,
        'is_gpt4': is_gpt4,
        'is_create': is_create,
        'is_form': is_form,
        'is_delete': is_delete,
        'dataframe': dataframe,
        'metadata': metadata,
        'workflow_data': workflow_data,
        'config_data': config_data,
        'execution_data': execution_data,
        'form_data': form_data,
        'delete_data': delete_data
    })

def display_chat_history():
    """Display the chat history with enhanced dataframe support"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; color: #666;">
            🤖 Start a conversation by typing a message below!<br>
            <small>Try natural language like: "Create 2 L2 circuits from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"</small>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="user-message">
            {chat['user']}
            <div class="timestamp">You • {chat['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response - use different styling based on message type
        if chat.get('delete_data') is not None:
            message_class = "delete-message"
            assistant_prefix = "🗑️ Deletion Assistant"
        elif chat.get('form_data') is not None:
            message_class = "form-message"
            assistant_prefix = "📋 Form Assistant"
        elif chat.get('config_data') is not None:
            message_class = "confirmation-message"
            assistant_prefix = "🛠️ Configuration Assistant"
        elif chat.get('execution_data') is not None:
            message_class = "create-message"
            assistant_prefix = "🚀 Execution Assistant"
        elif chat.get('is_create', False):
            message_class = "create-message"
            assistant_prefix = "🚀 Service Creation Assistant"
        elif chat.get('is_delete', False):
            message_class = "delete-message"
            assistant_prefix = "🗑️ Service Deletion Assistant"
        elif chat.get('is_gpt4', False):
            message_class = "gpt4-message"
            assistant_prefix = "🤖 Routing Director Assistant"
        else:
            message_class = "assistant-message"
            assistant_prefix = "Assistant"
        
        assistant_message = chat['assistant'].replace('\n', '<br>')
        st.markdown(f"""
        <div class="{message_class}">
            {assistant_message}
            <div class="timestamp">{assistant_prefix} • {chat['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display deletion confirmation if present
        if chat.get('delete_data') is not None:
            display_delete_confirmation(chat['delete_data'], metadata=chat.get('metadata'))
        
        # Display service forms if present
        elif chat.get('form_data') is not None:
            display_service_forms(chat['form_data'].get('form_templates', []), metadata=chat.get('metadata'))
        
        # Display service configurations if present
        elif chat.get('config_data') is not None:
            display_service_configurations(chat['config_data'], metadata=chat.get('metadata'))
        
        # Display execution results if present
        elif chat.get('execution_data') is not None:
            display_execution_results(chat['execution_data'], metadata=chat.get('metadata'))
        
        # Display workflow results if present
        elif chat.get('workflow_data') is not None:
            display_workflow_results(chat['workflow_data'], metadata=chat.get('metadata'))
        
        # Display enhanced dataframe if present
        elif chat.get('dataframe') is not None and not chat['dataframe'].empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            display_enhanced_dataframe(
                chat['dataframe'], 
                metadata=chat.get('metadata'),
                chat_index=i
            )
            st.markdown('</div>', unsafe_allow_html=True)

async def connect_to_server(mcp_client, server_script_path):
    """Connect to MCP server"""
    success = await mcp_client.connect(server_script_path)
    return success

def run_async(coro):
    """Run async function in Streamlit using threading"""
    import concurrent.futures
    
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    # Run the async function in a separate thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def main():
    # Initialize session state first
    initialize_session_state()
    
    # Header
    st.title("🤖 Routing Director MCP")
    
    # Server selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        server_script = st.selectbox(
            "Select MCP Server:",
            options=["mcpServer.py"],
            help="Choose which MCP server to connect to"
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Connection status and controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.session_state.connection_status:
            st.markdown('<div class="connection-status connected">🟢 Connected to MCP Server</div>', unsafe_allow_html=True)
            
            if st.session_state.mcp_client.available_tools:
                st.caption(f"📋 {len(st.session_state.mcp_client.available_tools)} tools available")
        else:
            st.markdown('<div class="connection-status disconnected">🔴 Not Connected to MCP Server</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔌 Connect", disabled=st.session_state.connection_status):
            with st.spinner("Connecting..."):
                success = run_async(connect_to_server(st.session_state.mcp_client, server_script))
                if success:
                    st.session_state.connection_status = True
                    if st.session_state.mcp_client.gpt4_enabled:
                        st.success("✅ Connected with GPT-4!")
                    else:
                        st.warning("⚠️ Connected but GPT-4 not available")
                    st.rerun()
                else:
                    st.session_state.connection_status = False
                    st.error("❌ Connection failed")
    
    with col3:
        if st.button("🔌 Disconnect", disabled=not st.session_state.connection_status):
            st.session_state.connection_status = False
            st.session_state.mcp_client.gpt4_enabled = False
            st.success("Disconnected")
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.pending_confirmation = None
            st.session_state.pending_forms = None
            st.session_state.pending_form_submission = None
            st.session_state.pending_delete_confirmation = None
            st.success("Chat cleared!")
            st.rerun()
    
    # GPT-4 AI Hint Box
    if st.session_state.connection_status and st.session_state.mcp_client.gpt4_enabled:
        with st.expander("💡 Natural Language Service Creation & Deletion", expanded=False):
            st.markdown("""
                <div class="ai-hint">
                🤖 <strong>GPT-4 Powered Natural Language Service Management</strong><br>
                Create and delete network services using simple, natural language commands. The system will analyze your request and prompt for any missing details through interactive forms.
                <br><br>
                <strong>Supported Services:</strong>
                <ul>
                    <li>🔗 <strong>L2 Circuit</strong> - Layer 2 circuit services (fully supported with forms)</li>
                    <li>⚡ <strong>EVPN ELAN</strong> - Ethernet VPN ELAN services (fully supported with forms)</li>
                    <li>🌐 <strong>L3VPN</strong> - Layer 3 VPN services (coming soon)</li>
                </ul>
                <br>
                <strong>Service Creation Examples:</strong>
                <ul>
                    <li><strong>L2 Circuit:</strong> "Create an L2 circuit for customer SINET"</li>
                    <li><strong>L2 Circuit Detailed:</strong> "Create L2 circuit from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET with service name test-l2ckt"</li>
                    <li><strong>EVPN ELAN:</strong> "Create EVPN service for customer SINET from PNH-ACX7024-A1 to TH-ACX7100-A6"</li>
                    <li><strong>EVPN Detailed:</strong> "Deploy ethernet VPN ELAN service named evpn-test-001 for SINET between PNH and TH devices"</li>
                    <li><strong>Multiple Services:</strong> "Create 3 EVPN services from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"</li>
                </ul>
                <br>
                <strong>Service Deletion Examples:</strong>
                <ul>
                    <li><strong>L2 Circuit Deletion:</strong> "Delete l2circuit1-135006"</li>
                    <li><strong>EVPN Deletion:</strong> "Delete evpn002"</li>
                    <li><strong>Verbose Deletion:</strong> "Remove the EVPN service named evpn-test-001"</li>
                    <li><strong>Alternative:</strong> "Terminate service evpn1-060622"</li>
                </ul>
                <br>
                <strong>🚀 Enhanced Interactive Workflow:</strong><br>
                <strong>For Service Creation:</strong><br>
                1. <strong>Parse Request:</strong> GPT-4 extracts basic service details and detects service type (L2 Circuit/EVPN)<br>
                2. <strong>Interactive Forms:</strong> System presents service-specific forms for missing mandatory details<br>
                3. <strong>Form Validation:</strong> Real-time validation of all required fields<br>
                4. <strong>Generate Configs:</strong> Automatically creates JSON and JUNOS CLI configurations<br>
                5. <strong>User Confirmation:</strong> Review configurations in collapsible sections before deployment<br>
                6. <strong>Execute Services:</strong> 2-step provisioning workflow with real-time status<br>
                7. <strong>Results Summary:</strong> Complete success/failure reporting<br>
                <br>
                <strong>For Service Deletion:</strong><br>
                1. <strong>Extract Service Name:</strong> GPT-4 identifies the service to delete from natural language<br>
                2. <strong>Find Service:</strong> System searches across all customers to locate the service<br>
                3. <strong>Show Confirmation:</strong> Display complete service details with collapsible JSON configuration<br>
                4. <strong>User Confirmation:</strong> Review service details and confirm permanent deletion<br>
                5. <strong>Execute Deletion:</strong> 3-step deletion workflow (modify operation → create order → execute)<br>
                6. <strong>Verify Deletion:</strong> Confirm service removal and provide status updates<br>
                <br>
                <strong>🔗 L2 Circuit Features:</strong><br>
                • VLAN encapsulation with random VLAN ID generation<br>
                • LDP signaling configuration<br>
                • Pseudowire setup with VC ID<br>
                • Default peer addresses and port configurations<br>
                <br>
                <strong>⚡ EVPN ELAN Features:</strong><br>
                • BGP EVPN routing with route targets and route distinguishers<br>
                • Site-based configuration with location details<br>
                • MAC learning and loop prevention<br>
                • Interface and network access configuration<br>
                • Performance settings (port speed, MAC limits)<br>
                • Multi-site ELAN topology support
                </div>
                """, unsafe_allow_html=True)
    
    # Show available tools if connected
    if st.session_state.connection_status and st.session_state.mcp_client.available_tools:
        with st.expander("🔧 Available Tools", expanded=False):
            for tool in st.session_state.mcp_client.available_tools:
                tool_name = tool.get('name', 'Unknown')
                tool_desc = tool.get('description', 'No description')
                
                # Highlight specific tools
                if tool_name == 'analyze_query':
                    st.markdown(f"🤖 **{tool_name}** (GPT-4 Powered): {tool_desc}")
                elif tool_name == 'create_service_intelligent':
                    st.markdown(f"🚀 **{tool_name}** (Natural Language Creation): {tool_desc}")
                elif tool_name == 'delete_service':
                    st.markdown(f"🗑️ **{tool_name}** (Natural Language Deletion): {tool_desc}")
                elif tool_name == 'confirm_delete_service':
                    st.markdown(f"⚠️ **{tool_name}** (Confirmed Deletion): {tool_desc}")
                elif tool_name == 'submit_service_forms':
                    st.markdown(f"📋 **{tool_name}** (Interactive Forms): {tool_desc}")
                elif tool_name == 'execute_service_creation':
                    st.markdown(f"⚡ **{tool_name}** (Bulk Execution): {tool_desc}")
                elif tool_name == 'create_service':
                    st.markdown(f"🛠️ **{tool_name}** (JSON-based Creation): {tool_desc}")
                elif 'l3vpn' in tool_name.lower():
                    st.markdown(f"🌐 **{tool_name}**: {tool_desc}")
                elif 'l2circuit' in tool_name.lower():
                    st.markdown(f"🔗 **{tool_name}**: {tool_desc}")
                elif 'evpn' in tool_name.lower():
                    st.markdown(f"⚡ **{tool_name}**: {tool_desc}")
                else:
                    st.markdown(f"**{tool_name}**: {tool_desc}")
    
    # Check for pending form submission
    if st.session_state.pending_form_submission:
        with st.spinner("🚀 Processing forms and generating configurations..."):
            try:
                # Submit forms to server
                response = run_async(
                    st.session_state.mcp_client.submit_service_forms(
                        st.session_state.pending_form_submission
                    )
                )
                
                # Parse response for configuration results
                has_dataframe, dataframe, clean_response, metadata, workflow_data, config_data, execution_data, form_data, delete_data = parse_response_for_dataframe(response)
                
                # Add to chat history
                add_to_chat_history(
                    f"Submitted configuration forms for {len(st.session_state.pending_form_submission)} service(s)",
                    clean_response,
                    is_gpt4=False,
                    is_create=True,
                    dataframe=dataframe,
                    metadata=metadata,
                    workflow_data=workflow_data,
                    config_data=config_data,
                    execution_data=execution_data,
                    form_data=form_data,
                    delete_data=delete_data
                )
                
                # Set pending confirmation if config data available
                if config_data and config_data.get('action_required') == 'user_confirmation':
                    st.session_state.pending_confirmation = config_data
                
                # Clear pending form submission
                st.session_state.pending_form_submission = None
                
                st.success("✅ Forms processed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing forms: {str(e)}")
                st.session_state.pending_form_submission = None
    
    # Chat area
    st.markdown("---")
    
    # Display chat history in a container
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # Service deletion confirmation section
    if st.session_state.pending_delete_confirmation:
        st.markdown("---")
        st.markdown("### 🗑️ Service Deletion Confirmation")
        
        delete_data = st.session_state.pending_delete_confirmation
        instance_name = delete_data.get('instance_name', 'Unknown')
        service_type = delete_data.get('service_type', 'l2circuit')
        service_name = delete_data.get('service_name', 'Unknown Service')
        
        st.markdown(f"""
        <div class="delete-confirmation-panel">
            <h4>⚠️ Confirm Permanent Deletion of {service_name} Service</h4>
            <p>You are about to permanently delete service: <strong>{instance_name}</strong></p>
            <p><strong>WARNING:</strong> This action cannot be undone!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🗑️ Confirm Deletion", key="confirm_deletion"):
                with st.spinner(f"🗑️ Deleting service {instance_name}..."):
                    try:
                        # Execute service deletion
                        response = run_async(
                            st.session_state.mcp_client.confirm_delete_service(
                                instance_name, 
                                service_type
                            )
                        )
                        
                        # Parse response for execution results
                        has_dataframe, dataframe, clean_response, metadata, workflow_data, config_data_result, execution_data, form_data, delete_data_result = parse_response_for_dataframe(response)
                        
                        # Add to chat history
                        add_to_chat_history(
                            f"Confirmed deletion of {service_name} service '{instance_name}'",
                            clean_response,
                            is_gpt4=False,
                            is_delete=True,
                            dataframe=dataframe,
                            metadata=metadata,
                            workflow_data=workflow_data,
                            config_data=config_data_result,
                            execution_data=execution_data,
                            form_data=form_data,
                            delete_data=delete_data_result
                        )
                        
                        # Clear pending delete confirmation
                        st.session_state.pending_delete_confirmation = None
                        
                        st.success(f"🗑️ Service deletion completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error deleting service: {str(e)}")
        
        with col2:
            if st.button("❌ Cancel", key="cancel_deletion"):
                st.session_state.pending_delete_confirmation = None
                st.info("Service deletion cancelled")
                st.rerun()
        
        with col3:
            st.caption("💡 Review the service details above before confirming. Deletion cannot be undone.")
    
    # Service configuration confirmation section
    if st.session_state.pending_confirmation:
        st.markdown("---")
        st.markdown("### 🚀 Service Creation Confirmation")
        
        config_data = st.session_state.pending_confirmation
        total_services = config_data.get('total_services', 0)
        service_name = config_data.get('service_name', 'Unknown Service')
        
        st.markdown(f"""
        <div class="confirmation-panel">
            <h4>📋 Ready to Create {total_services} {service_name} Service(s)</h4>
            <p>Please review the configurations above and confirm to proceed with service creation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Confirm & Create", key="confirm_creation"):
                with st.spinner(f"🚀 Creating {total_services} service(s)..."):
                    try:
                        # Execute service creation
                        response = run_async(
                            st.session_state.mcp_client.execute_service_creation(
                                json.dumps(config_data), 
                                confirm=True
                            )
                        )
                        
                        # Parse response for execution results
                        has_dataframe, dataframe, clean_response, metadata, workflow_data, config_data_result, execution_data, form_data, delete_data = parse_response_for_dataframe(response)
                        
                        # Add to chat history
                        add_to_chat_history(
                            f"Confirmed creation of {total_services} {service_name} service(s)",
                            clean_response,
                            is_gpt4=False,
                            is_create=True,
                            dataframe=dataframe,
                            metadata=metadata,
                            workflow_data=workflow_data,
                            config_data=config_data_result,
                            execution_data=execution_data,
                            form_data=form_data,
                            delete_data=delete_data
                        )
                        
                        # Clear pending confirmation
                        st.session_state.pending_confirmation = None
                        
                        st.success(f"✅ Service creation completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error creating services: {str(e)}")
        
        with col2:
            if st.button("❌ Cancel", key="cancel_creation"):
                st.session_state.pending_confirmation = None
                st.info("Service creation cancelled")
                st.rerun()
        
        with col3:
            st.caption("💡 Review the service configurations above before confirming. The system will execute the complete 2-step provisioning workflow for each service.")
    
    # Chat input at the bottom
    if st.session_state.connection_status:
        st.markdown("---")
        
        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                placeholder_text = "Try: 'Create EVPN service for SINET' or 'Create L2 circuit from PNH-ACX7024-A1 to TH-ACX7100-A6' or 'Delete evpn002'" if st.session_state.mcp_client.gpt4_enabled else "Type your message..."
                user_input = st.text_input(
                    "Type your message:",
                    placeholder=placeholder_text,
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button("📤 Send", use_container_width=True)
            
            if send_button and user_input:
                try:
                    with st.spinner("🤖 Analyzing your request....." if st.session_state.mcp_client.gpt4_enabled else "Processing your request..."):
                        # Send message to MCP server
                        response = run_async(st.session_state.mcp_client.send_message(user_input))
                        
                        # Check if response contains dataframe data, workflow data, config data, execution data, form data, or delete data
                        has_dataframe, dataframe, clean_response, metadata, workflow_data, config_data, execution_data, form_data, delete_data = parse_response_for_dataframe(response)
                        
                        # Check if deletion confirmation is required
                        if delete_data and delete_data.get('action_required') == 'delete_confirmation':
                            # Set pending delete confirmation
                            st.session_state.pending_delete_confirmation = delete_data
                            
                            # Add to chat history
                            add_to_chat_history(
                                user_input,
                                clean_response,
                                is_gpt4=st.session_state.mcp_client.gpt4_enabled,
                                is_delete=True,
                                delete_data=delete_data
                            )
                        
                        # Check if forms are required
                        elif form_data and form_data.get('action_required') == 'form_input':
                            # Set pending forms
                            st.session_state.pending_forms = form_data
                            
                            # Add to chat history
                            add_to_chat_history(
                                user_input,
                                clean_response,
                                is_gpt4=st.session_state.mcp_client.gpt4_enabled,
                                is_form=True,
                                form_data=form_data
                            )
                        
                        # Check if config confirmation is required
                        elif config_data and config_data.get('action_required') == 'user_confirmation':
                            # Set pending confirmation
                            st.session_state.pending_confirmation = config_data
                            
                            # Add to chat history
                            add_to_chat_history(
                                user_input,
                                clean_response,
                                is_gpt4=st.session_state.mcp_client.gpt4_enabled,
                                is_create=True,
                                config_data=config_data
                            )
                        else:
                            # Add to chat history
                            add_to_chat_history(
                                user_input, 
                                clean_response if (has_dataframe or workflow_data or config_data or execution_data or form_data or delete_data) else response, 
                                is_gpt4=st.session_state.mcp_client.gpt4_enabled,
                                is_create=workflow_data is not None or config_data is not None or execution_data is not None,
                                is_form=form_data is not None,
                                is_delete=delete_data is not None or (workflow_data is not None and 'delete' in str(workflow_data).lower()),
                                dataframe=dataframe if has_dataframe else None,
                                metadata=metadata if has_dataframe else None,
                                workflow_data=workflow_data,
                                config_data=config_data,
                                execution_data=execution_data,
                                form_data=form_data,
                                delete_data=delete_data
                            )
                        
                        # Rerun to update chat display
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        st.warning("⚠️ Please connect to an MCP server to start chatting.")
        st.markdown("""
        ### 🚀 Getting Started:
        1. Click "**🔌 Connect**" to establish connection with the MCP server
        2. Once connected, GPT-4 will automatically analyze your queries and detect service types
        3. Use natural language to create services like:
           - **L2 Circuit:** "Create an L2 circuit for customer SINET" (system will prompt for missing details)
           - **L2 Circuit Detailed:** "Create L2 circuit from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET with service name test-circuit"
           - **EVPN ELAN:** "Create EVPN service for customer SINET from PNH-ACX7024-A1 to TH-ACX7100-A6"
           - **EVPN Detailed:** "Deploy ethernet VPN ELAN service named evpn-test-001 for SINET between PNH and TH devices"
           - **Multiple Services:** "Create 3 EVPN services from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"
        4. Use natural language to delete services like:
           - **L2 Circuit Deletion:** "Delete l2circuit1-135006" (system will show confirmation with service details)
           - **EVPN Deletion:** "Delete evpn002" or "Remove the EVPN service named evpn-test-001"
           - **Alternative:** "Terminate service evpn1-060622"
        5. The system will:
           - Parse your request using GPT-4 and detect service type (L2 Circuit/EVPN/L3VPN)
           - Present interactive service-specific forms for any missing mandatory details (for creation)
           - Show confirmation details with collapsible JSON configuration (for deletion)
           - Generate appropriate JSON configurations with JUNOS CLI (for creation)
           - Ask for your confirmation with collapsible config sections
           - Execute the complete provisioning/deletion workflow
           - Provide real-time status updates
        6. View services with queries like:
           - **"Show me all L3VPN service instances"**
           - **"Display L2 circuit services"**
           - **"Show me EVPN services"**
           - **"Show me order history"**
        
        ### 🛠️ Supported Network Services:
        - **🔗 L2 Circuit** - Layer 2 circuit services with VLAN encapsulation (fully supported with interactive forms and deletion)
        - **⚡ EVPN ELAN** - Ethernet VPN ELAN services with BGP EVPN (fully supported with interactive forms and deletion)
        - **🌐 L3VPN** - Layer 3 VPN services (viewing supported, creation/deletion coming soon)
        
        ### 🚀 Enhanced Interactive Service Management Features:
        - **Natural Language Processing:** Describe what you want in plain English with automatic service type detection
        - **Service-Specific Forms:** L2 Circuit and EVPN have different form layouts tailored to their requirements
        - **Interactive Forms:** System prompts for missing mandatory details through user-friendly forms
        - **Real-time Validation:** Forms validate required fields before submission
        - **Intelligent Defaults:** Forms pre-populate with service-appropriate defaults based on your request
        - **Multi-Service Support:** Create multiple services with individual forms for each
        - **Automatic Configuration Generation:** JSON configs and JUNOS CLI created automatically for each service type
        - **Service-Aware Display:** Collapsible views organized by service type (L2 Circuit vs EVPN)
        - **User Confirmation:** Review before deployment with expandable config sections
        - **Service Deletion:** Intelligent service lookup with confirmation workflow for any service type
        - **2-Step Workflow:** Upload service → Deploy service (creation) / Modify → Execute (deletion)
        - **Real-time Status:** Monitor progress and final results
        - **Bulk Operations:** Handle multiple services efficiently
        
        ### 💡 Example Queries:
        **L2 Circuit Creation:**
        - "Create an L2 circuit for customer SINET" (system will ask for missing details)
        - "Create L2 circuit from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"
        - "Create 5 L2 circuits for SINET customer between two devices"
        - "Deploy L2 circuit services with custom VLAN and peer addresses"
        
        **EVPN ELAN Creation:**
        - "Create EVPN service for customer SINET" (system will ask for missing details)
        - "Create ethernet VPN from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"
        - "Deploy ELAN service named evpn-test-001 for SINET between PNH and TH"
        - "Create 3 EVPN services with custom route targets and site configurations"
        
        **Service Deletion (any type):**
        - "Delete l2circuit1-135006" (system will find and show service details)
        - "Remove service evpn002"
        - "Terminate the EVPN service named evpn-test-001"
        - "Delete service l2ckt-abc-123"
        
        **Service Viewing:**
        - "Show me all L3VPN services"
        - "Display L2 circuit instances"
        - "What EVPN services are running?"
        - "Show me the order history"
        
        **Information:**
        - "What devices are available?"
        - "Show me active customers"
        - "What API endpoints are available?"
        """)
    
    # Footer with GPT-4 info
    st.markdown("---")
    if st.session_state.connection_status and st.session_state.mcp_client.gpt4_enabled:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            🤖 This chat is powered by <strong>Routing Director & OpenAI GPT-4 Intelligence</strong><br>
            <small>Natural language service creation & deletion • Interactive forms • Intelligent routing • Real-time provisioning</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()