import json
import uuid
import random
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EnhancedServiceConfigGenerator:
    """Generate service configuration JSON bodies using OpenAI for parsing and JUNOS config generation"""
    
    def __init__(self, customers_file: str = "services/customers.json", devices_file: str = "services/devices.json"):
        self.customers_data = self._load_json_file(customers_file)
        self.devices_data = self._load_json_file(devices_file)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Service type configurations
        self.service_configs = {
            "l2circuit": {
                "design_id": "eline-l2circuit-nsm",
                "design_version": "0.5.11",
                "workflow_id": "create-eline-l2circuit-nsm-0_5_11"
            },
            "l3vpn": {
                "design_id": "l3vpn",
                "design_version": "0.7.33",
                "workflow_id": "create-l3vpn-0_7_33"
            },
            "evpn": {
                "design_id": "elan-evpn-csm",
                "design_version": "0.5.16",
                "workflow_id": "create-elan-evpn-csm-0_5_16"
            }
        }
        
        # Fixed organization ID
        self.org_id = "0eaf8613-632d-41d2-8de4-c2d242325d7e"
        
        # VLAN ID range for L2 circuits
        self.vlan_range = (1000, 1100)
        self.used_vlans = set()
        
        # Default configurations for mandatory fields
        self.default_configs = {
            "l2circuit": {
                "peer_addresses": {
                    "source": "10.40.40.6",
                    "dest": "10.40.40.1"
                },
                "port_configs": {
                    "source": {"port_id": "et-0/0/6", "id": "111"},
                    "dest": {"port_id": "et-0/0/8", "id": "111"}
                },
                "vc_id": "100",
                "vpn_network_access_id": "1"
            }
        }
        
        # Template file paths
        self.template_files = {
            "l2circuit": "services/l2circuit_template.json",
            "l3vpn": "services/l3vpn_template.json",
            "evpn": "services/evpn_template.json"
        }
        
        self.services_dir = Path("services")
        self.services_dir.mkdir(exist_ok=True)
        logger.info(f"Services directory created/verified: {self.services_dir}")
    
    def get_mandatory_fields(self, service_type: str) -> List[Dict[str, Any]]:
        """Get list of mandatory fields for a service type"""
        if service_type == "l2circuit":
            return [
                {
                    "field_name": "service_type",
                    "display_name": "Service Type",
                    "type": "select",
                    "options": ["l2circuit"],
                    "default": "l2circuit",
                    "required": True,
                    "description": "Type of service to create"
                },
                {
                    "field_name": "customer_name",
                    "display_name": "Customer Name",
                    "type": "select",
                    "options": [c.get("name", "") for c in self.customers_data if c.get("status") == "active"],
                    "default": "",
                    "required": True,
                    "description": "Name of the customer for this service"
                },
                {
                    "field_name": "service_name",
                    "display_name": "Service Name",
                    "type": "text",
                    "default": "",
                    "required": True,
                    "description": "Unique name for this service instance"
                },
                {
                    "field_name": "source_node",
                    "display_name": "Source Node",
                    "type": "select",
                    "options": [d.get("hostname", "") for d in self.devices_data.get("devices", [])],
                    "default": "",
                    "required": True,
                    "description": "Source device hostname"
                },
                {
                    "field_name": "dest_node",
                    "display_name": "Destination Node",
                    "type": "select",
                    "options": [d.get("hostname", "") for d in self.devices_data.get("devices", [])],
                    "default": "",
                    "required": True,
                    "description": "Destination device hostname"
                },
                {
                    "field_name": "source_peer_addr",
                    "display_name": "Source Peer Address",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["peer_addresses"]["source"],
                    "required": True,
                    "description": f"Peer address for source node (Default: {self.default_configs['l2circuit']['peer_addresses']['source']})"
                },
                {
                    "field_name": "dest_peer_addr",
                    "display_name": "Destination Peer Address",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["peer_addresses"]["dest"],
                    "required": True,
                    "description": f"Peer address for destination node (Default: {self.default_configs['l2circuit']['peer_addresses']['dest']})"
                },
                {
                    "field_name": "vc_id",
                    "display_name": "VC ID",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["vc_id"],
                    "required": True,
                    "description": f"Virtual circuit identifier (Default: {self.default_configs['l2circuit']['vc_id']})"
                },
                {
                    "field_name": "vlan_id",
                    "display_name": "VLAN ID",
                    "type": "number",
                    "default": str(self._generate_random_vlan()),
                    "required": True,
                    "description": f"VLAN identifier for the service (Random between {self.vlan_range[0]}-{self.vlan_range[1]})"
                },
                {
                    "field_name": "source_port_id",
                    "display_name": "Source Port ID",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["port_configs"]["source"]["port_id"],
                    "required": True,
                    "description": f"Physical port on source device (Default: {self.default_configs['l2circuit']['port_configs']['source']['port_id']})"
                },
                {
                    "field_name": "dest_port_id",
                    "display_name": "Destination Port ID",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["port_configs"]["dest"]["port_id"],
                    "required": True,
                    "description": f"Physical port on destination device (Default: {self.default_configs['l2circuit']['port_configs']['dest']['port_id']})"
                },
                {
                    "field_name": "source_port_access_id",
                    "display_name": "Source Port Access ID",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["port_configs"]["source"]["id"],
                    "required": True,
                    "description": f"Access ID for source port (Default: {self.default_configs['l2circuit']['port_configs']['source']['id']})"
                },
                {
                    "field_name": "dest_port_access_id",
                    "display_name": "Destination Port Access ID",
                    "type": "text",
                    "default": self.default_configs["l2circuit"]["port_configs"]["dest"]["id"],
                    "required": True,
                    "description": f"Access ID for destination port (Default: {self.default_configs['l2circuit']['port_configs']['dest']['id']})"
                }
            ]
        return []
    
        # Replace the validate_mandatory_fields method in your existing services_generator.py with this updated version:

    def validate_mandatory_fields(self, parsed_request: Dict[str, Any], service_type: str) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """
        Validate if all mandatory fields are present and return missing fields for form generation
        Returns: (is_complete, missing_fields_list, form_templates)
        """
        mandatory_fields = self.get_mandatory_fields(service_type)
        missing_fields = []
        form_templates = []
        
        # Check basic fields from parsed request - removed service_name check since we have service_names
        basic_fields = ["customer_name", "source_node", "dest_node"]
        missing_basic = []
        
        for field in basic_fields:
            if not parsed_request.get(field):
                missing_basic.append(field)
        
        # If basic fields are missing, we can't proceed to detailed form
        if missing_basic:
            return False, missing_basic, []
        
        # Generate form templates for each service
        quantity = parsed_request.get("quantity", 1)
        service_names = parsed_request.get("service_names", [])
        
        # Check if we have enough valid service names for the quantity requested
        valid_service_names = [name.strip() for name in service_names if isinstance(name, str) and name.strip()]
        has_all_service_names = len(valid_service_names) >= quantity
        
        for i in range(quantity):
            # Use provided service name or generate one
            if service_names and i < len(service_names):
                service_name = self._sanitize_service_name(service_names[i])
            else:
                service_name = self._generate_service_name(service_type, i + 1)
            
            # Create form template for this service
            form_template = {
                "service_index": i + 1,
                "form_id": f"{service_type}_form_{i}",
                "fields": []
            }
            
            # Populate form fields with default values
            for field_def in mandatory_fields:
                field_value = field_def["default"]
                
                # Override with parsed request values where available
                if field_def["field_name"] == "customer_name":
                    field_value = parsed_request.get("customer_name", field_def["default"])
                elif field_def["field_name"] == "service_name":
                    field_value = service_name
                elif field_def["field_name"] == "source_node":
                    field_value = parsed_request.get("source_node", field_def["default"])
                elif field_def["field_name"] == "dest_node":
                    field_value = parsed_request.get("dest_node", field_def["default"])
                elif field_def["field_name"] == "service_type":
                    field_value = service_type
                
                form_field = {
                    "field_name": field_def["field_name"],
                    "display_name": field_def["display_name"],
                    "type": field_def["type"],
                    "value": field_value,
                    "required": field_def["required"],
                    "description": field_def["description"]
                }
                
                # Add options for select fields
                if field_def["type"] == "select":
                    form_field["options"] = field_def["options"]
                
                form_template["fields"].append(form_field)
            
            form_templates.append(form_template)
        
        # FORCE FORMS FOR L2 CIRCUITS - Always show forms for user approval of default values
        if service_type == "l2circuit":
            logger.info(f"🔍 L2 Circuit service detected - forcing forms for user approval of default values")
            return False, [], form_templates  # Always show forms for L2 circuits
        
        # Original logic for other service types
        # If we have all basic fields and service names, proceed directly to config generation
        if has_all_service_names:
            return True, [], form_templates  # Skip forms for non-L2 circuits
        
        # Show forms if we don't have enough service names or need detailed configuration
        return False, [], form_templates  # Show forms
    
    def generate_config_from_form_data(self, form_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate service configurations from filled form data
        """
        try:
            configs = []
            junos_configs = []
            
            for form in form_data:
                # Extract field values from form
                field_values = {}
                for field in form.get("fields", []):
                    field_values[field["field_name"]] = field["value"]
                
                # Generate configuration for this service
                config = self.generate_l2circuit_config_from_form(field_values)
                configs.append(config)
                
                # Generate JUNOS config
                junos_config = self.generate_junos_config_with_openai(config)
                junos_configs.append({
                    "service_name": config.get("instance_id"),
                    "junos_config": junos_config
                })
            
            return {
                "success": True,
                "action_required": "user_confirmation",
                "service_type": form_data[0]["fields"][0]["value"] if form_data else "l2circuit",
                "total_services": len(configs),
                "json_configs": configs,
                "junos_configs": junos_configs,
                "form_data": form_data
            }
            
        except Exception as e:
            logger.error(f"Error generating config from form data: {e}")
            return {
                "success": False,
                "error": str(e),
                "form_data": form_data
            }
    
    def generate_l2circuit_config_from_form(self, field_values: Dict[str, str]) -> Dict[str, Any]:
        """Generate L2 Circuit configuration from form field values"""
        
        # Validate customer
        customer = self._find_customer_by_name(field_values["customer_name"])
        if not customer:
            raise ValueError(f"Customer '{field_values['customer_name']}' not found or not active")
        
        customer_id = customer.get("customer_id")
        
        # Validate nodes
        source_ne_id = self._get_ne_id(field_values["source_node"])
        dest_ne_id = self._get_ne_id(field_values["dest_node"])
        
        if not source_ne_id:
            raise ValueError(f"Source node '{field_values['source_node']}' not found in devices")
        if not dest_ne_id:
            raise ValueError(f"Destination node '{field_values['dest_node']}' not found in devices")
        
        # Generate UUID
        instance_uuid = str(uuid.uuid4())
        service_name = self._sanitize_service_name(field_values["service_name"])
        
        # Load template
        template = self._load_template("l2circuit")
        config = self.service_configs["l2circuit"]
        
        # Prepare template variables using form values
        template_variables = {
            "CUSTOMER_ID": customer_id,
            "DESIGN_ID": config["design_id"],
            "DESIGN_VERSION": config["design_version"],
            "SERVICE_NAME": service_name,
            "INSTANCE_UUID": instance_uuid,
            "CUSTOMER_NAME": field_values["customer_name"],
            "SOURCE_DESCRIPTION": f"{service_name}-{field_values['source_node'].lower().replace('-acx7024-a1', '').replace('-acx7100-a6', '').split('-')[-1] if '-' in field_values['source_node'] else 'acx1'}",
            "SOURCE_NE_ID": source_ne_id,
            "SOURCE_PEER_ADDR": field_values["source_peer_addr"],
            "SOURCE_TRANSPORT_INSTANCE": f"DELAY-to-{field_values['dest_node']}",
            "SOURCE_ACCESS_DESCRIPTION": f"to-{service_name}-ce1",
            "SOURCE_PORT_ID": field_values["source_port_access_id"],
            "SOURCE_PORT": field_values["source_port_id"],
            "SOURCE_NODE": field_values["source_node"],
            "DEST_DESCRIPTION": f"{service_name}-{field_values['dest_node'].lower().replace('-acx7024-a1', '').replace('-acx7100-a6', '').split('-')[-1] if '-' in field_values['dest_node'] else 'acx6'}",
            "DEST_NE_ID": dest_ne_id,
            "DEST_PEER_ADDR": field_values["dest_peer_addr"],
            "DEST_TRANSPORT_INSTANCE": f"DELAY-to-{field_values['source_node']}",
            "DEST_ACCESS_DESCRIPTION": f"to {service_name} ce1",
            "DEST_PORT_ID": field_values["dest_port_access_id"],
            "DEST_PORT": field_values["dest_port_id"],
            "DEST_NODE": field_values["dest_node"],
            "VLAN_ID": field_values["vlan_id"],
            "ORG_ID": self.org_id,
            "WORKFLOW_ID": config["workflow_id"]
        }
        
        # Substitute template variables
        l2circuit_config = self._substitute_template_variables(template, template_variables)
        
        return l2circuit_config

    def process_service_request_with_forms(self, user_query: str) -> Dict[str, Any]:
        """
        Process service request and determine if forms are needed
        """
        results = {
            "original_query": user_query,
            "parsed_request": {},
            "validation_errors": [],
            "needs_form": False,
            "form_templates": [],
            "success": False
        }
        
        try:
            # Step 1: Parse the request using OpenAI
            parsed_request = self.parse_service_request_with_openai(user_query)
            results["parsed_request"] = parsed_request
            
            service_type = parsed_request.get("service_type", "l2circuit")
            
            # Step 2: Validate mandatory fields and generate forms if needed
            is_complete, missing_fields, form_templates = self.validate_mandatory_fields(parsed_request, service_type)
            
            if missing_fields:
                # Basic fields are missing
                results["validation_errors"] = [f"Missing required field: {field}" for field in missing_fields]
                return results
            
            if not is_complete:
                # Need form input (either missing fields or user approval required for L2 circuits)
                results["needs_form"] = True
                results["form_templates"] = form_templates
                results["action_required"] = "form_input"
                results["service_type"] = service_type
                
                # Enhanced message for L2 circuits requiring default value approval
                if service_type == "l2circuit":
                    results["message"] = f"Please review and confirm the configuration for {len(form_templates)} {service_type.upper()} service(s). Default values are pre-populated but can be modified."
                else:
                    results["message"] = f"Please provide detailed configuration for {len(form_templates)} {service_type} service(s)"
                
                return results
            
            # If we reach here, all fields are available and no forms needed - proceed with normal generation
            logger.info("All basic fields provided and no forms required, proceeding directly to config generation")
            
            # Call the normal process_service_request method for direct config generation
            direct_results = self.process_service_request(user_query)
            
            # If direct processing was successful, return those results
            if direct_results.get("success"):
                return direct_results
            else:
                # If direct processing failed, fall back to forms
                results["needs_form"] = True
                results["form_templates"] = form_templates
                results["action_required"] = "form_input"
                results["service_type"] = service_type
                results["message"] = f"Direct config generation failed. Please provide detailed configuration for {len(form_templates)} {service_type} service(s)"
                results["direct_generation_error"] = direct_results.get("error", "Unknown error in direct generation")
                return results
            
        except Exception as e:
            logger.error(f"Error processing service request with forms: {e}")
            results["error"] = str(e)
            return results
    
    # Keep all existing methods from the original class
    def _create_service_directory(self, service_type: str) -> Path:
        """Create directory structure for service type"""
        service_dir = self.services_dir / service_type
        service_dir.mkdir(exist_ok=True)
        return service_dir
    
    def _generate_filename(self, service_type: str, instance_id: str, timestamp: str = None) -> str:
        """Generate filename for service configuration"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # creating filename and cleaning it
        clean_instance_id = re.sub(r'[^a-zA-Z0-9\-]', '', instance_id)
        filename = f"create_{service_type}_{clean_instance_id}_{timestamp}.json"
        return filename
    
    def save_config_to_file(self, config: Dict[str, Any], service_type: str, metadata: Dict[str, Any] = None) -> str:
        """Save service configuration to JSON file"""
        try:
            service_dir = self._create_service_directory(service_type)
            
            # Generate filename
            instance_id = config.get('instance_id', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self._generate_filename(service_type, instance_id, timestamp)
            filepath = service_dir / filename
            
            # Save the configuration directly
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved {service_type} configuration to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")
            return f"Error: {str(e)}"
    
    def save_multiple_configs(self, configs: List[Dict[str, Any]], service_type: str, request_metadata: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Save multiple service configurations to files"""
        results = {
            "saved_files": [],
            "errors": []
        }
        
        for i, config in enumerate(configs):
            try:
                # Save individual config
                filepath = self.save_config_to_file(config, service_type)
                
                if filepath.startswith("Error:"):
                    results["errors"].append(f"Config {i+1}: {filepath}")
                else:
                    results["saved_files"].append(filepath)
                    
            except Exception as e:
                results["errors"].append(f"Config {i+1}: {str(e)}")
        
        return results
    
    def _load_json_file(self, filename: str) -> Dict[str, Any]:
        """Load JSON data from file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {filename}")
            return {}
    
    def _load_template(self, service_type: str) -> Dict[str, Any]:
        """Load service configuration template from file"""
        template_file = self.template_files.get(service_type)
        if not template_file:
            raise ValueError(f"No template file defined for service type: {service_type}")
        
        try:
            with open(template_file, 'r') as f:
                template = json.load(f)
            logger.info(f"Loaded template for {service_type} from {template_file}")
            return template
        except FileNotFoundError:
            logger.error(f"Template file {template_file} not found")
            raise FileNotFoundError(f"Template file {template_file} not found. Please ensure the template exists.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file {template_file}: {e}")
            raise ValueError(f"Invalid JSON in template file {template_file}")
    
    def _substitute_template_variables(self, template: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Replace template variables with actual values"""
        template_str = json.dumps(template)
        
        # Replace all template variables
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            template_str = template_str.replace(placeholder, str(value))
        
        # Check for any remaining unsubstituted variables
        remaining_vars = re.findall(r'\{\{([^}]+)\}\}', template_str)
        if remaining_vars:
            logger.warning(f"Unsubstituted template variables found: {remaining_vars}")
        
        return json.loads(template_str)
    
    def _find_customer_by_name(self, customer_name: str) -> Optional[Dict[str, Any]]:
        """Find customer by name from customers.json"""
        for customer in self.customers_data:
            if (customer.get("name", "").lower() == customer_name.lower() and 
                customer.get("status") == "active"):
                return customer
        return None
    
    def _find_device_by_hostname(self, hostname: str) -> Optional[Dict[str, Any]]:
        """Find device by hostname from devices.json"""
        devices = self.devices_data.get("devices", [])
        for device in devices:
            if device.get("hostname", "").lower() == hostname.lower():
                return device
        return None
    
    def _get_ne_id(self, hostname: str) -> Optional[str]:
        """Get network element ID (sourceId) for a hostname"""
        device = self._find_device_by_hostname(hostname)
        return device.get("sourceId") if device else None
    
    def _generate_random_vlan(self) -> int:
        """Generate a random VLAN ID within the specified range"""
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            vlan_id = random.randint(*self.vlan_range)
            if vlan_id not in self.used_vlans:
                self.used_vlans.add(vlan_id)
                return vlan_id
            attempts += 1
        
        # If we can't find an unused VLAN, just return a random one
        vlan_id = random.randint(*self.vlan_range)
        self.used_vlans.add(vlan_id)
        return vlan_id
    
    def _generate_service_name(self, service_type: str, index: int = 1) -> str:
        """Generate a random service name if not provided"""
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{service_type}{index}-{timestamp}"
    
    def _sanitize_service_name(self, name: str) -> str:
        """Sanitize service name to be API compliant"""
        if not name:
            return self._generate_service_name("service")
        
        # Replace invalid characters with hyphens
        sanitized = name.replace('_', '-')
        sanitized = sanitized.replace(' ', '-')
        
        # Replace special characters with hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', sanitized)
        
        # Ensure it doesn't start or end with hyphen
        sanitized = sanitized.strip('-')
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            sanitized = self._generate_service_name("service")
        
        return sanitized
    
    def parse_service_request_with_openai(self, user_query: str) -> Dict[str, Any]:
        """Parse user query using OpenAI to extract service creation requirements"""
        
        # Get available customers and devices for context
        available_customers = [customer.get("name", "") for customer in self.customers_data if customer.get("status") == "active"]
        available_devices = [device.get("hostname", "") for device in self.devices_data.get("devices", [])]
        
        system_prompt = f"""
        You are a network service configuration parser. Parse the user's request to extract service creation requirements.
        
        Available customers: {', '.join(available_customers)}
        Available devices: {', '.join(available_devices)}
        
        Return a JSON object with the following structure:
        {{
            "service_type": "l2circuit|l3vpn|evpn",
            "quantity": number_of_services,
            "customer_name": "exact_customer_name_from_available_list",
            "source_node": "exact_device_hostname_from_available_list", 
            "dest_node": "exact_device_hostname_from_available_list",
            "service_names": ["list_of_service_names_if_specified"]
        }}
        
        Rules:
        - service_type: default to "l2circuit" unless specifically mentioned (l3vpn, evpn)
        - quantity: extract number or default to 1
        - customer_name: must match exactly from available customers list
        - source_node and dest_node: must match exactly from available devices list
        - service_names: ALWAYS return as an array. If service names are mentioned, include them. If not mentioned, return empty array []
        
        Return only the JSON object, no other text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1
            )
            
            parsed_result = json.loads(response.choices[0].message.content)
            
            # Ensure service_names is always a list
            if "service_names" not in parsed_result:
                parsed_result["service_names"] = []
            elif not isinstance(parsed_result["service_names"], list):
                # If it's a single string, convert to list
                parsed_result["service_names"] = [parsed_result["service_names"]]
            
            logger.info(f"OpenAI parsed request: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error parsing request with OpenAI: {e}")
            return {
                "service_type": "l2circuit",
                "quantity": 1,
                "customer_name": "",
                "source_node": "",
                "dest_node": "",
                "service_names": []
            }
    
    def generate_l2circuit_config(self, 
                                  customer_name: str,
                                  source_node: str,
                                  dest_node: str,
                                  service_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate L2 Circuit configuration JSON using template file"""
        
        # Validate customer
        customer = self._find_customer_by_name(customer_name)
        if not customer:
            raise ValueError(f"Customer '{customer_name}' not found or not active")
        
        customer_id = customer.get("customer_id")
        
        # Validate nodes
        source_ne_id = self._get_ne_id(source_node)
        dest_ne_id = self._get_ne_id(dest_node)
        
        if not source_ne_id:
            raise ValueError(f"Source node '{source_node}' not found in devices")
        if not dest_ne_id:
            raise ValueError(f"Destination node '{dest_node}' not found in devices")
        
        # Generate and sanitize service name
        if not service_name:
            service_name = self._generate_service_name("l2circuit")
        else:
            service_name = self._sanitize_service_name(service_name)
        
        # Generate UUIDs and VLAN
        instance_uuid = str(uuid.uuid4())
        vlan_id = self._generate_random_vlan()
        
        # Load template
        template = self._load_template("l2circuit")
        config = self.service_configs["l2circuit"]
        
        # Use default configurations
        peer_addresses = self.default_configs["l2circuit"]["peer_addresses"]
        port_configs = self.default_configs["l2circuit"]["port_configs"]
        
        # Prepare template variables
        template_variables = {
            "CUSTOMER_ID": customer_id,
            "DESIGN_ID": config["design_id"],
            "DESIGN_VERSION": config["design_version"],
            "SERVICE_NAME": service_name,
            "INSTANCE_UUID": instance_uuid,
            "CUSTOMER_NAME": customer_name,
            "SOURCE_DESCRIPTION": f"{service_name}-{source_node.lower().replace('-acx7024-a1', '').replace('-acx7100-a6', '').split('-')[-1] if '-' in source_node else 'acx1'}",
            "SOURCE_NE_ID": source_ne_id,
            "SOURCE_PEER_ADDR": peer_addresses["source"],
            "SOURCE_TRANSPORT_INSTANCE": f"DELAY-to-{dest_node}",
            "SOURCE_ACCESS_DESCRIPTION": f"to-{service_name}-ce1",
            "SOURCE_PORT_ID": port_configs["source"]["id"],
            "SOURCE_PORT": port_configs["source"]["port_id"],
            "SOURCE_NODE": source_node,
            "DEST_DESCRIPTION": f"{service_name}-{dest_node.lower().replace('-acx7024-a1', '').replace('-acx7100-a6', '').split('-')[-1] if '-' in dest_node else 'acx6'}",
            "DEST_NE_ID": dest_ne_id,
            "DEST_PEER_ADDR": peer_addresses["dest"],
            "DEST_TRANSPORT_INSTANCE": f"DELAY-to-{source_node}",
            "DEST_ACCESS_DESCRIPTION": f"to {service_name} ce1",
            "DEST_PORT_ID": port_configs["dest"]["id"],
            "DEST_PORT": port_configs["dest"]["port_id"],
            "DEST_NODE": dest_node,
            "VLAN_ID": vlan_id,
            "ORG_ID": self.org_id,
            "WORKFLOW_ID": config["workflow_id"]
        }
        
        # Substitute template variables
        l2circuit_config = self._substitute_template_variables(template, template_variables)
        
        return l2circuit_config
    
    def generate_junos_config_with_openai(self, json_config: Dict[str, Any]) -> str:
        """Generate JUNOS configuration from JSON using OpenAI"""
        
        system_prompt = """
        You are a JUNOS network configuration expert. Convert the provided L2 circuit JSON configuration into JUNOS CLI commands.
        
        Based on the JSON configuration, generate the appropriate JUNOS configuration for both nodes including:
        1. Interface configuration with VLAN encapsulation
        2. L2VPN/L2Circuit configuration
        3. LDP signaling configuration
        4. LSP configuration for transport
        
        Return only the JUNOS configuration commands, properly formatted and commented for clarity.
        Use the specific interface, VLAN, and peer information from the JSON.
        """
        
        try:
            # Extract key information from JSON for the prompt
            vpn_service = json_config.get("l2vpn_ntw", {}).get("vpn_services", {}).get("vpn_service", [{}])[0]
            vpn_nodes = vpn_service.get("vpn_nodes", {}).get("vpn_node", [])
            
            context_info = {
                "service_name": json_config.get("instance_id"),
                "vpn_nodes": []
            }
            
            for node in vpn_nodes:
                node_info = {
                    "hostname": node.get("vpn_node_id"),
                    "ne_id": node.get("ne_id"),
                    "interface": node.get("vpn_network_accesses", {}).get("vpn_network_access", [{}])[0].get("port_id"),
                    "vlan_id": node.get("vpn_network_accesses", {}).get("vpn_network_access", [{}])[0].get("connection", {}).get("encapsulation", {}).get("dot1q", {}).get("c_vlan_id"),
                    "peer_addr": node.get("signaling_options", [{}])[0].get("ac_pw_list", [{}])[0].get("peer_addr"),
                    "transport_instance": node.get("underlay_transport", {}).get("transport_instance_id")
                }
                context_info["vpn_nodes"].append(node_info)
            
            user_prompt = f"""
            Convert this L2 circuit JSON configuration to JUNOS CLI commands:
            
            JSON Configuration:
            {json.dumps(json_config, indent=2)}
            
            Key Information:
            - Service Name: {context_info['service_name']}
            - Nodes: {len(context_info['vpn_nodes'])}
            
            Generate complete JUNOS configuration for both nodes.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            junos_config = response.choices[0].message.content
            logger.info("Successfully generated JUNOS configuration with OpenAI")
            return junos_config
            
        except Exception as e:
            logger.error(f"Error generating JUNOS config with OpenAI: {e}")
            return f"Error generating JUNOS configuration: {str(e)}"
    
    def generate_multiple_configs(self, parsed_request: Dict[str, Any], save_to_file: bool = True) -> List[Dict[str, Any]]:
        """Generate multiple service configurations based on parsed request"""
        configs = []
        
        service_type = parsed_request["service_type"]
        quantity = parsed_request["quantity"]
        customer_name = parsed_request["customer_name"]
        source_node = parsed_request["source_node"]
        dest_node = parsed_request["dest_node"]
        service_names = parsed_request.get("service_names", [])
        
        # Validate required fields
        if not customer_name:
            raise ValueError("Customer name is required")
        if not source_node or not dest_node:
            raise ValueError("Source and destination nodes are required")
        
        # Generate configurations
        for i in range(quantity):
            # Use provided service name or generate one
            if service_names and i < len(service_names) and service_names[i].strip():
                service_name = self._sanitize_service_name(service_names[i])
            else:
                service_name = self._generate_service_name(service_type, i + 1)
            
            if service_type == "l2circuit":
                config = self.generate_l2circuit_config(
                    customer_name=customer_name,
                    source_node=source_node,
                    dest_node=dest_node,
                    service_name=service_name
                )
                configs.append(config)
            else:
                # For future implementation of L3VPN and EVPN
                raise NotImplementedError(f"Service type '{service_type}' not yet implemented")
        
        # Save configurations to files if requested
        if save_to_file and configs:
            try:
                # Save all configurations
                save_results = self.save_multiple_configs(configs, service_type)
                
                logger.info(f"Saved {len(save_results['saved_files'])} configuration files")
                if save_results['errors']:
                    logger.warning(f"Errors saving {len(save_results['errors'])} files: {save_results['errors']}")
                        
            except Exception as e:
                logger.error(f"Error saving configurations: {e}")
        
        return configs
    
    def validate_request(self, parsed_request: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parsed service request and return any errors"""
        errors = []
        
        # Check customer
        if not parsed_request["customer_name"]:
            errors.append("Customer name is required")
        elif not self._find_customer_by_name(parsed_request["customer_name"]):
            errors.append(f"Customer '{parsed_request['customer_name']}' not found or not active")
        
        # Check nodes
        if not parsed_request["source_node"]:
            errors.append("Source node is required")
        elif not self._find_device_by_hostname(parsed_request["source_node"]):
            errors.append(f"Source node '{parsed_request['source_node']}' not found")
        
        if not parsed_request["dest_node"]:
            errors.append("Destination node is required")
        elif not self._find_device_by_hostname(parsed_request["dest_node"]):
            errors.append(f"Destination node '{parsed_request['dest_node']}' not found")
        
        # Check quantity
        if parsed_request["quantity"] <= 0:
            errors.append("Quantity must be greater than 0")
        if parsed_request["quantity"] > 10:
            errors.append("Maximum 10 services can be created at once")
        
        # Validate service names if provided
        service_names = parsed_request.get("service_names", [])
        if service_names:
            for i, name in enumerate(service_names):
                if not name or not name.strip():
                    errors.append(f"Service name {i+1} is empty")
                else:
                    # Check for invalid characters
                    if re.search(r'[^a-zA-Z0-9\-]', name):
                        errors.append(f"Service name '{name}' contains invalid characters. Only letters, numbers, and hyphens are allowed.")
                    if name.startswith('-') or name.endswith('-'):
                        errors.append(f"Service name '{name}' cannot start or end with a hyphen")
        
        return len(errors) == 0, errors
    
    def process_service_request(self, user_query: str) -> Dict[str, Any]:
        """
        Complete workflow: Parse query, validate, generate configs, and create JUNOS configs
        """
        results = {
            "original_query": user_query,
            "parsed_request": {},
            "validation_errors": [],
            "json_configs": [],
            "junos_configs": [],
            "saved_files": [],
            "success": False
        }
        
        try:
            # Step 1: Parse the request using OpenAI
            print("🔍 Parsing service request with OpenAI...")
            parsed_request = self.parse_service_request_with_openai(user_query)
            results["parsed_request"] = parsed_request
            print(f"Parsed request: {parsed_request}")
            
            # Step 2: Validate the request
            print("\n✅ Validating request...")
            is_valid, errors = self.validate_request(parsed_request)
            results["validation_errors"] = errors
            
            if not is_valid:
                print(f"❌ Validation failed: {errors}")
                return results
            
            print("✅ Request validation passed")
            
            # Step 3: Generate JSON configurations
            print(f"\n🔧 Generating {parsed_request['quantity']} JSON configuration(s)...")
            configs = self.generate_multiple_configs(parsed_request, save_to_file=True)
            results["json_configs"] = configs
            print(f"✅ Generated {len(configs)} JSON configuration(s)")
            
            # Step 4: Generate JUNOS configurations using OpenAI
            print(f"\n🛠️ Generating JUNOS configurations with OpenAI...")
            for i, config in enumerate(configs):
                print(f"Generating JUNOS config {i+1}/{len(configs)}...")
                junos_config = self.generate_junos_config_with_openai(config)
                results["junos_configs"].append({
                    "service_name": config.get("instance_id"),
                    "junos_config": junos_config
                })
            
            print(f"✅ Generated {len(results['junos_configs'])} JUNOS configuration(s)")
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"Error processing service request: {e}")
            results["error"] = str(e)
            print(f"❌ Error: {e}")
            return results


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        generator = EnhancedServiceConfigGenerator()
            
    except Exception as e:
        print(f"Failed to initialize generator: {e}")