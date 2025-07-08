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
        
        # Fixed organization ID -> Need changes here [MDAVALA] check later - get it from .env 
        self.org_id = "0eaf8613-632d-41d2-8de4-c2d242325d7e"
        
        # VLAN ID range for L2 circuits -> Need changes here [MDAVALA] check later  
        self.vlan_range = (1000, 1100)
        self.used_vlans = set()
        
        # Fixed peer addresses for L2 circuit signaling -> Need changes here [MDAVALA] check later  
        self.peer_addresses = {
            "source": "10.40.40.6",  # First node peer address
            "dest": "10.40.40.1"     # Second node peer address
        }
        
        # Fixed port configurations -> Need changes here [MDAVALA] check later
        self.port_configs = {
            "source": {"port_id": "et-0/0/6", "id": "111"},
            "dest": {"port_id": "et-0/0/8", "id": "111"}
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
        """
        Save service configuration to JSON file
        
        Args:
            config: Service configuration dictionary
            service_type: Type of service (l2circuit, l3vpn, evpn)
            metadata: Additional metadata to include in the file
            
        Returns:
            Path to the saved file
        """
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
        """
        Save multiple service configurations to files
        
        Args:
            configs: List of service configurations
            service_type: Type of service
            request_metadata: Metadata about the original request
            
        Returns:
            Dictionary with saved file paths and any errors
        """
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
        
        # If we can't find an unused VLAN, just return a random one -> Need changes here [MDAVALA] check later
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
        
        # Replace invalid characters with hyphens as routing director not supporting this
        sanitized = name.replace('_', '-')
        sanitized = sanitized.replace(' ', '-')
        
        # Replace special characters with hyphens as routing director not supporting this
        sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', sanitized)
        
        # Ensure it doesn't start or end with hyphen as routing director not supporting this
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
        - service_names: only include if explicitly mentioned in the request
        
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
            logger.info(f"OpenAI parsed request: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error parsing request with OpenAI: {e}")
    
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
            "SOURCE_PEER_ADDR": self.peer_addresses["source"],
            "SOURCE_TRANSPORT_INSTANCE": f"DELAY-to-{dest_node}",
            "SOURCE_ACCESS_DESCRIPTION": f"to-{service_name}-ce1",
            "SOURCE_PORT_ID": self.port_configs["source"]["id"],
            "SOURCE_PORT": self.port_configs["source"]["port_id"],
            "SOURCE_NODE": source_node,
            "DEST_DESCRIPTION": f"{service_name}-{dest_node.lower().replace('-acx7024-a1', '').replace('-acx7100-a6', '').split('-')[-1] if '-' in dest_node else 'acx6'}",
            "DEST_NE_ID": dest_ne_id,
            "DEST_PEER_ADDR": self.peer_addresses["dest"],
            "DEST_TRANSPORT_INSTANCE": f"DELAY-to-{source_node}",
            "DEST_ACCESS_DESCRIPTION": f"to {service_name} ce1",
            "DEST_PORT_ID": self.port_configs["dest"]["id"],
            "DEST_PORT": self.port_configs["dest"]["port_id"],
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
        service_names = parsed_request["service_names"]
        
        # Validate required fields
        if not customer_name:
            raise ValueError("Customer name is required")
        if not source_node or not dest_node:
            raise ValueError("Source and destination nodes are required")
        
        # Generate configurations
        for i in range(quantity):
            # Use provided service name or generate one
            if service_names and i < len(service_names):
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
        
        # Check quantity -> Need changes here [MDAVALA] check later currently restriction of 10 need to test it later
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
        
        Args:
            user_query: Natural language service request
            
        Returns:
            Dictionary containing all results and generated configs
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
            print(" Parsing service request with OpenAI...")
            parsed_request = self.parse_service_request_with_openai(user_query)
            results["parsed_request"] = parsed_request
            print(f"Parsed request: {parsed_request}")
            
            # Step 2: Validate the request
            print("\n Validating request...")
            is_valid, errors = self.validate_request(parsed_request)
            results["validation_errors"] = errors
            
            if not is_valid:
                print(f" Validation failed: {errors}")
                return results
            
            print("Request validation passed")
            
            # Step 3: Generate JSON configurations
            print(f"\n Generating {parsed_request['quantity']} JSON configuration(s)...")
            configs = self.generate_multiple_configs(parsed_request, save_to_file=True)
            results["json_configs"] = configs
            print(f"Generated {len(configs)} JSON configuration(s)")
            
            # Step 4: Generate JUNOS configurations using OpenAI
            print(f"\n Generating JUNOS configurations with OpenAI...")
            for i, config in enumerate(configs):
                print(f"Generating JUNOS config {i+1}/{len(configs)}...")
                junos_config = self.generate_junos_config_with_openai(config)
                results["junos_configs"].append({
                    "service_name": config.get("instance_id"),
                    "junos_config": junos_config
                })
            
            print(f"Generated {len(results['junos_configs'])} JUNOS configuration(s)")
            
            results["success"] = True
            return results
            
        except Exception as e:
            logger.error(f"Error processing service request: {e}")
            results["error"] = str(e)
            print(f"Error: {e}")
            return results


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        generator = EnhancedServiceConfigGenerator()
            
    except Exception as e:
        print(f"Failed to initialize generator: {e}")