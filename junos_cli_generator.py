"""
JUNOS CLI Configuration Generator
Uses OpenAI to convert JSON service configurations to JUNOS CLI commands
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()

class JunosCLIGenerator:
    """Generate JUNOS CLI configurations from service JSON using OpenAI"""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize with OpenAI client"""
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.available = True
            logger.info("JUNOS CLI Generator initialized with OpenAI")
        else:
            self.openai_client = None
            self.available = False
            logger.warning("OpenAI API key not found - CLI generation not available")
    
    def generate_l2circuit_cli(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate JUNOS CLI configuration for L2 Circuit service
        Returns a dictionary with device hostname as key and CLI config as value
        """
        if not self.available:
            return {"error": "OpenAI not available for CLI generation"}
        
        try:
            # Extract service information
            instance_id = config.get('instance_id', 'unknown')
            l2vpn_data = config.get('l2vpn_ntw', {})
            vpn_services = l2vpn_data.get('vpn_services', {})
            vpn_service = vpn_services.get('vpn_service', [{}])[0]
            vpn_nodes = vpn_service.get('vpn_nodes', {}).get('vpn_node', [])
            
            cli_configs = {}
            
            # Generate CLI for each device/node
            for i, node in enumerate(vpn_nodes):
                hostname = node.get('vpn_node_id', f'device_{i+1}')
                
                # Create a simplified JSON structure for this specific node
                node_config = {
                    "service_type": "L2 Circuit",
                    "instance_id": instance_id,
                    "customer": vpn_service.get('customer_name', 'Unknown'),
                    "device_hostname": hostname,
                    "node_config": node
                }
                
                # Generate CLI using OpenAI
                cli_config = self._generate_cli_with_openai(node_config, "l2circuit")
                cli_configs[hostname] = cli_config
            
            return cli_configs
            
        except Exception as e:
            logger.error(f"Error generating L2 Circuit CLI: {e}")
            return {"error": f"Failed to generate CLI: {str(e)}"}
    
    def generate_l3vpn_cli(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate JUNOS CLI configuration for L3VPN service"""
        if not self.available:
            return {"error": "OpenAI not available for CLI generation"}
        
        try:
            instance_id = config.get('instance_id', 'unknown')
            # Extract L3VPN specific configuration
            # This would be similar to L2 circuit but for L3VPN structure
            
            return {"placeholder": f"# L3VPN CLI for {instance_id}\n# Coming soon..."}
            
        except Exception as e:
            logger.error(f"Error generating L3VPN CLI: {e}")
            return {"error": f"Failed to generate CLI: {str(e)}"}
    
    def generate_evpn_cli(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate JUNOS CLI configuration for EVPN service"""
        if not self.available:
            return {"error": "OpenAI not available for CLI generation"}
        
        try:
            instance_id = config.get('instance_id', 'unknown')
            # Extract EVPN specific configuration
            
            return {"placeholder": f"# EVPN CLI for {instance_id}\n# Coming soon..."}
            
        except Exception as e:
            logger.error(f"Error generating EVPN CLI: {e}")
            return {"error": f"Failed to generate CLI: {str(e)}"}
    
    def _generate_cli_with_openai(self, config: Dict[str, Any], service_type: str) -> str:
        """Use OpenAI to convert JSON config to JUNOS CLI"""
        try:
            # Create a detailed prompt for JUNOS CLI generation
            prompt = f"""
                You are an expert JUNOS network engineer. Convert the following service configuration JSON to JUNOS CLI commands.

                Service Type: {service_type.upper()}
                Configuration JSON: {json.dumps(config, indent=2)}

                Generate ONLY the JUNOS CLI configuration commands that would be applied to configure this service on the specified device. Follow these guidelines:

                1. Generate complete, production-ready JUNOS CLI commands
                2. Include proper interface configuration if specified
                3. Include VLAN configuration if applicable
                4. Include L2 circuit or pseudowire configuration for L2 services
                5. Include routing protocol configuration if applicable
                6. Use proper JUNOS syntax and hierarchy
                7. Include commit command at the end
                8. Do NOT include any explanations, just the CLI commands
                9. Start with 'configure' command
                10. Use proper indentation for JUNOS hierarchy

                Format the output as clean JUNOS CLI commands that can be copy-pasted directly into a JUNOS device.

                Example format:
                configure
                set interfaces et-0/0/6 unit 111 description "service-description"
                set interfaces et-0/0/6 unit 111 vlan-id 1000
                ...
                commit
                """
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert JUNOS network engineer who generates clean, production-ready CLI configurations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent, accurate output
                max_tokens=2000
            )
            
            cli_config = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's proper CLI
            if not cli_config.startswith('configure'):
                cli_config = f"configure\n{cli_config}"
            
            if not cli_config.endswith('commit'):
                cli_config = f"{cli_config}\ncommit"
            
            return cli_config
            
        except Exception as e:
            logger.error(f"Error calling OpenAI for CLI generation: {e}")
            return f"# Error generating CLI configuration\n# {str(e)}\n# Please check OpenAI API configuration"
    
    def generate_cli_for_service(self, config: Dict[str, Any], service_type: str) -> Dict[str, str]:
        """
        Main method to generate CLI for any service type
        Returns dictionary with device hostname as key and CLI config as value
        """
        if not self.available:
            return {"error": "CLI generation not available - check OpenAI API key"}
        
        logger.info(f"Generating JUNOS CLI for {service_type} service")
        
        try:
            if service_type.lower() == "l2circuit":
                return self.generate_l2circuit_cli(config)
            elif service_type.lower() == "l3vpn":
                return self.generate_l3vpn_cli(config)
            elif service_type.lower() == "evpn":
                return self.generate_evpn_cli(config)
            else:
                return {"error": f"Unsupported service type: {service_type}"}
                
        except Exception as e:
            logger.error(f"Error in CLI generation: {e}")
            return {"error": f"CLI generation failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Test the CLI generator
    generator = JunosCLIGenerator()