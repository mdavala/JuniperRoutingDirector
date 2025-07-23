#!/usr/bin/env python3

import pandas as pd
import json
from anthropic import Anthropic
import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
class EnhancedVPNServiceAnalyzer:
    def __init__(self, api_key: str = None, use_llm: bool = True):
        """
        Enhanced VPN Service Analyzer supporting l2vpn, l2circuit, l3vpn, evpn elan, evpn vpws
        
        Args:
            api_key: Anthropic API key (optional)
            use_llm: Whether to use LLM for summary generation (default: True)
        """
        self.use_llm = use_llm
        self.client = None
        
        if use_llm:
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if self.api_key:
                try:
                    self.client = Anthropic(api_key=self.api_key)
                    print("✅ LLM enabled")
                except Exception as e:
                    print(f"⚠️  LLM initialization failed, falling back to manual generation: {e}")
                    self.use_llm = False
            else:
                print("⚠️  No API key provided, using manual summary generation")
                self.use_llm = False
        else:
            print("📝 Using manual summary generation (LLM disabled)")
        
        self.hardware_data = None
        self.vpn_data = None
        self.device_lookup = {}
        
        # Service type mappings
        self.service_type_mapping = {
            'l2vpn': 'L2VPN',
            'l2circuit': 'L2Circuit', 
            'l3vpn': 'L3VPN',
            'evpn elan': 'EVPN ELAN',
            'evpn vpws': 'EVPN VPWS'
        }
    
    def load_excel_data(self, excel_file: str, vpn_sheet_name: str = 'L2VPN'):
        """Load data from Excel file"""
        try:
            # Load Hardware sheet
            self.hardware_data = pd.read_excel(excel_file, sheet_name='Hardware', engine='openpyxl')
            print(f"✅ Loaded {len(self.hardware_data)} hardware records")
            
            # Load VPN sheet (could be L2VPN, L3VPN, etc.)
            self.vpn_data = pd.read_excel(excel_file, sheet_name=vpn_sheet_name, engine='openpyxl')
            print(f"✅ Loaded {len(self.vpn_data)} VPN records from {vpn_sheet_name} sheet")
            
            # Create device lookup (IP to hostname)
            for _, device in self.hardware_data.iterrows():
                if pd.notna(device['lo0.0 inet ip']):
                    ip = device['lo0.0 inet ip'].split('/')[0]
                    self.device_lookup[ip] = device['hostname']
            
            print(f"✅ Created device lookup for {len(self.device_lookup)} devices")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return False
    
    def parse_vlan_info(self, outer_vlan: str, inner_vlan: str) -> Dict[str, str]:
        """Parse VLAN information to determine CVLAN/SVLAN"""
        vlan_info = {'cvlan': None, 'svlan': None, 'vlan_type': 'untagged'}
        
        if pd.isna(outer_vlan) or outer_vlan in ['0', 'None', '']:
            vlan_info['vlan_type'] = 'untagged'
            return vlan_info
        
        # Parse outer VLAN
        if '0x9100' in str(outer_vlan):
            # S-VLAN (Service VLAN)
            svlan_match = re.search(r'0x9100\.(\d+)', str(outer_vlan))
            if svlan_match:
                vlan_info['svlan'] = svlan_match.group(1)
                vlan_info['vlan_type'] = 'svlan'
        else:
            # Regular VLAN (Customer VLAN)
            try:
                vlan_num = int(float(str(outer_vlan)))
                if vlan_num > 0:
                    vlan_info['cvlan'] = str(vlan_num)
                    vlan_info['vlan_type'] = 'cvlan'
            except:
                pass
        
        # Parse inner VLAN if present
        if pd.notna(inner_vlan) and inner_vlan not in ['0', 'None', '']:
            if '0x9100' in str(inner_vlan):
                cvlan_match = re.search(r'0x9100\.(\d+)', str(inner_vlan))
                if cvlan_match:
                    vlan_info['cvlan'] = cvlan_match.group(1)
                    if vlan_info['svlan']:
                        vlan_info['vlan_type'] = 'double_tagged'
        
        return vlan_info
    
    def analyze_service_topology(self, service_connections: List[Dict]) -> Tuple[str, Dict]:
        """Analyze service topology to determine if it's hub-and-spoke or any-to-any"""
        
        # Count connections per device
        device_connection_count = defaultdict(int)
        unique_devices = set()
        
        for conn in service_connections:
            device_connection_count[conn['local_device']] += 1
            unique_devices.add(conn['local_device'])
            if conn['remote_device'] != 'Unknown':
                unique_devices.add(conn['remote_device'])
        
        total_devices = len(unique_devices)
        max_connections = max(device_connection_count.values()) if device_connection_count else 0
        
        # Determine topology
        if total_devices <= 2:
            topology_type = "point-to-point"
            hub_device = None
        elif max_connections >= (total_devices - 1):
            # One device connects to most others = hub-and-spoke
            topology_type = "hub-and-spoke"
            hub_device = max(device_connection_count, key=device_connection_count.get)
        else:
            topology_type = "any-to-any"
            hub_device = None
        
        topology_info = {
            'type': topology_type,
            'hub_device': hub_device,
            'total_devices': total_devices,
            'device_list': list(unique_devices)
        }
        
        return topology_type, topology_info
    
    def generate_service_summary_with_llm(self, service_data: Dict) -> str:
        """Use LLM to generate concise service summary"""
        
        service_type = service_data['service_type']
        customer = service_data.get('customer', 'Unknown')
        topology_info = service_data['topology_info']
        
        prompt = f"""
        Generate a single-line summary for this VPN service based on the following data:

        Service Details:
        - Service Name: {service_data['service_name']}
        - Service Type: {service_type}
        - Customer: {customer}
        - Route Target: {service_data['route_target']}
        - Topology: {topology_info}
        - Devices: {service_data['devices']}
        - VLAN Information: {service_data['vlan_details']}

        Generate EXACTLY ONE LINE following these patterns:

        For point-to-point:
        "[SERVICE_TYPE] service for customer [CUSTOMER] configured between [DEVICE1] and [DEVICE2]"

        For hub-and-spoke:
        "[SERVICE_TYPE] service for customer [CUSTOMER] configured between [HUB_DEVICE] as HUB, [SPOKE_DEVICES] are spokes"

        For any-to-any:
        "[SERVICE_TYPE] service for customer [CUSTOMER] configured between [ALL_DEVICES] as any-to-any service type"

        Include VLAN details if significant:
        "[SERVICE_TYPE] service for customer [CUSTOMER] configured between [DEVICE1] with cvlan [VLAN1], [DEVICE2] with cvlan [VLAN2] as [TOPOLOGY] service type"

        Rules:
        1. Use EXACT service type: {service_type}
        2. Use EXACT customer name: {customer}
        3. Keep it to ONE line only
        4. Include topology type for multi-point services
        5. Include VLAN info only if VLANs are configured and different per device
        6. Use "configured between" format consistently
        7. Replace [PLACEHOLDERS] with actual values from the data
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            # Ensure it's just one line
            return summary.split('\n')[0].strip()
            
        except Exception as e:
            # Fallback to manual generation
            return self.generate_manual_summary(service_data)
    
    def generate_manual_summary(self, service_data: Dict) -> str:
        """Fallback manual summary generation"""
        service_type = service_data['service_type']
        customer = service_data.get('customer', 'Unknown')
        topology = service_data['topology_info']
        
        if topology['type'] == 'point-to-point':
            devices = topology['device_list']
            if len(devices) >= 2:
                return f"{service_type} service for customer {customer} configured between {devices[0]} and {devices[1]}"
        
        elif topology['type'] == 'hub-and-spoke':
            hub = topology['hub_device']
            spokes = [d for d in topology['device_list'] if d != hub]
            spokes_str = ', '.join(spokes)
            return f"{service_type} service for customer {customer} configured between {hub} as HUB, {spokes_str} are spokes"
        
        elif topology['type'] == 'any-to-any':
            devices_str = ', '.join(topology['device_list'])
            return f"{service_type} service for customer {customer} configured between {devices_str} as any-to-any service type"
        
        return f"{service_type} service for customer {customer} - configuration details incomplete"
    
    def analyze_all_services(self) -> List[str]:
        """Analyze all VPN services and generate single-line summaries"""
        
        if self.vpn_data is None:
            return ["❌ No VPN data loaded"]
        
        # Group by service instance
        service_groups = defaultdict(list)
        
        for _, record in self.vpn_data.iterrows():
            if record.get('connection-status') == 'Up' and pd.notna(record.get('remote-pe')):
                instance_name = record.get('instance-name', 'Unknown')
                
                # Get remote device name
                remote_hostname = self.device_lookup.get(
                    record.get('remote-pe', ''), 
                    f"Unknown({record.get('remote-pe', 'N/A')})"
                )
                
                # Parse VLAN information
                vlan_info = self.parse_vlan_info(
                    record.get('outer-vlan', ''), 
                    record.get('inner-vlan', '')
                )
                
                service_groups[instance_name].append({
                    'local_device': record.get('hostname', 'N/A'),
                    'remote_device': remote_hostname,
                    'interface': record.get('interface-name', 'N/A'),
                    'instance_type': record.get('Instance Type', 'l2vpn'),
                    'route_target': record.get('Route Target', 'N/A'),
                    'customer': record.get('customer', record.get('Customer', 'SINET')),  # Default to SINET
                    'vlan_info': vlan_info
                })
        
        # Generate summaries for each service
        summaries = []
        
        for service_name, connections in service_groups.items():
            if not connections:
                continue
            
            try:
                # Determine service type
                instance_types = [conn['instance_type'].lower() for conn in connections]
                most_common_type = Counter(instance_types).most_common(1)[0][0]
                service_type = self.service_type_mapping.get(most_common_type, most_common_type.upper())
                
                # Analyze topology
                topology_type, topology_info = self.analyze_service_topology(connections)
                
                # Get customer (use first non-Unknown customer)
                customers = [conn['customer'] for conn in connections if conn['customer'] not in ['Unknown', 'N/A', '']]
                customer = customers[0] if customers else 'SINET'
                
                # Get route target
                route_targets = [conn['route_target'] for conn in connections if conn['route_target'] != 'N/A']
                route_target = route_targets[0] if route_targets else 'N/A'
                
                # Collect VLAN details
                vlan_details = {}
                for conn in connections:
                    device = conn['local_device']
                    vlan_info = conn['vlan_info']
                    if vlan_info['cvlan']:
                        vlan_details[device] = f"cvlan {vlan_info['cvlan']}"
                    elif vlan_info['svlan']:
                        vlan_details[device] = f"svlan {vlan_info['svlan']}"
                
                # Prepare service data for LLM
                service_data = {
                    'service_name': service_name,
                    'service_type': service_type,
                    'customer': customer,
                    'route_target': route_target,
                    'topology_info': topology_info,
                    'devices': connections,
                    'vlan_details': vlan_details
                }
                
                # Generate summary - try LLM first if available, fallback to manual
                if self.use_llm and self.client:
                    try:
                        summary = self.generate_service_summary_with_llm(service_data)
                    except Exception as llm_error:
                        print(f"⚠️  LLM failed for {service_name}, using manual generation: {llm_error}")
                        summary = self.generate_manual_summary(service_data)
                else:
                    summary = self.generate_manual_summary(service_data)
                
                summaries.append(f"{len(summaries) + 1}. {summary}")
                
            except Exception as e:
                print(f"❌ Error processing service {service_name}: {e}")
                summaries.append(f"{len(summaries) + 1}. Error processing service {service_name}")
        
        return summaries
    
    def run_analysis(self, excel_file: str, vpn_sheet_name: str = 'L2VPN') -> List[str]:
        """Run complete VPN service analysis"""
        
        print("🚀 Starting Enhanced VPN Service Analysis...")
        
        # Load data
        if not self.load_excel_data(excel_file, vpn_sheet_name):
            return ["❌ Failed to load Excel data"]
        
        # Analyze services
        print("🔍 Analyzing VPN services...")
        summaries = self.analyze_all_services()
        
        print(f"✅ Generated {len(summaries)} service summaries")
        return summaries

# Convenience functions
def analyze_vpn_services(excel_file: str, vpn_sheet_name: str = 'L2VPN', api_key: str = None, use_llm: bool = True) -> List[str]:
    """Quick VPN service analysis"""
    analyzer = EnhancedVPNServiceAnalyzer(api_key=api_key, use_llm=use_llm)
    return analyzer.run_analysis(excel_file, vpn_sheet_name)

def print_service_summaries(excel_file: str, vpn_sheet_name: str = 'L2VPN', api_key: str = None, use_llm: bool = True):
    """Print VPN service summaries to console"""
    summaries = analyze_vpn_services(excel_file, vpn_sheet_name, api_key, use_llm)
    
    print("\n" + "="*80)
    print("📋 VPN SERVICE SUMMARIES")
    print("="*80)
    
    for summary in summaries:
        print(summary)
    
    print("="*80)

# Example usage
def main():
    """Example usage"""
    
    excel_file = 'SINET_Inventory.xlsx'
    
    # Analyze L2VPN services
    print("🔍 Analyzing L2VPN Services...")
    
    # Try with LLM first, fallback to manual if no API key
    try:
        l2vpn_summaries = analyze_vpn_services(excel_file, 'L2VPN', use_llm=True)
    except Exception as e:
        print(f"⚠️  Falling back to manual generation: {e}")
        l2vpn_summaries = analyze_vpn_services(excel_file, 'L2VPN', use_llm=False)
    
    print("\n📋 L2VPN SERVICE SUMMARIES:")
    for summary in l2vpn_summaries:
        print(summary)
    
    # If you have other sheets (L3VPN, EVPN, etc.), analyze them too
    # l3vpn_summaries = analyze_vpn_services(excel_file, 'L3VPN', use_llm=False)
    # evpn_summaries = analyze_vpn_services(excel_file, 'EVPN', use_llm=False)

def test_without_llm():
    """Test function that works without LLM/API key"""
    excel_file = 'SINET_Inventory.xlsx'
    print("🧪 Testing without LLM...")
    summaries = analyze_vpn_services(excel_file, 'L2VPN', use_llm=False)
    
    print("\n📋 SERVICE SUMMARIES (Manual Generation):")
    for summary in summaries:
        print(summary)

if __name__ == "__main__":
    
    main()

# Usage Examples:

# 1. Quick analysis with LLM:
# summaries = analyze_vpn_services('SINET_Inventory.xlsx', 'L2VPN')

# 2. Without LLM (no API key required):
# summaries = analyze_vpn_services('SINET_Inventory.xlsx', 'L2VPN', use_llm=False)

# 3. Print to console:
# print_service_summaries('SINET_Inventory.xlsx', 'L2VPN', use_llm=False)

# 4. Analyze different service types:
# l2vpn_summaries = analyze_vpn_services('SINET_Inventory.xlsx', 'L2VPN', use_llm=False)
# l3vpn_summaries = analyze_vpn_services('SINET_Inventory.xlsx', 'L3VPN', use_llm=False)

# 5. With explicit API key:
# analyzer = EnhancedVPNServiceAnalyzer(api_key="your-api-key", use_llm=True)
# summaries = analyzer.run_analysis('SINET_Inventory.xlsx', 'L2VPN')