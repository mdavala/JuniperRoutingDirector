#!/usr/bin/env python3

import pandas as pd
from jnpr.junos import Device
from jnpr.junos.exception import ConnectError, RpcError
import warnings
import xml.etree.ElementTree as ET
import xml.dom.minidom

def print_xml_readable(xml_element):
    """Print XML in readable format"""
    try:
        xml_string = ET.tostring(xml_element, encoding='unicode')
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        # Remove empty lines for cleaner output
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        print('\n'.join(lines))
    except Exception as e:
        print(f"Error formatting XML: {e}")
        # Fallback to raw output
        print(ET.tostring(xml_element, encoding='unicode'))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load Excel file with IP addresses and ports
excel_file = 'router_list.xlsx'

try:
    df = pd.read_excel(excel_file, engine='openpyxl')
except FileNotFoundError:
    print(f"Error: Excel file '{excel_file}' not found!")
    print("Please ensure the file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit(1)

# Debug: Print available columns
print("Available columns in Excel file:")
print(df.columns.tolist())
print("\nDataFrame contents:")
print(df.head())

# Clean column names (remove any leading/trailing spaces)
df.columns = df.columns.str.strip()

# Verify we have data
if df.empty:
    print("Error: Excel file is empty!")
    exit(1)

# Device credentials
username = 'jcluser'
password = 'Juniper!1'
base_host = '66.129.234.204'

# Storage for collected data
hardware_data = []
l2vpn_data = []

def get_hardware_info(dev):
    """Extract hardware information from device"""
    try:
        # Get device facts (includes hostname, model, version)
        facts = dev.facts
        hostname = facts['hostname']
        product_model = facts['model']
        junos_version = facts['version']
        
        # Get loopback interface IP
        lo0_ip = None
        #Get Interface configs
        config = dev.rpc.get_config(filter_xml='<configuration><interfaces><interface><name>lo0</name></interface></interfaces></configuration>')
        print_xml_readable(config)
        #lo0_ip = config.find('.//interface[name="lo0"]/unit/family/inet/address/name').text
        lo0_ips = config.findall('.//interface[name="lo0"]/unit/family/inet/address/name')
        ip_list = [ip.text for ip in lo0_ips]
        print(ip_list)
        lo0_ip = ip_list[0]
        
        hardware_info = {
            'hostname': hostname,
            'product-model': product_model,
            'junos-version': junos_version,
            'lo0.0 inet ip': lo0_ip
        }
        
        print(f"Hardware info for {hostname}: Lo0 IP = {lo0_ip}")
        
        # Debug output if lo0_ip is still None
        if not lo0_ip:
            print(f"WARNING: Could not retrieve lo0.0 IP address for {hostname}")
            lo0_ip = "Not found"
        
        hardware_info['lo0.0 inet ip'] = lo0_ip
        return hardware_info
        
    except Exception as e:
        print(f"Error getting hardware info: {e}")
        return None

def get_l2vpn_info(dev, hostname):
    """Extract L2VPN connection information"""
    l2vpn_connections = []
    
    try:
        # Get L2VPN connections
        l2vpn_info = dev.rpc.get_l2vpn_connection_information()
        
        for instance in l2vpn_info.findall('.//instance'):
            instance_name = instance.findtext('instance-name')
            local_site = instance.findtext('local-site-id')
            
            for connection in instance.findall('.//connection'):
                remote_pe = connection.findtext('remote-pe')
                conn_status = connection.findtext('connection-status')
                interface_name = connection.findtext('.//local-interface/interface-name')
                interface_status = connection.findtext('.//local-interface/interface-status')
                
                entry = {
                    'hostname': hostname,
                    'instance-name': instance_name,
                    'Instance Type': None,
                    'local-site': local_site,
                    'connection-status': conn_status,
                    'remote-pe': remote_pe,
                    'interface-name': interface_name,
                    'interface id': None,
                    'unit id': None,
                    'IFD description': None,
                    'Unit Description': None,
                    'interface-status': interface_status,
                    'Route Target': None,
                    'outer-vlan': None,
                    'inner-vlan': None   
                }
                
                # Parse VLAN information if connection is up
                if conn_status == "Up" and interface_name:
                    try:
                        ifd, unit_id = interface_name.split('.')
                        entry['interface id'] = ifd
                        entry['unit id'] = unit_id
                        
                        # Get interface configuration
                        vlan_info = get_vlan_info(dev, ifd, unit_id)
                        entry.update(vlan_info)
                        
                    except Exception as e:
                        print(f"Error parsing VLAN config for {interface_name}: {e}")
                        entry['outer-vlan'] = 'parse error'
                        entry['inner-vlan'] = 'parse error'
                else:
                    entry['outer-vlan'] = 'None'
                    entry['inner-vlan'] = 'None'
                
                l2vpn_connections.append(entry)
        
        # Get routing instance information for route targets
        get_routing_instances(dev, hostname, l2vpn_connections)
        
        return l2vpn_connections
        
    except RpcError as e:
        print(f"Error getting L2VPN info for {hostname}: {e}")
        return []

def get_vlan_info(dev, ifd, unit_id):
    """Get VLAN configuration for specific interface and unit"""
    vlan_info = {
        'IFD description': None,
        'Unit Description': None,
        'outer-vlan': '0',
        'inner-vlan': '0'
    }
    
    try:
        # Get interface configuration
        config = dev.rpc.get_config(filter_xml=f'<configuration><interfaces><interface><name>{ifd}</name></interface></interfaces></configuration>')
        
        for interface in config.findall('.//interface'):
            if interface.findtext('name') == ifd:
                # Get IFD description
                ifd_desc = interface.findtext('description')
                if ifd_desc:
                    vlan_info['IFD description'] = ifd_desc
                    print(f"IFD Description = {ifd_desc}")
                
                # Find the specific unit
                for unit in interface.findall('.//unit'):
                    if unit.findtext('name') == unit_id:
                        # Get unit description
                        unit_desc = unit.findtext('description')
                        if unit_desc:
                            vlan_info['Unit Description'] = unit_desc
                            print(f'Unit: {unit_id} description is {unit_desc}')
                        
                        # Get VLAN tags
                        vlan_tags = unit.find('.//vlan-tags')
                        vlan_id = unit.findtext('.//vlan-id')
                        
                        if vlan_tags is not None:
                            outer = vlan_tags.findtext('outer')
                            inner = vlan_tags.findtext('inner')
                            
                            if outer and inner:
                                vlan_info['outer-vlan'] = outer
                                vlan_info['inner-vlan'] = inner
                            elif outer and not inner:
                                vlan_info['outer-vlan'] = outer
                                vlan_info['inner-vlan'] = '0'
                            else:
                                vlan_info['outer-vlan'] = 'config error'
                                vlan_info['inner-vlan'] = 'config error'
                        elif vlan_id:
                            vlan_info['outer-vlan'] = vlan_id
                            vlan_info['inner-vlan'] = '0'
                        
                        break
                break
                
    except RpcError as e:
        print(f"Error getting VLAN info for {ifd}.{unit_id}: {e}")
        vlan_info['outer-vlan'] = 'error'
        vlan_info['inner-vlan'] = 'error'
    
    return vlan_info

def get_routing_instances(dev, hostname, l2vpn_connections):
    """Get routing instance information and update L2VPN entries with route targets"""
    try:
        # Get routing instances configuration
        config = dev.rpc.get_config(filter_xml='<configuration><routing-instances/></configuration>')
        
        for instance in config.findall('.//instance'):
            name = instance.findtext('name')
            community = instance.findtext('.//community')
            instance_type = instance.findtext('instance-type')
            
            print(f'Instance type = {instance_type}')
            
            # Update corresponding L2VPN entries
            for entry in l2vpn_connections:
                if entry['hostname'] == hostname and entry['instance-name'] == name:
                    entry['Route Target'] = community
                    entry['Instance Type'] = instance_type
                    
    except RpcError as e:
        print(f"Error getting routing instances for {hostname}: {e}")

# Main execution loop
for index, row in df.iterrows():
    # Handle different possible column names for port
    if 'Port' in df.columns:
        ssh_port = row['Port']
    elif 'port' in df.columns:
        ssh_port = row['port']
    else:
        print("Error: No 'Port' or 'port' column found in Excel file")
        print(f"Available columns: {df.columns.tolist()}")
        break
    
    # Handle different possible column names for IP address
    if 'IP Address' in df.columns:
        host_ip = row['IP Address']
    elif 'ip' in df.columns:
        host_ip = row['ip']
    elif 'host' in df.columns:
        host_ip = row['host']
    else:
        # Use default IP if no IP column found
        host_ip = base_host
        print(f"No IP column found, using default: {base_host}")
    
    print(f"\nConnecting to {host_ip} port:{ssh_port}...")
    
    # Create device connection
    dev = Device(
        host=host_ip,
        port=ssh_port,
        user=username,
        password=password,
        auto_probe=10,  # Auto-probe timeout
        normalize=True  # Normalize facts
    )
    
    try:
        # Open connection
        dev.open()
        print(f"Successfully connected to {host_ip}:{ssh_port}")
        
        # Get device hostname for reference
        hostname = dev.facts['hostname']
        
        # Collect hardware information
        print("####### Extracting Hardware Info ######")
        hw_info = get_hardware_info(dev)
        if hw_info:
            hardware_data.append(hw_info)
        
        # Collect L2VPN information
        print("####### Extracting L2VPN Info ######")
        l2vpn_info = get_l2vpn_info(dev, hostname)
        l2vpn_data.extend(l2vpn_info)
        
        print("####### Finished extracting data ######")
        
    except ConnectError as e:
        print(f"Failed to connect to {host_ip}:{ssh_port}: {e}")
        continue
    except Exception as e:
        print(f"Unexpected error with {host_ip}:{ssh_port}: {e}")
        continue
    finally:
        # Always close the connection
        if dev.connected:
            dev.close()
            print(f"Closed connection to {host_ip}:{ssh_port}")

# Save collected data to Excel
try:
    with pd.ExcelWriter('SINET_Inventory.xlsx', engine='openpyxl') as writer:
        # Create Hardware sheet
        if hardware_data:
            pd.DataFrame(hardware_data).to_excel(writer, sheet_name='Hardware', index=False)
            print(f"Saved {len(hardware_data)} hardware records to Hardware sheet")
        
        # Create L2VPN sheet
        if l2vpn_data:
            pd.DataFrame(l2vpn_data).to_excel(writer, sheet_name='L2VPN', index=False)
            print(f"Saved {len(l2vpn_data)} L2VPN records to L2VPN sheet")
    
    print("\nData has been successfully written to SINET_Inventory.xlsx")
    
except Exception as e:
    print(f"Error writing to Excel file: {e}")

print("\nScript execution completed!")