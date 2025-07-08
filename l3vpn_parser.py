import json
import pandas as pd

def parse_l3vpn_json(json_data):
    """
    Simple function to parse L3VPN services from JSON data
    Returns: DataFrame with L3VPN services and reference data dictionary
    """
    
    def extract_devices(l3vpn_ntw):
        """Extract device ne_id and site_id pairs"""
        devices = []
        try:
            vpn_services = l3vpn_ntw.get('vpn_services', {})
            for vpn_service in vpn_services.get('vpn_service', []):
                vpn_nodes = vpn_service.get('vpn_nodes', {})
                for node in vpn_nodes.get('vpn_node', []):
                    ne_id = node.get('ne_id', '')
                    site_id = node.get('site_id', '')
                    if ne_id and site_id:
                        devices.append(f"{ne_id}({site_id})")
        except:
            pass
        return devices
    
    def get_customer_name(l3vpn_ntw):
        """Get customer name from l3vpn_ntw"""
        try:
            vpn_services = l3vpn_ntw.get('vpn_services', {})
            vpn_service_list = vpn_services.get('vpn_service', [])
            if vpn_service_list:
                return vpn_service_list[0].get('customer_name', '')
        except:
            pass
        return ''
    
    def extract_assurance_data(item):
        """Extract active assurance test result data"""
        assurance_info = {
            'summary': 'N/A',
            'assurance_status': 'No data',
            'test_results': 'No data',
            'test_ids': 'No data'
        }
        
        try:
            active_assurance = item.get('active_assurance_test_result', {})
            
            # Get overall summary
            assurance_info['summary'] = active_assurance.get('summary', 'N/A')
            
            # Process nodes
            nodes = active_assurance.get('nodes', [])
            
            node_statuses = []
            test_results = []
            test_ids = []
            
            for node in nodes:
                device_id = node.get('device', '')
                node_status = node.get('status', '')
                
                # Store node status
                if node_status:
                    node_statuses.append(f"{device_id[:12]}...:{node_status}")
                
                # Process test results for this node
                test_result_list = node.get('test_results', [])
                for test_result in test_result_list:
                    test_status = test_result.get('status', '')
                    test_id = test_result.get('test_id', '')
                    
                    if test_status:
                        test_results.append(f"{device_id[:12]}...:{test_status}")
                    
                    if test_id:
                        test_ids.append(f"{device_id[:12]}...:{test_id[:8]}...")
            
            # Create display strings
            assurance_info['assurance_status'] = ' | '.join(node_statuses) if node_statuses else 'No status data'
            assurance_info['test_results'] = ' | '.join(test_results) if test_results else 'No test results'
            assurance_info['test_ids'] = ' | '.join(test_ids) if test_ids else 'No test IDs'
            
        except Exception as e:
            print(f"Error extracting assurance data: {e}")
        
        return assurance_info
    
    # Handle both single object and list
    data_list = json_data if isinstance(json_data, list) else [json_data]
    
    services = []
    reference_data = {}
    
    for item in data_list:
        # Only process L3VPN services
        if item.get('design_id') == 'l3vpn':
            
            # Extract required fields
            service_name = item.get('instance_id', '')
            customer_id = item.get('customer_id', '')
            status = item.get('instance_status', '')
            instance_id = item.get('instance_uuid', '')
            
            # Extract from l3vpn_ntw
            l3vpn_ntw = item.get('l3vpn_ntw', {})
            customer_name = get_customer_name(l3vpn_ntw)
            devices = extract_devices(l3vpn_ntw)
            
            # Extract assurance data
            assurance_data = extract_assurance_data(item)
            
            # Create service record
            service_record = {
                'L3VPN Service Name': service_name,
                'Customer ID': customer_id,
                'Customer Name': customer_name,
                'Status': status,
                'Instance ID': instance_id,
                'Devices': ' <-> '.join(devices),
                'Device Count': len(devices),
                'Assurance Summary': assurance_data['summary'],
                'Assurance Status': assurance_data['assurance_status'],
                'Test Results': assurance_data['test_results'],
                'Test IDs': assurance_data['test_ids']
            }
            
            services.append(service_record)
            
            # Store reference data
            reference_data[service_name] = {
                'l3vpn_ntw': l3vpn_ntw,
                'l3vpn_svc': item.get('l3vpn_svc', {}),
                'active_assurance_test_result': item.get('active_assurance_test_result', {}),
                'assurance_data': assurance_data
            }
    
    # Create DataFrame
    df = pd.DataFrame(services)

    
    return df, reference_data

# Quick usage example
if __name__ == "__main__":
    # Load your JSON file
    with open('services/get_instances.json', 'r') as file:
        data = json.load(file)
    
    # Parse L3VPN services
    services_df, ref_data = parse_l3vpn_json(data)
    print("*"*60)
    print(services_df)
    print("*"*60)
    
    # Display results
    print("=== L3VPN Services ===")
    if not services_df.empty:
        print(services_df.to_string(index=False))
        print(f"\nTotal Services: {len(services_df)}")

        services_df.to_csv("l3vpn_services.csv", index=False)
        print("✅ Services saved to vpn_services.csv")

        # Example: Access reference data for LLM
        for service_name in services_df['L3VPN Service Name']:
            #print(f"\nReference data available for: {service_name}")
            if service_name in ref_data:
                l3vpn_ntw = ref_data[service_name]['l3vpn_ntw']
                l3vpn_svc = ref_data[service_name]['l3vpn_svc']
                assurance_result = ref_data[service_name]['active_assurance_test_result']
    else:
        print("No L3VPN services found")