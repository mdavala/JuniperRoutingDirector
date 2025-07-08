import json
import pandas as pd

def parse_evpn_json(json_data):
    """
    Simple function to parse L2VPN EVPN services from JSON data
    Returns: DataFrame with EVPN services and reference data dictionary
    """
    
    def extract_devices(l2vpn_ntw):
        """Extract device ne_id from vpn_nodes"""
        devices = []
        try:
            vpn_services = l2vpn_ntw.get('vpn_services', {})
            for vpn_service in vpn_services.get('vpn_service', []):
                vpn_nodes = vpn_service.get('vpn_nodes', {})
                for node in vpn_nodes.get('vpn_node', []):
                    ne_id = node.get('ne_id', '')
                    if ne_id:
                        devices.append(ne_id)
        except:
            pass
        return devices
    
    def get_customer_name(l2vpn_ntw):
        """Get customer name from l2vpn_ntw"""
        try:
            vpn_services = l2vpn_ntw.get('vpn_services', {})
            vpn_service_list = vpn_services.get('vpn_service', [])
            if vpn_service_list:
                return vpn_service_list[0].get('customer_name', '')
        except:
            pass
        return ''
    
    def extract_order_status_data(item):
        """Extract order status and workflow information"""
        status_info = {
            'order_status': 'N/A',
            'components_status': 'N/A',
            'workflow_status': 'N/A'
        }
        
        try:
            order_status = item.get('order_status', {})
            
            # Get overall status
            status_info['order_status'] = order_status.get('status', 'N/A')
            
            # Get component statuses
            components = order_status.get('components', [])
            component_statuses = []
            
            for component in components:
                comp_type = component.get('component_type', '')
                comp_data = component.get('component_data', [])
                
                if comp_data:
                    # Get latest status from component data
                    latest_status = comp_data[-1].get('status', '')
                    if comp_type and latest_status:
                        component_statuses.append(f"{comp_type}:{latest_status}")
            
            status_info['components_status'] = ' | '.join(component_statuses) if component_statuses else 'No component data'
            
            # Get workflow trace summary
            workflow_trace = order_status.get('workflow_trace', [])
            if workflow_trace:
                successful_tasks = sum(1 for task in workflow_trace if task.get('status') == 'success')
                total_tasks = len(workflow_trace)
                status_info['workflow_status'] = f"{successful_tasks}/{total_tasks} tasks successful"
            else:
                status_info['workflow_status'] = 'No workflow data'
                
        except Exception as e:
            print(f"Error extracting order status data: {e}")
        
        return status_info
    
    # Handle both single object and list
    data_list = json_data if isinstance(json_data, list) else [json_data]
    
    services = []
    reference_data = {}
    
    for item in data_list:
        # Only process L2VPN EVPN services
        if item.get('design_id') == 'elan-evpn-csm':
            
            # Extract required fields
            service_name = item.get('instance_id', '')
            customer_id = item.get('customer_id', '')
            status = item.get('instance_status', '')
            instance_id = item.get('instance_uuid', '')
            
            # Extract from l2vpn_ntw
            l2vpn_ntw = item.get('l2vpn_ntw', {})
            customer_name = get_customer_name(l2vpn_ntw)
            devices = extract_devices(l2vpn_ntw)
            
            # Extract order status data
            status_data = extract_order_status_data(item)
            
            # Create service record
            service_record = {
                'L2VPN EVPN Service Name': service_name,
                'Customer ID': customer_id,
                'Customer Name': customer_name,
                'Status': status,
                'Instance ID': instance_id,
                'Devices': ' → '.join(devices),
                'Device Count': len(devices),
                'Order Status': status_data['order_status'],
                'Components Status': status_data['components_status'],
                'Workflow Status': status_data['workflow_status']
            }
            
            services.append(service_record)
            
            # Store reference data
            reference_data[service_name] = {
                'l2vpn_ntw': l2vpn_ntw,
                'l2vpn_svc': item.get('l2vpn_svc', {}),
                'order_status': item.get('order_status', {}),
                'status_data': status_data
            }
    
    # Create DataFrame
    df = pd.DataFrame(services)
    
    return df, reference_data

# Quick usage example
if __name__ == "__main__":
    # Load your JSON file
    with open('services/get_instances.json', 'r') as file:
        data = json.load(file)
    
    # Parse L2VPN EVPN services
    services_df, ref_data = parse_evpn_json(data)
    print("*"*60)
    print(services_df)
    print("*"*60)
    
    # Display results
    print("=== L2VPN EVPN Services ===")
    if not services_df.empty:
        print(services_df.to_string(index=False))
        print(f"\nTotal Services: {len(services_df)}")

        services_df.to_csv("l2vpn_evpn_services.csv", index=False)
        print("✅ Services saved to l2vpn_evpn_services.csv")

        # Example: Access reference data for LLM
        for service_name in services_df['L2VPN EVPN Service Name']:
            #print(f"\nReference data available for: {service_name}")
            if service_name in ref_data:
                l2vpn_ntw = ref_data[service_name]['l2vpn_ntw']
                l2vpn_svc = ref_data[service_name]['l2vpn_svc']
                order_status = ref_data[service_name]['order_status']
    else:
        print("No L2VPN EVPN services found")