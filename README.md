# Routing Director MCP Server

A comprehensive Model Context Protocol (MCP) Server and Client for Juniper's Routing Director with GPT-4 powered intelligent service management.

## 🚀 Overview

This project provides an intelligent interface to Juniper's Routing Director through MCP (Model Context Protocol), enabling natural language service creation, deletion, and management with GPT-4 powered analysis. The system includes both a server component and a user-friendly Streamlit web client.

## ✨ Key Features

### 🤖 GPT-4 Powered Intelligence
- **Natural Language Processing**: Create and delete services using simple English commands
- **Intelligent Query Analysis**: Automatic detection of service types and required operations
- **Smart Service Routing**: GPT-4 determines the best tool for each request

### 🛠️ Service Management
- **L2 Circuit Services**: Full lifecycle management with interactive forms
- **L3VPN Services**: Viewing and management (creation/deletion coming soon)
- **EVPN Services**: Viewing and management (creation/deletion coming soon)

### 🌐 Interactive Web Interface
- **Streamlit Chat Interface**: Natural conversation flow for service management
- **Interactive Forms**: User-friendly forms with default values for service configuration
- **Real-time Status**: Monitor service creation and deletion progress
- **Configuration Preview**: Review JSON configs and JUNOS CLI before deployment

### 🔧 Advanced Features
- **2-Step Workflow**: Upload → Deploy for service creation
- **3-Step Deletion**: Modify → Create Order → Execute for service deletion
- **Bulk Operations**: Handle multiple services efficiently
- **Data Visualization**: Enhanced tables with quality analysis and export options
- **CLI Generation**: Automatic JUNOS CLI configuration generation

## 📋 Prerequisites

- Python 3.8 or higher
- Access to Juniper Routing Director
- OpenAI API key (for GPT-4 functionality)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create .env file**
   Create a `.env` file in the root directory with your credentials:
   ```env
   USERNAME=your_routing_director_username
   PASSWORD=your_routing_director_password
   ORG_ID=your_organization_id
   OPENAI_API_KEY=your_openai_api_key
   ```

## 🚀 Quick Start

### Starting the Web Client

```bash
streamlit run mcpClient.py
```

The web interface will open at `http://localhost:8501`

### Basic Usage

1. **Connect to MCP Server**: Click "🔌 Connect" in the web interface
2. **Natural Language Commands**: Type commands like:
   - `"Create 2 L2 circuits from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET"`
   - `"Delete l2circuit1-135006"`
   - `"Show me all L3VPN services"`

## 📚 Usage Examples

### Service Creation Examples

```
Create an L2 circuit for customer SINET
Create L2 circuit from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET with service name test-l2ckt
Create 3 L2 circuits from PNH-ACX7024-A1 to TH-ACX7100-A6 for customer SINET
```

### Service Deletion Examples

```
Delete l2circuit1-135006
Remove the L2 circuit service named test-circuit
Terminate service l2circuit1-135006
```

### Service Viewing Examples

```
Show me all L3VPN services
Display L2 circuit instances
What EVPN services are running?
Show me the order history
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `USERNAME` | Routing Director username | `abc@juniper.net` |
| `PASSWORD` | Routing Director password | `xyz` |
| `ORG_ID` | Organization ID in Routing Director | `123456789` |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | `sk-proj-eaVqS7tRMA...` |

### Server Configuration

The MCP server runs on Routing Director API at `https://x.x.x.x:48800` by default. Modify the `BASE_URL` in `mcpServer.py` if needed.

## 🏗️ Architecture

### Components

1. **mcpServer.py**: Main MCP server with GPT-4 integration
2. **mcpClient.py**: Streamlit web client
3. **l3vpn_parser.py**: L3VPN service parser
4. **l2ckt_parser.py**: L2 Circuit service parser
5. **l2vpn_evpn_parser.py**: EVPN service parser
6. **services_generator.py**: Automatic service configuration generator
7. **junos_cli_generator.py**: JUNOS CLI configuration generator

### Workflow

1. **Query Analysis**: GPT-4 analyzes natural language input
2. **Tool Selection**: System determines appropriate action
3. **Form Generation**: Interactive forms for missing details
4. **Configuration**: JSON and CLI configurations generated
5. **Confirmation**: User reviews configurations
6. **Execution**: 2-step deployment workflow
7. **Monitoring**: Real-time status updates

## 🛡️ Security

- Credentials stored in `.env` file (not in version control)
- Basic authentication with Routing Director
- HTTPS communication with API endpoints
- Input validation and error handling

## 🐛 Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify `.env` credentials
   - Check network connectivity to Routing Director
   - Ensure Routing Director is accessible

2. **GPT-4 Not Available**
   - Verify `OPENAI_API_KEY` in `.env`
   - Check OpenAI API quota and billing

3. **Service Creation Failed**
   - Verify customer exists in Routing Director
   - Check device names in inventory
   - Review service configuration parameters

### Debug Mode

Enable debug mode in the web interface to see:
- Session state information
- API request/response details
- Service generator status

## 📄 API Endpoints

The server provides access to these Routing Director endpoints:

- `GET /order/instances` - Service instances
- `GET /order/orders` - Service orders
- `GET /order/customers` - Customer information
- `POST /order` - Create service
- `POST /order/.../exec` - Execute service


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs in the web interface
3. Use debug mode for detailed information
4. Create an issue in the repository

## 🔄 Version History

- **v1.0.0**: Initial release with GPT-4 integration
- Support for L2 Circuit creation and deletion
- Interactive web interface
- Natural language processing

---

**Note**: Remember to keep your `.env` file secure and never commit it to version control. The `.env` file should contain your actual credentials for Routing Director and OpenAI API access.
