from opensearchpy import OpenSearch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenSearch connection details from environment variables
host = os.getenv('OPENSEARCH_HOST', 'localhost')
port = int(os.getenv('OPENSEARCH_PORT', 9200))
username = os.getenv('OPENSEARCH_USERNAME', 'admin')
password = os.getenv('OPENSEARCH_PASSWORD', 'admin')

# Create OpenSearch client
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=(username, password),
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

try:
    # Test the connection
    response = client.info()
    print("Successfully connected to OpenSearch!")
    print("OpenSearch version:", response['version']['number'])
except Exception as e:
    print("Failed to connect to OpenSearch:")
    print(str(e)) 