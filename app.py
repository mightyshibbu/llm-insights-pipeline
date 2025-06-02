import os
os.environ["STREAMLIT_WATCHER_PATCH_MODULES"] = "false"

# Add torch import fix
try:
    import torch
    torch.classes = None  # Prevent Streamlit from inspecting torch C++ extensions
except ImportError:
    pass  # Ignore if torch is not installed

import streamlit as st
import psutil
import socket
import sys
import time
from pathlib import Path
import logging
import re
import anthropic  # Add at the top with other imports

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Add process lock mechanism
def is_process_running(port=8502):
    """Check if another instance is running on the same port"""
    try:
        # Try to create a socket on the Streamlit port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result == 0:
            logger.info(f"Port {port} is in use")
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking port: {str(e)}")
        return False

def kill_existing_processes():
    """Kill any existing Python processes running this app"""
    current_pid = os.getpid()
    killed_processes = []
    
    # First, try to find and kill Streamlit processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip current process
            if proc.pid == current_pid:
                continue
                
            # Check if it's a Python process running our app
            if proc.name().lower().startswith('python'):
                cmdline = proc.cmdline()
                if any('streamlit' in cmd.lower() for cmd in cmdline) and any('app.py' in cmd for cmd in cmdline):
                    logger.info(f"Found existing Streamlit process (PID: {proc.pid}), attempting graceful shutdown...")
                    try:
                        # Try graceful termination first
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)  # Wait up to 3 seconds
                            logger.info(f"Process {proc.pid} terminated gracefully")
                            killed_processes.append(proc.pid)
                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            logger.warning(f"Process {proc.pid} did not terminate gracefully, force killing...")
                            proc.kill()
                            killed_processes.append(proc.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(f"Could not terminate process {proc.pid}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing PID {proc.pid}: {str(e)}")
            continue
    
    # Wait for port to be released with exponential backoff
    max_wait = 15  # Maximum seconds to wait
    wait_interval = 0.5  # Initial wait interval
    waited = 0
    attempts = 0
    
    while is_process_running() and waited < max_wait:
        time.sleep(wait_interval)
        waited += wait_interval
        attempts += 1
        # Exponential backoff
        wait_interval = min(wait_interval * 1.5, 2.0)
        logger.info(f"Waiting for port to be released... (attempt {attempts}, {waited:.1f}s)")
    
    if waited >= max_wait:
        logger.error("Timeout waiting for port to be released")
        return False
    
    if killed_processes:
        logger.info(f"Successfully terminated {len(killed_processes)} processes: {killed_processes}")
    else:
        logger.info("No existing processes found to terminate")
    
    # Additional verification that port is actually free
    if is_process_running():
        logger.error("Port is still in use after process termination")
        return False
    
    return True

# Check for existing processes before starting
if is_process_running():
    logger.warning("Another instance of the app is already running on port 8502")
    if not kill_existing_processes():
        logger.error("Failed to terminate existing processes. Please manually close any running instances.")
        sys.exit(1)
    logger.info("Successfully cleared existing processes, starting new instance...")
    time.sleep(2)  # Additional wait to ensure clean startup

# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Email Analysis Dashboard")

# Add port configuration for Streamlit
os.environ["STREAMLIT_SERVER_PORT"] = "8502"

import duckdb
import json
from datetime import datetime, timedelta
import os
from groq import Groq
import requests
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from functools import partial
import re
from bs4 import BeautifulSoup
import html2text
from typing import Dict, Any
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np
import plotly.express as px
import pandas as pd
import psutil

# Load environment variables only once
if 'env_loaded' not in st.session_state:
    # Force reload of environment variables
    load_dotenv(override=True)
    st.session_state.env_loaded = True
    logger.info("Environment variables loaded")
    
    # Clear any cached clients
    if 'groq_client' in st.session_state:
        del st.session_state.groq_client
    if 'clients_initialized' in st.session_state:
        del st.session_state.clients_initialized
    
    # Add debug logging for API keys
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        # Log only first 4 and last 4 characters for security
        masked_key = f"{groq_key[:4]}...{groq_key[-4:]}" if len(groq_key) > 8 else "***"
        logger.info(f"Groq API key loaded (masked): {masked_key}")
    else:
        logger.error("Groq API key not found in environment variables")

# Model configurations
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat').strip()  # Remove any whitespace
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

# Add after the existing model configurations
MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'groq')  # Default to groq, can be 'deepseek' or 'groq'

# Helper function to safely parse environment variables
def safe_env_int(var_name, default):
    value = os.getenv(var_name, str(default))
    # Remove any comments (everything after #) and strip whitespace
    value = value.split('#')[0].strip()
    # Remove any trailing comments in parentheses
    value = re.sub(r'\s*\([^)]*\)', '', value)
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid value for {var_name}: {value}, using default: {default}")
        return default

def safe_env_float(var_name, default):
    value = os.getenv(var_name, str(default))
    # Remove any comments (everything after #) and strip whitespace
    value = value.split('#')[0].strip()
    # Remove any trailing comments in parentheses
    value = re.sub(r'\s*\([^)]*\)', '', value)
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid value for {var_name}: {value}, using default: {default}")
        return default

# Query model configurations
QUERY_MODEL_CONFIGS = {
    'deepseek': {
        'name': DEEPSEEK_MODEL,
        'max_tokens': safe_env_int('DEEPSEEK_MAX_TOKENS', 8192),
        'temperature': safe_env_float('DEEPSEEK_TEMPERATURE', 0.5)
    },
    'groq': {
        'name': GROQ_MODEL,
        'max_tokens': safe_env_int('GROQ_MAX_TOKENS', 8192),
        'temperature': safe_env_float('GROQ_TEMPERATURE', 0.5),
        'top_p': safe_env_float('GROQ_TOP_P', 1),
        'frequency_penalty': safe_env_float('GROQ_FREQUENCY_PENALTY', 0)
    }
}

# Analysis model configuration (using DeepSeek)
ANALYSIS_MODEL_CONFIG = {
    'large': {
        'groq': {
            'name': GROQ_MODEL,
            'max_tokens': 2048,  # Reduced from 4096
            'temperature': 0.3,
            'use_cases': ['complex_analysis', 'summarization', 'pattern_detection']
        },
        'deepseek': {
            'name': DEEPSEEK_MODEL,
            'max_tokens': 2048,  # Reduced from 4096
            'temperature': 0.3,
            'use_cases': ['complex_analysis', 'summarization', 'pattern_detection']
        }
    },
    'small': {
        'groq': {
            'name': GROQ_MODEL,
            'max_tokens': 1024,  # Reduced from 2048
            'temperature': 0.2,
            'use_cases': ['simple_queries', 'factual_lookup', 'basic_summary']
        },
        'deepseek': {
            'name': DEEPSEEK_MODEL,
            'max_tokens': 1024,  # Reduced from 2048
            'temperature': 0.2,
            'use_cases': ['simple_queries', 'factual_lookup', 'basic_summary']
        }
    }
}

# Load Anthropic API key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = 'claude-3-5-sonnet-20241022'

# Add Anthropic model config with conservative token limits
ANALYSIS_MODEL_CONFIG['small']['anthropic'] = {
    'name': ANTHROPIC_MODEL,
    'max_tokens': 1024,  # Reduced from 2048
    'temperature': 0.2,
    'use_cases': ['simple_queries', 'factual_lookup', 'basic_summary']
}
ANALYSIS_MODEL_CONFIG['large']['anthropic'] = {
    'name': ANTHROPIC_MODEL,
    'max_tokens': 2048,  # Reduced from 4096
    'temperature': 0.3,
    'use_cases': ['complex_analysis', 'summarization', 'pattern_detection']
}

# Add at the top with other global variables
DB_PATH = 'emails.db'  # Define database path globally

# Add after the existing model configurations
INCIDENT_CATEGORIES = [
    "Temperature", "Pressure", "Mechanical", "Fluid Leak", 
    "Safety", "Electrical", "Others"
]

SEVERITY_LEVELS = ["Low", "Medium", "High"]

# OpenSearch configuration
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', '9200'))
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', 'admin')
OPENSEARCH_INDEX_NAME = "email_summaries"
OPENSEARCH_INDEX_SETTINGS = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True,
            "knn.algo_param.ef_search": 100
        }
    },
    "mappings": {
        "properties": {
            "email_id": {"type": "long"},
            "subject": {"type": "text"},
            "summary": {"type": "text"},
            "summary_vector": {
                "type": "knn_vector",
                "dimension": 384,  # Updated to match all-MiniLM-L6-v2 model output
                "method": {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                    "parameters": {
                        "ef_search": 100,
                        "m": 16,
                        "ef_construction": 200
                    }
                }
            },
            "incident_type": {"type": "keyword"},
            "severity": {"type": "keyword"},
            "created_at": {"type": "date"},
            "created_at_timestamp": {"type": "long"},
            "is_analyzed": {"type": "boolean"}
        }
    }
}

# Initialize sentence transformer model for embeddings
if 'embedding_model' not in st.session_state:
    try:
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initialized sentence transformer model for embeddings")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        st.error("Failed to initialize embedding model. Please check the logs for details.")
        st.stop()

# Initialize OpenSearch client
if 'opensearch_client' not in st.session_state:
    try:
        logger.info("Attempting to initialize OpenSearch client...")
        logger.info(f"OpenSearch host: {OPENSEARCH_HOST}, port: {OPENSEARCH_PORT}")
        
        # First check if OpenSearch is accessible and get plugin info
        try:
            response = requests.get(f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/_cat/plugins")
            logger.info(f"Available OpenSearch plugins: {response.text}")
            
            # Check specifically for KNN plugin
            response = requests.get(f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/_cat/plugins?format=json")
            plugins = response.json()
            knn_plugin = next((p for p in plugins if 'knn' in p.get('component', '').lower()), None)
            if knn_plugin:
                logger.info(f"KNN plugin found: {knn_plugin}")
            else:
                logger.error("KNN plugin not found in OpenSearch plugins")
                raise Exception("KNN plugin not found in OpenSearch")
        except Exception as e:
            logger.error(f"Failed to verify OpenSearch plugins: {str(e)}")
            raise
        
        # Try to get current index settings if index exists
        try:
            if requests.get(f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/{OPENSEARCH_INDEX_NAME}").status_code == 200:
                response = requests.get(f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}/{OPENSEARCH_INDEX_NAME}/_settings")
                logger.info(f"Current index settings: {response.json()}")
        except Exception as e:
            logger.info(f"Index does not exist or error getting settings: {str(e)}")
        
        # Initialize client
        logger.info("Initializing OpenSearch client with settings...")
        logger.info(f"Index settings to be applied: {json.dumps(OPENSEARCH_INDEX_SETTINGS, indent=2)}")
        
        st.session_state.opensearch_client = OpenSearch(
            hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Create index if it doesn't exist
        if not st.session_state.opensearch_client.indices.exists(index=OPENSEARCH_INDEX_NAME):
            logger.info(f"Creating new index: {OPENSEARCH_INDEX_NAME}")
            try:
                st.session_state.opensearch_client.indices.create(
                    index=OPENSEARCH_INDEX_NAME,
                    body=OPENSEARCH_INDEX_SETTINGS
                )
                logger.info(f"Successfully created OpenSearch index: {OPENSEARCH_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Failed to create index: {str(e)}")
                logger.error(f"Index settings that failed: {json.dumps(OPENSEARCH_INDEX_SETTINGS, indent=2)}")
                raise
        else:
            logger.info(f"Using existing OpenSearch index: {OPENSEARCH_INDEX_NAME}")
        
        logger.info("OpenSearch client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing OpenSearch client: {str(e)}")
        st.error("Failed to initialize OpenSearch client. Please check the logs for details.")
        st.stop()

# Get OpenSearch client from session state
try:
    opensearch_client = st.session_state.opensearch_client
except Exception as e:
    logger.error(f"Error accessing OpenSearch client: {str(e)}")
    st.error("Failed to access OpenSearch client. Please check the logs for details.")
    st.stop()

def get_db_connection():
    """Get a database connection, creating one if it doesn't exist"""
    try:
        if 'conn' not in st.session_state or st.session_state.conn is None:
            logger.info("No active database connection, initializing new one")
            conn = init_database()
            if conn is None:
                logger.error("Failed to initialize database connection")
                return None
            st.session_state.conn = conn
        return st.session_state.conn
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}")
        return None

def get_available_deepseek_models():
    """Get list of available models from DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{DEEPSEEK_API_BASE}/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()['data']
            model_names = [model['id'] for model in models]
            logger.info(f"Available DeepSeek models: {model_names}")
            return model_names
        else:
            error_msg = f"Error getting DeepSeek models: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return []
    except Exception as e:
        logger.error(f"Error fetching DeepSeek models: {str(e)}")
        return []

# Initialize database connection only once
if 'db_initialized' not in st.session_state:
    # Force reinitialize database to ensure correct schema
    try:
        # First, try to delete the database file if it exists
        if os.path.exists(DB_PATH):
            try:
                # Close any existing connections
                if 'conn' in st.session_state and st.session_state.conn is not None:
                    try:
                        st.session_state.conn.close()
                    except:
                        pass
                    st.session_state.conn = None
                
                # Small delay to ensure connections are closed
                time.sleep(0.5)
                
                # Delete the database file
                os.remove(DB_PATH)
                logger.info(f"Deleted existing database file {DB_PATH}")
            except Exception as e:
                logger.error(f"Error deleting database file: {str(e)}")
                st.error("Failed to initialize database. Please check the logs for details.")
                st.stop()
        
        # Create a new connection with a fresh database
        conn = duckdb.connect(DB_PATH)
        
        # Create tables with correct schema
        conn.execute("""
            CREATE TABLE emails (
                id BIGINT PRIMARY KEY,
                email_subject TEXT,
                email_text_body TEXT,
                email_to TEXT,
                email_from TEXT,
                incident_type TEXT,
                severity TEXT,
                is_analyzed BOOLEAN DEFAULT FALSE,
                analyzed_at TIMESTAMP DEFAULT NULL,
                summary TEXT,
                analysis_quality FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE email_analysis (
                id BIGINT PRIMARY KEY,
                email_id BIGINT,
                procedural_deviations TEXT,
                recurrence_indicators TEXT,
                systemic_trends TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email_id) REFERENCES emails(id)
            )
        """)

        conn.execute("""
            CREATE TABLE query_cache (
                id BIGINT PRIMARY KEY,
                query_text TEXT,
                response_text TEXT,
                context_size BIGINT,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(query_text, context_size, model_name)
            )
        """)
        
        st.session_state.conn = conn
        st.session_state.db_initialized = True
        logger.info("Database initialized with correct schema")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        st.error("Failed to initialize database. Please check the logs for details.")
        st.stop()

# Initialize clients only once using session state
if 'clients_initialized' not in st.session_state:
    # Force reload of environment variables before client initialization
    load_dotenv(override=True)
    groq_key = os.getenv('GROQ_API_KEY')
    
    # Initialize Groq client
    if groq_key:
        try:
            st.session_state.groq_client = Groq(api_key=groq_key)
            # Test the client with a minimal request
            test_response = st.session_state.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            st.session_state.groq_available = True
            logger.info("Groq client initialized and tested successfully")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            st.session_state.groq_available = False
            st.warning("Groq API is not available. Will use DeepSeek for queries.")
    else:
        logger.warning("Groq API key not found in environment variables")
        st.session_state.groq_available = False
        st.warning("Groq API key not found. Will use DeepSeek for queries.")
    
    # Initialize DeepSeek model
    try:
        available_models = get_available_deepseek_models()
        if available_models:
            logger.info(f"Found {len(available_models)} available DeepSeek models")
            # Prefer deepseek-chat if available, otherwise use the first model
            if 'deepseek-chat' in available_models:
                st.session_state.deepseek_model = 'deepseek-chat'
                logger.info("Using deepseek-chat model")
            else:
                st.session_state.deepseek_model = available_models[0]
                logger.info(f"Using fallback model: {available_models[0]}")
        else:
            logger.warning("Could not fetch available DeepSeek models, using default")
            st.session_state.deepseek_model = 'deepseek-chat'
    except Exception as e:
        logger.error(f"Error initializing DeepSeek model: {str(e)}")
        st.session_state.deepseek_model = 'deepseek-chat'
    
    st.session_state.clients_initialized = True
    logger.info("Clients initialized")

# Use session state clients and model
groq_client = st.session_state.get('groq_client')
groq_available = st.session_state.get('groq_available', False)
DEEPSEEK_MODEL = st.session_state.deepseek_model

# Get database connection - use get_db_connection() instead of direct access
conn = get_db_connection()
if conn is None:
    logger.error("Failed to get database connection")
    st.error("Failed to get database connection. Please check the logs for details.")
    st.stop()

# Increase context size limit for better coverage
MAX_CONTEXT_SIZE = 30000  # Increased from 15000
MAX_EMAILS_PER_CONTEXT = 50  # Increased from 16

def get_deepseek_model_config():
    """Get the DeepSeek model configuration"""
    return {
        'name': DEEPSEEK_MODEL,  # Always use the configured DeepSeek model
        'max_tokens': safe_env_int('DEEPSEEK_MAX_TOKENS', 8192),
        'temperature': safe_env_float('DEEPSEEK_TEMPERATURE', 0.5)
    }

def get_groq_model_config():
    """Get the Groq model configuration"""
    return {
        'name': GROQ_MODEL,
        'max_tokens': safe_env_int('GROQ_MAX_TOKENS', 8192),
        'temperature': safe_env_float('GROQ_TEMPERATURE', 0.5),
        'top_p': safe_env_float('GROQ_TOP_P', 1),
        'frequency_penalty': safe_env_float('GROQ_FREQUENCY_PENALTY', 0)
    }

# Query model configurations - use functions to ensure fresh configs
QUERY_MODEL_CONFIGS = {
    'deepseek': get_deepseek_model_config,
    'groq': get_groq_model_config
}

def get_deepseek_response(messages, model_config, model_name):
    """Make a request to DeepSeek API directly"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Fixed API endpoint URL - ensure no duplicate /v1/
        api_url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
        logger.info(f"Making DeepSeek API request to {api_url}")
        
        # Use the provided model name
        logger.info(f"Using DeepSeek model: {model_name}")
        
        # Only include supported parameters
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": model_config['temperature'],
            "max_tokens": model_config['max_tokens']
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            error_msg = f"DeepSeek API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            logger.error(f"Request details - URL: {api_url}")
            logger.error(f"Request details - Model: {model_name}")
            logger.error(f"Request details - Data: {data}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"Error calling DeepSeek API: {str(e)}")
        raise

def get_groq_response(messages, model_config):
    """Get response from Groq API with proper error handling"""
    try:
        # Validate and ensure max_tokens is a valid integer
        max_tokens = model_config.get('max_tokens')
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            logger.warning(f"Invalid max_tokens value: {max_tokens}, using default of 2048")
            max_tokens = 2048  # Default to a reasonable value
        
        # Only include supported parameters
        request_params = {
            'model': model_config['name'],
            'messages': messages,
            'temperature': model_config.get('temperature', 0.5),
            'max_tokens': max_tokens
        }
        
        # Add optional parameters if they exist
        if 'top_p' in model_config:
            request_params['top_p'] = model_config['top_p']
        if 'frequency_penalty' in model_config:
            request_params['frequency_penalty'] = model_config['frequency_penalty']
        
        logger.info(f"Making Groq API request with model: {model_config['name']}, max_tokens: {max_tokens}")
        response = groq_client.chat.completions.create(**request_params)
        logger.info("Successfully received response from Groq API")
        return response
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise

def get_available_models():
    """
    Get list of available models from Together AI.
    
    Returns:
        list: List of available model names
    """
    try:
        # Check if we already have the list in session state
        if 'available_models' in st.session_state:
            return st.session_state.available_models
        
        # Get models from Together AI
        try:
            models = together_client.Models.list()
            available_models = [model.id for model in models]
            st.session_state.available_models = available_models
            logger.info(f"Found {len(available_models)} available models: {available_models}")
            return available_models
        except Exception as e:
            logger.warning(f"Error getting Together AI models: {str(e)}")
            return [DEEPSEEK_MODEL]  # Fallback to default model
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return [DEEPSEEK_MODEL]  # Fallback to default model

def check_model_availability(model_name):
    """
    Check if a model is available by making a minimal test request.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Make a minimal test request
        response = together_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Test"},
                {"role": "user", "content": "Test"}
            ],
            temperature=0.1,
            max_tokens=5
        )
        return True
    except Exception as e:
        logger.warning(f"Model {model_name} not available: {str(e)}")
        return False

def summarize_emails_bulk(batch, model_name=None):
    """Generate summaries for a batch of emails using the configured model provider"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an expert at summarizing maintenance and service-related emails. 
                For each email, provide ONE concise sentence focusing on the most critical issue or action needed.
                Guidelines:
                1. Start each summary with 'Email ID X:'
                2. Keep summaries to ONE sentence
                3. Focus on the most critical issue
                4. Include the type of issue (e.g., mechanical, electrical)
                5. Mention severity if critical
                6. Be specific but brief
                7. Format: 'Email ID X: [Your one-sentence summary]'"""
            },
            {
                "role": "user",
                "content": f"Summarize these maintenance/service emails in ONE sentence each, focusing on the most critical issue or action needed:\n\n" + 
                          "\n\n".join([f"Email ID {id}:\nSubject: {subject}\nBody: {body}" for id, subject, body in batch])
            }
        ]

        response = get_model_response(messages, 'small')
        # Handle both string responses and object responses
        if isinstance(response, str):
            summaries_text = response.strip()
        else:
            summaries_text = response.choices[0].message.content.strip()
        
        # Parse summaries
        summaries = {}
        for line in summaries_text.split('\n'):
            if line.startswith('Email ID'):
                try:
                    parts = line.split(':', 1)
                    email_id = int(parts[0].split()[-1])
                    summary = parts[1].strip()
                    summaries[email_id] = summary
                except (ValueError, IndexError):
                    continue
        
        return summaries
    except Exception as e:
        logger.error(f"Error in bulk summarization: {str(e)}")
        return {}

def select_model(query_type, query_text=None):
    """
    Select the appropriate model based on query type and complexity.
    For analysis tasks, returns DeepSeek model config.
    For queries, returns Groq model config.
    """
    if query_type in ['simple_queries', 'factual_lookup', 'basic_summary']:
        return ANALYSIS_MODEL_CONFIG['small'].copy()  # Return a copy to avoid modifying the original
    return ANALYSIS_MODEL_CONFIG['large'].copy()  # Return a copy to avoid modifying the original


def clean_html_to_text(html):
    # Parse HTML and remove script/style
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
        tag.extract()
    # Get full text with line breaks
    text = soup.get_text(separator="\n")
    # Optionally, also try html2text for better formatting:
    # text = html2text.html2text(html)
    
    # Split into lines and trim
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Filter out common boilerplate/disclaimer lines
    stop_keywords = [
        r"confidentiality notice", r"not intended recipient", 
        r"unauthorized", r"unsubscribe", r"copyright", r"disclaimer"
    ]
    filtered = []
    for line in lines:
        low = line.lower()
        if any(re.search(kw, low) for kw in stop_keywords):
            # Stop processing further if it's the start of a footer
            break
        filtered.append(line)
    
    # Re-join into clean text
    return "\n".join(filtered)

def analyze_email_content(subject: str, body: str, model_name: str = None) -> Dict[str, Any]:
    """Analyze email content using the configured model provider"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an expert at analyzing maintenance and service-related emails. 
                Your task is to accurately categorize incidents and assess their severity.
                You MUST:
                1. Follow the exact format specified
                2. Choose the most appropriate category based on the actual content
                3. Provide specific reasoning from the email
                4. Be conservative in severity assessment - only mark as High if truly critical
                5. Consider both immediate and potential impacts"""
            },
            {
                "role": "user",
                "content": f"""Analyze this maintenance/service email and categorize it according to the incident type and severity.

        Email Subject: {subject}
        Email Body: {body}

        Guidelines for categorization:

        INCIDENT TYPES (choose ONE most relevant):
        - Temperature: Issues related to temperature control, heating, cooling, thermal systems
        - Pressure: Pressure-related issues, pressure vessels, pressure control systems
        - Mechanical: Mechanical failures, equipment breakdowns, moving parts, structural issues
        - Fluid Leak: Any type of fluid leakage, spills, containment issues
        - Safety: Safety concerns, hazards, compliance issues, emergency situations
        - Electrical: Electrical systems, power issues, electrical equipment failures
        - Others: Any issue that doesn't fit the above categories

        SEVERITY LEVELS (choose ONE):
        - High: Critical issues requiring immediate attention, safety hazards, system failures
          Examples: Active leaks, electrical hazards, safety violations, critical system failures
        - Medium: Issues that need attention but aren't critical
          Examples: Non-critical equipment malfunctions, maintenance needs, performance issues
        - Low: Routine maintenance, non-urgent issues, general inquiries
          Examples: Regular maintenance requests, minor issues, general questions

        Please analyze and respond in EXACTLY this format:
        INCIDENT_TYPE: [Choose ONE from: Temperature, Pressure, Mechanical, Fluid Leak, Safety, Electrical, Others]
        SEVERITY: [Choose ONE from: Low, Medium, High]
        REASONING: [Brief explanation for your classification, citing specific details from the email]

        Important:
        1. Be specific in your reasoning
        2. Cite actual details from the email
        3. Consider both subject and body content
        4. If multiple issues exist, choose the most critical one
        5. Use EXACTLY the format shown above"""
            }
        ]

        response = get_model_response(messages, 'small')
        # Handle both string responses and object responses
        if isinstance(response, str):
            analysis_text = response.strip()
        else:
            analysis_text = response.choices[0].message.content.strip()
        
        # Parse the response
        lines = analysis_text.split('\n')
        analysis = {
            'incident_type': 'Others',  # Default values
            'severity': 'Low',
            'reasoning': 'No specific analysis available'
        }
        
        for line in lines:
            if line.startswith('INCIDENT_TYPE:'):
                incident_type = line.split(':', 1)[1].strip()
                if incident_type in INCIDENT_CATEGORIES:
                    analysis['incident_type'] = incident_type
            elif line.startswith('SEVERITY:'):
                severity = line.split(':', 1)[1].strip()
                if severity in SEVERITY_LEVELS:
                    analysis['severity'] = severity
            elif line.startswith('REASONING:'):
                analysis['reasoning'] = line.split(':', 1)[1].strip()
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing email content: {str(e)}")
        return {
            'incident_type': 'Others',
            'severity': 'Low',
            'reasoning': f'Error during analysis: {str(e)}'
        }

def store_email(email_data):
    """Store email in DuckDB and only its summary in OpenSearch"""
    try:
        email = email_data['Email']
        # Extract HTML and convert to clean text
        html_body = email.get('HtmlBody', '')
        text_body = clean_html_to_text(html_body)
        
        # Get database connection
        conn = get_db_connection()
        
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM emails").fetchone()[0]
        
        # Get model name from session state
        model_name = st.session_state.deepseek_model
        
        # Analyze email content using LLM
        analysis = analyze_email_content(email['Subject'], text_body, model_name)
        
        # Generate summary for the email
        summary = summarize_emails_bulk([(next_id, text_body, email['Subject'])], model_name)[next_id]
        
        # Store the email in DuckDB
        conn.execute('''
            INSERT INTO emails (
                id, email_subject, email_text_body, email_to, email_from,
                incident_type, severity, is_analyzed, summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, ?, CURRENT_TIMESTAMP)
        ''', (
            next_id,
            email['Subject'],
            text_body,
            email['To'],
            email['From'],
            analysis['incident_type'],
            analysis['severity'],
            summary  # Store summary in DuckDB as well
        ))
        
        # Store only the summary in OpenSearch
        try:
            current_time = datetime.now()
            
            # Generate embedding for the summary
            try:
                # Ensure the embedding is a numpy array of float32
                embedding = st.session_state.embedding_model.encode(
                    summary,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype('float32')
                
                # Verify dimension matches the model output
                expected_dim = 384  # all-MiniLM-L6-v2 dimension
                if embedding.shape[0] != expected_dim:
                    raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {expected_dim}")
                
                # Convert to list for OpenSearch
                summary_vector = embedding.tolist()
                
                # Store in OpenSearch using the correct API
                opensearch_client.index(
                    index=OPENSEARCH_INDEX_NAME,
                    body={  # Changed from 'document' to 'body'
                        'email_id': next_id,
                        'subject': email['Subject'],
                        'incident_type': analysis['incident_type'],
                        'severity': analysis['severity'],
                        'created_at': current_time.isoformat(),
                        'created_at_timestamp': int(current_time.timestamp()),
                        'is_analyzed': False,
                        'summary': summary,
                        'summary_vector': summary_vector
                    },
                    refresh=True
                )
                logger.info(f"Stored summary for email {next_id} in OpenSearch")
            except Exception as e:
                logger.error(f"Error generating or storing embedding: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error storing in OpenSearch: {str(e)}")
            # Continue even if OpenSearch storage fails - we still have the email in DuckDB
        
        logger.info(f"Stored email {next_id} in DuckDB")
        logger.info(f"Analysis: Type={analysis['incident_type']}, Severity={analysis['severity']}")
        return True
    except Exception as e:
        logger.error(f"Error storing email: {str(e)}")
        return False

def store_multiple_emails(emails_data):
    """Store multiple emails in database"""
    logger.info(f"Starting batch import of {len(emails_data)} emails")
    success_count = 0
    error_count = 0
    error_messages = []

    # Get database connection
    conn = get_db_connection()

    # Start a transaction
    conn.execute("BEGIN TRANSACTION")
    try:
        for i, email_data in enumerate(emails_data, 1):
            try:
                if store_email(email_data):
                    success_count += 1
                else:
                    error_count += 1
                    error_messages.append(f"Failed to store email {i}")
            except Exception as e:
                error_count += 1
                error_messages.append(f"Error processing email {i}: {str(e)}")
        
        # Commit the transaction if all successful
        if error_count == 0:
            conn.execute("COMMIT")
            logger.info(f"Batch import completed successfully: {success_count} emails imported")
        else:
            conn.execute("ROLLBACK")
            logger.error(f"Batch import failed: {error_count} errors, rolling back transaction")
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(f"Transaction failed: {str(e)}")
        error_count = len(emails_data)
        success_count = 0
        error_messages.append(f"Transaction error: {str(e)}")

    return success_count, error_count, error_messages

def get_text_similarity(text1, text2):
    """Calculate text similarity using TF-IDF and cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception as e:
        logger.error(f"Error calculating text similarity: {str(e)}")
        return 0.0

def create_similarity_batches(emails, batch_size=10, similarity_threshold=0.7):
    """Create batches of emails based on text similarity"""
    logger.info(f"Creating similarity-based batches with threshold {similarity_threshold}")
    
    # Prepare email texts
    email_texts = {}
    for id, text_body, subject in emails:
        full_text = f"Subject: {subject}\nBody: {text_body}"
        email_texts[id] = {
            'text': full_text,
            'email': (id, text_body, subject)
        }
    
    # Create batches based on similarity
    batches = []
    processed_ids = set()
    
    for id, data in email_texts.items():
        if id in processed_ids:
            continue
            
        current_batch = [data['email']]
        processed_ids.add(id)
        
        # Find similar emails
        for other_id, other_data in email_texts.items():
            if other_id in processed_ids:
                continue
                
            similarity = get_text_similarity(data['text'], other_data['text'])
            if similarity >= similarity_threshold:
                current_batch.append(other_data['email'])
                processed_ids.add(other_id)
                
                if len(current_batch) >= batch_size:
                    break
        
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches)} similarity-based batches")
    return batches

def process_batch(batch, batch_num, total_batches, model_name):
    """Process a single batch of emails using DeepSeek as primary and Anthropic as fallback"""
    try:
        # Use a separate logger for thread operations to avoid Streamlit context issues
        thread_logger = logging.getLogger(f'thread_{batch_num}')
        thread_logger.setLevel(logging.INFO)
        
        # Add a handler if none exists
        if not thread_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            thread_logger.addHandler(handler)
        
        thread_logger.info(f"Processing batch {batch_num} of {total_batches}")
        
        # Get bulk summaries for all emails in the batch
        summaries = summarize_emails_bulk(batch, model_name)
        
        # Prepare context from summarized emails - more concise format
        context = "Maintenance and service emails:\n"
        for id, text_body, subject in batch:
            summary = summaries.get(id, f"Critical issue: {subject[:50]}...")
            context += f"\nEmail {id}: {summary}\n"
        
        prompt = f"""You are an expert at analyzing maintenance and service-related emails. Analyze these summarized emails and provide specific insights.

        {context}

        You MUST provide analysis in EXACTLY these three sections, with these EXACT headers:

        PROCEDURAL DEVIATIONS:
        [Your analysis here]

        RECURRENCE INDICATORS:
        [Your analysis here]

        SYSTEMIC TRENDS:
        [Your analysis here]

        For each section:
        1. Start with the EXACT header shown above
        2. Provide EXACTLY 1 bullet point (no more, no less)
        3. Each bullet point MUST start with "• Email ID X:" where X is the actual email ID
        4. Keep each bullet point to ONE sentence
        5. Focus on the most critical issue or pattern
        6. Be specific but concise"""

        # Use large model for complex analysis with DeepSeek as primary and Anthropic as fallback
        thread_logger.info("Using DeepSeek as primary model with Anthropic fallback for batch analysis")
        response = get_model_response(
            messages=[
                {"role": "system", "content": "You are an expert at analyzing maintenance and service-related emails. You MUST follow the exact format specified in the prompt, including exactly 1 bullet point per section. Keep each bullet point to ONE sentence and focus on the most critical issue."},
                {"role": "user", "content": prompt}
            ],
            model_size='large'  # Use large model for batch analysis
        )
        
        # Parse the response
        try:
            parts = re.split(r'(PROCEDURAL DEVIATIONS:|RECURRENCE INDICATORS:|SYSTEMIC TRENDS:)', response)
            section_map = {}
            for idx in range(1, len(parts), 2):
                header = parts[idx].rstrip(':')
                content = parts[idx+1].strip()
                # Ensure each bullet point is a single sentence
                bullet_points = content.split('•')
                cleaned_points = []
                for point in bullet_points:
                    if point.strip():
                        # Take only the first sentence
                        first_sentence = point.strip().split('.')[0] + '.'
                        cleaned_points.append('•' + first_sentence)
                section_map[header] = '\n'.join(cleaned_points)
        except Exception as e:
            thread_logger.error(f"Error parsing response for batch {batch_num}: {str(e)}")
            thread_logger.error(f"Response text: {response}")
            return None

        # Store analysis results for each email in the batch using a transaction
        thread_conn = None
        try:
            # Create a new database connection for this thread
            thread_conn = duckdb.connect(DB_PATH)
            thread_conn.execute("BEGIN TRANSACTION")
            
            # Update each email in the batch - using analyzed_at instead of analysis_timestamp
            for id, _, _ in batch:
                thread_conn.execute('''
                    UPDATE emails 
                    SET is_analyzed = TRUE,
                        analyzed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', [id])
            
            thread_conn.execute("COMMIT")
            thread_logger.info(f"Successfully processed batch {batch_num}")
            return section_map
            
        except Exception as e:
            if thread_conn:
                thread_conn.execute("ROLLBACK")
            thread_logger.error(f"Error storing batch {batch_num} results: {str(e)}")
            return None
        finally:
            if thread_conn:
                try:
                    thread_conn.close()
                except:
                    pass
            
    except Exception as e:
        thread_logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return None

def get_insights(include_analyzed=False, batch_size=10, use_similarity_batching=False, similarity_threshold=0.7):
    """Get insights for emails using parallel batch processing"""
    logger.info("Starting insights generation")
    
    # Get the model name before starting threads
    model_name = st.session_state.deepseek_model
    
    # Get database connection first
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to get database connection")
        return {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'procedural_deviations': 'Error: Could not connect to database',
            'recurrence_indicators': 'Error: Could not connect to database',
            'systemic_trends': 'Error: Could not connect to database',
            'email_count': 0,
            'analysis_stats': {
                'total_emails': 0,
                'analyzed_emails': 0,
                'referenced_emails': 0,
                'batches_processed': 0
            }
        }
    
    # Get emails based on include_analyzed flag
    query = '''
        SELECT id, email_text_body, email_subject
        FROM emails 
        WHERE is_analyzed = FALSE OR ? = TRUE
        ORDER BY id DESC
    '''
    try:
        all_emails = conn.execute(query, [include_analyzed]).fetchall()
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'procedural_deviations': f'Error: {str(e)}',
            'recurrence_indicators': f'Error: {str(e)}',
            'systemic_trends': f'Error: {str(e)}',
            'email_count': 0,
            'analysis_stats': {
                'total_emails': 0,
                'analyzed_emails': 0,
                'referenced_emails': 0,
                'batches_processed': 0
            }
        }
    
    if not all_emails:
        logger.info("No emails found in database")
        return {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'procedural_deviations': 'No emails found in database',
            'recurrence_indicators': 'No emails found in database',
            'systemic_trends': 'No emails found in database',
            'email_count': 0,
            'analysis_stats': {
                'total_emails': 0,
                'analyzed_emails': 0,
                'referenced_emails': 0,
                'batches_processed': 0
            }
        }
    
    logger.info(f"Found {len(all_emails)} emails to analyze")
    
    # Create batches based on similarity or chronological order
    if use_similarity_batching:
        batches = create_similarity_batches(all_emails, batch_size, similarity_threshold)
    else:
        batches = [all_emails[i:i + batch_size] for i in range(0, len(all_emails), batch_size)]
    
    # Process batches in parallel with a thread pool
    all_insights = {
        'PROCEDURAL DEVIATIONS': [],
        'RECURRENCE INDICATORS': [],
        'SYSTEMIC TRENDS': []
    }
    
    # Use a smaller number of workers to avoid overwhelming the system
    max_workers = min(3, len(batches))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with the total number of batches and model name
        process_batch_partial = partial(process_batch, total_batches=len(batches), model_name=model_name)
        
        # Submit all batches for processing
        future_to_batch = {
            executor.submit(process_batch_partial, batch, i+1): batch 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                section_map = future.result()
                if section_map:
                    all_insights["PROCEDURAL DEVIATIONS"].extend(section_map["PROCEDURAL DEVIATIONS"].splitlines())
                    all_insights["RECURRENCE INDICATORS"].extend(section_map["RECURRENCE INDICATORS"].splitlines())
                    all_insights["SYSTEMIC TRENDS"].extend(section_map["SYSTEMIC TRENDS"].splitlines())
            except Exception as e:
                logger.error(f"Error processing batch result: {str(e)}")

    # Combine insights from all batches
    combined_insights = {
        'PROCEDURAL DEVIATIONS': '\n'.join(all_insights['PROCEDURAL DEVIATIONS']),
        'RECURRENCE INDICATORS': '\n'.join(all_insights['RECURRENCE INDICATORS']),
        'SYSTEMIC TRENDS': '\n'.join(all_insights['SYSTEMIC TRENDS'])
    }
    
    # Count references to specific emails in the analysis
    email_references = {}
    for section in combined_insights.values():
        for id in [id for id, _, _ in all_emails]:
            if str(id) in section:
                email_references[id] = email_references.get(id, 0) + 1
    
    # Get final analysis stats
    try:
        conn = get_db_connection()
        total_analyzed = conn.execute("SELECT COUNT(*) FROM emails WHERE is_analyzed = TRUE").fetchone()[0]
    except Exception as e:
        logger.error(f"Error getting analysis stats: {str(e)}")
        total_analyzed = 0
    
    analysis_stats = {
        'total_emails': len(all_emails),
        'analyzed_emails': total_analyzed,
        'referenced_emails': len(email_references),
        'batches_processed': len(batches),
        'batching_method': 'similarity' if use_similarity_batching else 'chronological'
    }
    
    logger.info(f"Analysis complete. Stats: {analysis_stats}")
    
    return {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'procedural_deviations': combined_insights['PROCEDURAL DEVIATIONS'] or 'No procedural deviations identified',
        'recurrence_indicators': combined_insights['RECURRENCE INDICATORS'] or 'No recurrence indicators identified',
        'systemic_trends': combined_insights['SYSTEMIC TRENDS'] or 'No systemic trends identified',
        'email_count': len(all_emails),
        'analysis_stats': analysis_stats
    }

def get_email_context(limit=10, query_text=None, days_back=30, min_analysis_quality=0.5, st_session=None):
    """Get context from analyzed emails using OpenSearch vector search and stored summaries"""
    try:
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days_back)
        date_threshold_timestamp = int(date_threshold.timestamp())
        logger.info(f"Retrieving context with parameters: limit={limit}, days_back={days_back}, date_threshold={date_threshold.isoformat()}")
        
        if query_text:
            try:
                logger.info(f"Processing query: '{query_text}'")
                
                # Generate query embedding using the stored model
                query_embedding = st.session_state.embedding_model.encode(
                    query_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype('float32')
                
                # Ensure the embedding is a list of floats
                query_vector = query_embedding.tolist()
                logger.info(f"Generated query vector of length {len(query_vector)}")
                
                # Query OpenSearch using vector search
                logger.info("Querying OpenSearch using vector search...")
                query_body = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "knn": {
                                        "summary_vector": {
                                            "vector": query_vector,  # Use the list version
                                            "k": limit * 2
                                        }
                                    }
                                }
                            ],
                            "filter": [
                                {
                                    "range": {
                                        "created_at_timestamp": {
                                            "gte": date_threshold_timestamp
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "fields": ["email_id", "subject", "incident_type", "severity", "created_at", "summary"],
                    "size": limit * 2,
                    "_source": True
                }
                
                results = opensearch_client.search(
                    index=OPENSEARCH_INDEX_NAME,
                    body=query_body
                )
                
                if not results['hits']['hits']:
                    logger.warning("No results from vector search")
                    return f"No relevant emails found in the last {days_back} days.", 0
                
                # Get email IDs from OpenSearch results
                email_ids = [int(hit['_source']['email_id']) for hit in results['hits']['hits']]
                logger.info(f"Found {len(email_ids)} relevant emails from vector search")
                
                # Get full email details from DuckDB
                conn = get_db_connection()
                emails = conn.execute('''
                    SELECT id, email_subject, email_text_body, incident_type, severity
                    FROM emails 
                    WHERE id IN ({})
                '''.format(','.join('?' * len(email_ids))), email_ids).fetchall()
                
                # Build context from results using stored summaries
                context_parts = []
                total_chars = 0
                used_emails = set()
                
                for hit in results['hits']['hits']:
                    email_id = int(hit['_source']['email_id'])
                    if email_id in used_emails or total_chars >= MAX_CONTEXT_SIZE:
                        continue
                    
                    # Find matching email details
                    email_detail = next((e for e in emails if e[0] == email_id), None)
                    if email_detail:
                        id, subject, body, inc_type, sev = email_detail
                        
                        # Use the stored summary from OpenSearch
                        stored_summary = hit['_source']['summary']
                        similarity_score = 1 - hit['_score']  # Convert distance to similarity
                        
                        email_content = f"""Email {id} (Similarity: {similarity_score:.2f}):
Subject: {subject}
Incident Type: {inc_type}
Severity: {sev}
Summary: {stored_summary}
Body Preview: {body[:200]}..."""
                        
                        email_size = len(email_content)
                        if total_chars + email_size <= MAX_CONTEXT_SIZE:
                            used_emails.add(email_id)
                            context_parts.append(email_content)
                            total_chars += email_size
                            logger.info(f"Added email {id} to context (size: {email_size} chars, total: {total_chars} chars)")
                
                # Build final context
                context = f"Similar email summaries (showing {len(used_emails)} emails, {total_chars} chars)\n"
                context += f"Query: {query_text}\n"
                context += f"Analysis period: Last {days_back} days\n"
                context += "\n".join(context_parts)
                
                logger.info(f"Final context built with {len(used_emails)} emails, {total_chars} total characters")
                return context, len(used_emails)
                
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                return f"Error retrieving email context: {str(e)}", 0
        else:
            # If no query text, get recent emails directly from OpenSearch
            logger.info("No query text provided, retrieving recent emails")
            query_body = {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "created_at_timestamp": {
                                        "gte": date_threshold_timestamp
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {"created_at_timestamp": {"order": "desc"}}
                ],
                "fields": ["email_id", "subject", "incident_type", "severity", "created_at", "summary"],
                "size": limit,
                "_source": True
            }
            
            results = opensearch_client.search(
                index=OPENSEARCH_INDEX_NAME,
                body=query_body
            )
            
            if not results['hits']['hits']:
                logger.warning(f"No emails found in the last {days_back} days")
                return f"No emails found in the last {days_back} days.", 0
            
            # Get email IDs and fetch full details from DuckDB
            email_ids = [int(hit['_source']['email_id']) for hit in results['hits']['hits']]
            conn = get_db_connection()
            emails = conn.execute('''
                SELECT id, email_subject, email_text_body, incident_type, severity
                FROM emails 
                WHERE id IN ({})
            '''.format(','.join('?' * len(email_ids))), email_ids).fetchall()
            
            # Build context using stored summaries
            context_parts = []
            total_chars = 0
            
            for hit in results['hits']['hits']:
                email_id = int(hit['_source']['email_id'])
                if total_chars >= MAX_CONTEXT_SIZE:
                    break
                
                # Find matching email details
                email_detail = next((e for e in emails if e[0] == email_id), None)
                if email_detail:
                    id, subject, body, inc_type, sev = email_detail
                    stored_summary = hit['_source']['summary']
                    
                    email_content = f"""Email {id}:
Subject: {subject}
Incident Type: {inc_type}
Severity: {sev}
Summary: {stored_summary}
Body Preview: {body[:200]}..."""
                    
                    email_size = len(email_content)
                    if total_chars + email_size <= MAX_CONTEXT_SIZE:
                        context_parts.append(email_content)
                        total_chars += email_size
                        logger.info(f"Added email {id} to context (size: {email_size} chars, total: {total_chars} chars)")
            
            # Build final context
            context = f"Recent email summaries (showing {len(context_parts)} emails, {total_chars} chars)\n"
            context += f"Analysis period: Last {days_back} days\n"
            context += "\n".join(context_parts)
            
            logger.info(f"Final context built with {len(context_parts)} emails, {total_chars} total characters")
            return context, len(context_parts)
        
    except Exception as e:
        logger.error(f"Error getting email context: {str(e)}")
        return "Error retrieving email context. Please check the logs for details.", 0

def get_fallback_context(emails, query_text, limit, days_back, model_name):
    """Fallback method using simple text similarity when vector search fails"""
    try:
        # Use TF-IDF for simple text similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        texts = [f"{subject} {body[:500]}" for _, subject, body, _, _ in emails]
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Get top similar emails
        top_indices = similarities.argsort()[-limit:][::-1]
        
        context_parts = []
        total_chars = 0
        
        for idx in top_indices:
            if total_chars >= MAX_CONTEXT_SIZE:
                break
                
            id, subject, body, inc_type, sev = emails[idx]
            similarity = similarities[idx]
            
            # Generate summary for this email
            summary = summarize_emails_bulk([(id, body, subject)], model_name)[id]
            email_content = f"""Email {id} (Similarity: {similarity:.2f}):
Subject: {subject}
Incident Type: {inc_type}
Severity: {sev}
Summary: {summary}
Body Preview: {body[:200]}..."""
            
            email_size = len(email_content)
            if total_chars + email_size <= MAX_CONTEXT_SIZE:
                context_parts.append(email_content)
                total_chars += email_size
        
        context = f"Similar email summaries (showing {len(context_parts)} emails, {total_chars} chars)\n"
        context += f"Query: {query_text}\n"
        context += f"Analysis period: Last {days_back} days\n"
        context += "\n".join(context_parts)
        
        return context, len(context_parts)
        
    except Exception as e:
        logger.error(f"Error in fallback context generation: {str(e)}")
        return "Error generating context. Please try a different query.", 0

def cache_query_result(query_text, response_text, context_size, model_name):
    """Cache query result with model information"""
    try:
        conn = get_db_connection()
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM query_cache").fetchone()[0]
        
        # Convert response to string if it's a ChatCompletion object
        if hasattr(response_text, 'choices'):
            response_text = response_text.choices[0].message.content
        elif not isinstance(response_text, str):
            response_text = str(response_text)
        
        # First delete any existing cache entry for this query to avoid duplicates
        conn.execute('''
            DELETE FROM query_cache
            WHERE query_text = ? AND context_size = ? AND model_name = ?
        ''', (query_text, context_size, model_name))
        
        # Insert new cache entry
        conn.execute('''
            INSERT INTO query_cache (id, query_text, response_text, context_size, model_name)
            VALUES (?, ?, ?, ?, ?)
        ''', (next_id, query_text, response_text, context_size, model_name))
        logger.info(f"Cached query result for: {query_text[:50]}... using model {model_name}")
    except Exception as e:
        logger.error(f"Error caching query result: {str(e)}")

def store_analysis_results(email_id, procedural_deviations, recurrence_indicators, systemic_trends):
    """Store LLM analysis results in database"""
    try:
        conn = get_db_connection()
        # Defensive check to convert None to empty string
        pd = procedural_deviations if procedural_deviations is not None else ""
        ri = recurrence_indicators if recurrence_indicators is not None else ""
        st = systemic_trends if systemic_trends is not None else ""

        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM analysis_results_new").fetchone()[0]
        
        conn.execute('''
            INSERT INTO analysis_results_new 
            (id, email_id, procedural_deviations, recurrence_indicators, systemic_trends)
            VALUES (?, ?, ?, ?, ?)
        ''', (next_id, email_id, pd, ri, st))
        logger.info(f"Stored analysis results for email {email_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing analysis results: {str(e)}")
        return False

def get_similar_emails(query_text, limit=5):
    """Get similar emails using OpenSearch vector similarity search"""
    try:
        # Query OpenSearch for similar documents
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": {
                            "knn": {
                                "field": "summary_vector",
                                "vector": st.session_state.embedding_model.encode([query_text]).tolist(),
                                "k": limit
                            }
                        }
                    }
                }
            }
        )
        
        if not results['hits']['hits']:
            return []
        
        # Get additional details from DuckDB
        conn = get_db_connection()
        email_ids = [int(hit['_source']['email_id']) for hit in results['hits']['hits']]
        
        # Get full email details
        email_details = conn.execute('''
            SELECT id, email_subject, email_text_body
            FROM emails 
            WHERE id IN ({})
        '''.format(','.join('?' * len(email_ids))), email_ids).fetchall()
        
        # Create results list with similarity scores
        results_list = []
        for hit in results['hits']['hits']:
            email_id = int(hit['_source']['email_id'])
            # Find matching email details
            email_detail = next((e for e in email_details if e[0] == email_id), None)
            if email_detail:
                _, subject, text = email_detail
                results_list.append((email_id, subject, text, 1 - hit['_score']))  # Convert similarity to distance
        
        return results_list
    except Exception as e:
        logger.error(f"Error finding similar emails: {str(e)}")
        return []

def get_cached_query(query_text, context_size, model_name):
    """Get cached query result if available"""
    try:
        conn = get_db_connection()
        if conn is None:
            return None
            
        # Look for exact match in cache
        result = conn.execute('''
            SELECT response_text 
            FROM query_cache 
            WHERE query_text = ? 
            AND context_size = ? 
            AND model_name = ?
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (query_text, context_size, model_name)).fetchone()
        
        if result:
            logger.info(f"Found cached response for query using model {model_name}")
            return result[0]
        return None
    except Exception as e:
        logger.error(f"Error checking query cache: {str(e)}")
        return None

def query_llm_with_context(query_text, context, model_name='groq'):
    """
    Query the LLM with email context using the specified model.
    Falls back to DeepSeek if Groq is unavailable.
    """
    try:
        # If Groq is requested but not available, fall back to DeepSeek
        if model_name == 'groq' and not groq_available:
            logger.warning("Groq requested but not available, falling back to DeepSeek")
            model_name = 'deepseek'
            st.warning("Groq is not available. Using DeepSeek instead.")
        
        # Get model config using the appropriate function
        model_config_fn = QUERY_MODEL_CONFIGS.get(model_name)
        if not model_config_fn:
            raise ValueError(f"Invalid model name: {model_name}")
        
        # Get fresh model config
        model_config = model_config_fn()
        
        # Check cache first
        cached_response = get_cached_query(query_text, len(context.split('\n')), model_name)
        if cached_response:
            logger.info(f"Using cached response for query with model {model_name}")
            return cached_response

        # Prepare the prompt with context
        messages = [
            {"role": "system", "content": """You are an expert at analyzing maintenance and service-related emails. 
            Your task is to answer questions about these emails based on the provided context.
            Guidelines:
            1. Base your answers ONLY on the information provided in the context
            2. If the context doesn't contain relevant information, say so
            3. Be specific and cite email IDs when referencing particular emails
            4. Keep responses concise but informative
            5. Focus on actionable insights and patterns
            6. If you notice any critical issues, highlight them
            7. Format your response in clear, readable sections if appropriate
            8. For 'show all emails' queries, list each email with its ID, subject, incident type, and severity"""},
            {"role": "user", "content": f"""Context from analyzed emails:
            {context}
            
            Question: {query_text}
            
            Please provide a clear, concise answer based on the email context above."""}
        ]

        # Get response from selected model
        logger.info(f"Querying {model_name} with context")
        try:
            if model_name == 'groq' and groq_available:
                response = get_groq_response(messages, model_config)
                # Extract content from Groq response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
                else:
                    response_text = str(response)
            else:  # deepseek
                # Use the actual model name from session state for DeepSeek
                actual_model_name = st.session_state.deepseek_model
                logger.info(f"Using DeepSeek model: {actual_model_name}")
                response_text = get_deepseek_response(messages, model_config, actual_model_name)
            
            # Cache the response
            cache_query_result(query_text, response_text, len(context.split('\n')), model_name)
            return response_text
            
        except Exception as e:
            if model_name == 'groq':
                logger.error(f"Error with Groq API: {str(e)}")
                logger.info("Falling back to DeepSeek")
                # Try DeepSeek as fallback
                try:
                    actual_model_name = st.session_state.deepseek_model
                    logger.info(f"Using DeepSeek fallback model: {actual_model_name}")
                    response_text = get_deepseek_response(messages, model_config, actual_model_name)
                    # Cache the response with the fallback model
                    cache_query_result(query_text, response_text, len(context.split('\n')), 'deepseek')
                    return response_text
                except Exception as fallback_error:
                    logger.error(f"Fallback to DeepSeek also failed: {str(fallback_error)}")
                    raise
            else:
                raise
            
    except Exception as e:
        logger.error(f"Error querying LLM with context: {str(e)}")
        return f"Error processing your query: {str(e)}"

def get_db_size():
    """Get the current size of the database file"""
    try:
        if os.path.exists(DB_PATH):  # Use global DB_PATH
            # Force a flush and checkpoint of any pending writes
            if 'conn' in st.session_state and st.session_state.conn is not None:
                try:
                    # Force checkpoint and close all connections
                    st.session_state.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    st.session_state.conn.execute("PRAGMA optimize")
                    st.session_state.conn.close()
                    st.session_state.conn = None
                except:
                    pass
            
            # Small delay to ensure file handles are released
            time.sleep(0.1)
            
            # Get the file size
            size_bytes = os.path.getsize(DB_PATH)  # Use global DB_PATH
            
            # Reconnect after getting size
            st.session_state.conn = duckdb.connect(DB_PATH)  # Use global DB_PATH
            
            return size_bytes
        return 0
    except Exception as e:
        logger.error(f"Error getting database size: {str(e)}")
        return 0

# Add these functions before the UI code, in the correct order
def reinitialize_database():
    """Force reinitialization of the database more efficiently"""
    try:
        # Close current connection
        if 'conn' in st.session_state and st.session_state.conn is not None:
            try:
                st.session_state.conn.execute("PRAGMA force_checkpoint")
                st.session_state.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")
                try:
                    st.session_state.conn.close()
                    logger.info("Database connection closed (fallback)")
                except Exception as close_error:
                    logger.warning(f"Error in fallback connection close: {str(close_error)}")
            finally:
                st.session_state.conn = None
        
        # Delete the database file
        if os.path.exists(DB_PATH):
            try:
                # Find and terminate processes holding the file
                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    try:
                        for file in proc.open_files():
                            if DB_PATH in file.path:
                                logger.info(f"Found process {proc.pid} ({proc.name()}) holding database file")
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                    logger.info(f"Process {proc.pid} terminated gracefully")
                                except psutil.TimeoutExpired:
                                    logger.warning(f"Process {proc.pid} did not terminate gracefully, force killing...")
                                    proc.kill()
                                    proc.wait(timeout=1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        logger.warning(f"Error checking process {proc.pid}: {str(e)}")
                        continue
                
                time.sleep(0.5)
                os.remove(DB_PATH)
                logger.info(f"Deleted database file {DB_PATH}")
            except Exception as e:
                logger.error(f"Error deleting database file: {str(e)}")
                return False
        
        # Create new connection and initialize
        try:
            st.session_state.conn = duckdb.connect(DB_PATH)
            if not init_database():
                logger.error("Failed to reinitialize database schema")
                return False
            logger.info("Database reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating new database connection: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error reinitializing database: {str(e)}")
        return False

def recategorize_emails():
    """Recategorize all emails in the database"""
    try:
        conn = get_db_connection()
        # Get all emails that need categorization
        emails = conn.execute('''
            SELECT id, email_subject, email_text_body 
            FROM emails 
            WHERE incident_type = 'Others' OR severity = 'Low'
        ''').fetchall()
        
        if not emails:
            logger.info("No emails need recategorization")
            return 0, 0
        
        success_count = 0
        error_count = 0
        
        for email_id, subject, body in emails:
            try:
                # Analyze the email
                analysis = analyze_email_content(subject, body)
                
                # Update the database
                conn.execute('''
                    UPDATE emails 
                    SET incident_type = ?, severity = ?
                    WHERE id = ?
                ''', (analysis['incident_type'], analysis['severity'], email_id))
                
                success_count += 1
                logger.info(f"Recategorized email {email_id}: {analysis['incident_type']} - {analysis['severity']}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error recategorizing email {email_id}: {str(e)}")
        
        return success_count, error_count
    except Exception as e:
        logger.error(f"Error in recategorization process: {str(e)}")
        return 0, 0

def add_reinitialize_button():
    """Add a button to reinitialize the database"""
    if st.button('🔄 Reinitialize Database', type='secondary'):
        if reinitialize_database():
            st.success('Database reinitialized successfully!')
            st.rerun()
        else:
            st.error('Failed to reinitialize database. Check logs for details.')

def add_recategorize_button():
    """Add a button to recategorize emails"""
    if st.button('🔄 Recategorize Emails', type='secondary', help='Reanalyze and recategorize all emails'):
        with st.spinner('Recategorizing emails...'):
            success, errors = recategorize_emails()
            if success > 0:
                st.success(f'✅ Recategorized {success} emails successfully!')
            if errors > 0:
                st.error(f'❌ Failed to recategorize {errors} emails')
            if success == 0 and errors == 0:
                st.info('No emails needed recategorization')

def get_total_analyzed_emails():
    """Get the total number of analyzed emails from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": {
                            "term": {
                                "is_analyzed": True
                            }
                        }
                    }
                }
            }
        )
        return results['hits']['total']['value']
    except Exception as e:
        logger.error(f"Error getting total analyzed emails: {str(e)}")
        return 0

def determine_relevant_categories(query_text):
    """Determine which analysis categories are relevant to the user's query"""
    query_lower = query_text.lower()
    
    # Define category keywords
    category_keywords = {
        'procedural_deviations': [
            'procedure', 'process', 'protocol', 'deviation', 'mistake', 'error',
            'procedure', 'step', 'method', 'approach', 'guideline', 'standard',
            'violation', 'breach', 'non-compliance', 'procedure', 'process'
        ],
        'recurrence_indicators': [
            'recur', 'repeat', 'pattern', 'frequency', 'often', 'repeatedly',
            'recurring', 'consistent', 'regular', 'cycle', 'trend', 'pattern',
            'frequent', 'repetitive', 'recurring', 'repeated'
        ],
        'systemic_trends': [
            'system', 'trend', 'overall', 'general', 'broad', 'widespread',
            'systemic', 'across', 'throughout', 'common', 'universal', 'global',
            'trend', 'pattern', 'theme', 'systemic'
        ]
    }
    
    # Count keyword matches for each category
    category_scores = {
        'procedural_deviations': sum(1 for word in category_keywords['procedural_deviations'] if word in query_lower),
        'recurrence_indicators': sum(1 for word in category_keywords['recurrence_indicators'] if word in query_lower),
        'systemic_trends': sum(1 for word in category_keywords['systemic_trends'] if word in query_lower)
    }
    
    # If no specific category is detected, return all categories
    if sum(category_scores.values()) == 0:
        return ['procedural_deviations', 'recurrence_indicators', 'systemic_trends']
    
    # Return categories that have at least one keyword match
    return [category for category, score in category_scores.items() if score > 0]

def clear_emails_table():
    """Clear all emails from the database and vector store"""
    try:
        # First clear the OpenSearch index to release any file handles
        clear_vector_store()
        
        # Get current connection
        conn = st.session_state.conn
        if conn is not None:
            try:
                # Close connection properly without parameters
                conn.execute("PRAGMA force_checkpoint")  # Remove TRUNCATE parameter
                conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")
                # Try to close without PRAGMA if it fails
                try:
                    conn.close()
                    logger.info("Database connection closed (fallback)")
                except Exception as close_error:
                    logger.warning(f"Error in fallback connection close: {str(close_error)}")
            st.session_state.conn = None  # Clear session state connection
        
        # Small delay to ensure connections are fully closed
        time.sleep(0.5)
        
        # Try to delete the database file with improved process handling
        max_retries = 3
        retry_delay = 1.0
        db_path = 'emails.db'
        
        for attempt in range(max_retries):
            try:
                if os.path.exists(db_path):
                    # Only check Python and Streamlit processes that might be using the file
                    target_processes = []
                    current_pid = os.getpid()
                    
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            # Skip current process
                            if proc.pid == current_pid:
                                continue
                                
                            # Only check Python and Streamlit processes
                            if proc.name().lower() in ['python.exe', 'pythonw.exe', 'streamlit.exe']:
                                cmdline = proc.cmdline()
                                # Check if this is a process running our app
                                if any('app.py' in cmd.lower() for cmd in cmdline):
                                    target_processes.append(proc)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                    
                    # Try to terminate target processes
                    for proc in target_processes:
                        try:
                            logger.info(f"Found process {proc.pid} ({proc.name()}) that might be using the database")
                            # Try graceful termination first
                            proc.terminate()
                            try:
                                proc.wait(timeout=2)  # Wait up to 2 seconds
                                logger.info(f"Process {proc.pid} terminated gracefully")
                            except psutil.TimeoutExpired:
                                # Force kill if graceful termination fails
                                logger.warning(f"Process {proc.pid} did not terminate gracefully, force killing...")
                                proc.kill()
                                proc.wait(timeout=1)
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"Could not terminate process {proc.pid}: {str(e)}")
                    
                    # Additional delay after process termination
                    time.sleep(0.5)
                    
                    # Now try to delete the file
                    os.remove(db_path)
                    logger.info("Database file deleted")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed to delete database: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to delete database after {max_retries} attempts: {str(e)}")
                    return False
        
        # Reinitialize the database
        try:
            new_conn = init_database()
            if new_conn is None:
                logger.error("Failed to reinitialize database")
                return False
            # Update session state only after successful initialization
            st.session_state.conn = new_conn
            logger.info("Database reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error reinitializing database: {str(e)}")
            # Ensure connection is None if initialization fails
            st.session_state.conn = None
            return False
            
    except Exception as e:
        logger.error(f"Error clearing emails table: {str(e)}")
        # Ensure connection is None on any error
        st.session_state.conn = None
        return False

def update_email_embeddings(email_ids=None):
    """Update embeddings for specified emails or all emails if no IDs provided"""
    try:
        conn = get_db_connection()
        
        # Get emails to update
        if email_ids:
            query = '''
                SELECT id, email_subject, email_text_body, incident_type, severity
                FROM emails 
                WHERE id IN ({})
            '''.format(','.join('?' * len(email_ids)))
            emails = conn.execute(query, email_ids).fetchall()
        else:
            emails = conn.execute('''
                SELECT id, email_subject, email_text_body, incident_type, severity
                FROM emails 
                ORDER BY id DESC
            ''').fetchall()
        
        if not emails:
            logger.info("No emails to update")
            return 0, 0
        
        success_count = 0
        error_count = 0
        
        # Process in smaller batches to avoid overwhelming the LLM
        batch_size = 5  # Reduced batch size for better reliability
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            try:
                # Prepare batch for summarization
                batch_data = [(id, body, subject) for id, subject, body, _, _ in batch]
                
                # Generate summaries with retry logic
                max_retries = 3
                summaries = None
                for attempt in range(max_retries):
                    try:
                        summaries = summarize_emails_bulk(batch_data, st.session_state.deepseek_model)
                        # Verify summaries were generated properly
                        if all(summaries.get(id) and len(summaries[id].strip()) > 0 for id, _, _ in batch_data):
                            break
                        logger.warning(f"Attempt {attempt + 1}: Some summaries were empty, retrying...")
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1)  # Wait before retry
                
                if not summaries:
                    raise Exception("Failed to generate summaries after all retries")
                
                # Prepare metadata for OpenSearch
                metadatas = []
                documents = []
                ids = []
                
                for id, subject, body, inc_type, sev in batch:
                    summary = summaries.get(id)
                    if not summary or not summary.strip():
                        # Generate a basic summary if LLM summary failed
                        summary = f"Subject: {subject}\nIncident Type: {inc_type}\nSeverity: {sev}\nSummary: Critical issue related to {inc_type.lower()} with {sev.lower()} severity."
                        logger.warning(f"Using basic summary for email {id}")
                    
                    current_time = datetime.now()
                    metadatas.append({
                        'email_id': id,
                        'subject': subject,
                        'incident_type': inc_type,
                        'severity': sev,
                        'created_at': current_time.isoformat(),
                        'created_at_timestamp': int(current_time.timestamp()),
                        'is_analyzed': True,
                        'summary_type': 'llm' if summaries.get(id) else 'basic'
                    })
                    documents.append(summary)
                    ids.append(str(id))
                
                # Update OpenSearch
                try:
                    # Delete existing entries
                    opensearch_client.delete_by_query(
                        index=OPENSEARCH_INDEX_NAME,
                        body={
                            "query": {
                                "bool": {
                                    "filter": {
                                        "terms": {
                                            "email_id": ids
                                        }
                                    }
                                }
                            }
                        }
                    )
                    
                    # Add updated entries
                    for id, subject, summary, inc_type, sev in metadatas:
                        opensearch_client.index(
                            index=OPENSEARCH_INDEX_NAME,
                            body={  # Changed from 'document' to 'body'
                                'subject': subject,
                                'incident_type': inc_type,
                                'severity': sev,
                                'created_at': created_at,
                                'created_at_timestamp': created_at_timestamp,
                                'is_analyzed': is_analyzed,
                                'summary': summary,
                                'summary_vector': st.session_state.embedding_model.encode([summary]).tolist()
                            },
                            refresh=True
                        )
                    success_count += len(batch)
                    logger.info(f"Updated embeddings for batch of {len(batch)} emails")
                except Exception as e:
                    error_count += len(batch)
                    logger.error(f"Error updating OpenSearch for batch: {str(e)}")
                
                # Small delay between batches to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                error_count += len(batch)
                logger.error(f"Error processing batch: {str(e)}")
                continue
        
        # Log final results
        if success_count > 0:
            logger.info(f"Successfully updated {success_count} email embeddings")
        if error_count > 0:
            logger.warning(f"Failed to update {error_count} email embeddings")
        
        return success_count, error_count
    except Exception as e:
        logger.error(f"Error in update_email_embeddings: {str(e)}")
        return 0, 0

def clear_vector_store():
    """Clear the OpenSearch index more efficiently"""
    try:
        # Delete and recreate index in one operation
        opensearch_client.indices.delete(index=OPENSEARCH_INDEX_NAME)
        opensearch_client.indices.create(
            index=OPENSEARCH_INDEX_NAME,
            body=OPENSEARCH_INDEX_SETTINGS
        )
        logger.info("OpenSearch index cleared and recreated")
        return True
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        return False

def get_vector_store_stats():
    """Get statistics about the OpenSearch index"""
    try:
        count = opensearch_client.count(index=OPENSEARCH_INDEX_NAME)
        return {
            'total_documents': count,
            'collection_name': OPENSEARCH_INDEX_NAME,
            'embedding_function': 'SentenceTransformer'
        }
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        return None

def recreate_opensearch_index():
    """Recreate the OpenSearch index with correct settings"""
    try:
        # Delete existing index if it exists
        if st.session_state.opensearch_client.indices.exists(index=OPENSEARCH_INDEX_NAME):
            logger.info(f"Deleting existing index: {OPENSEARCH_INDEX_NAME}")
            st.session_state.opensearch_client.indices.delete(index=OPENSEARCH_INDEX_NAME)
        
        # Create new index with correct settings
        logger.info("Creating new index with updated settings")
        st.session_state.opensearch_client.indices.create(
            index=OPENSEARCH_INDEX_NAME,
            body=OPENSEARCH_INDEX_SETTINGS
        )
        logger.info("Successfully recreated OpenSearch index")
        return True
    except Exception as e:
        logger.error(f"Error recreating OpenSearch index: {str(e)}")
        return False

# Add these functions to the UI section
def add_vector_store_management():
    """Add vector store management controls to the UI"""
    st.markdown("### Vector Store Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('🔄 Update All Embeddings', type='secondary'):
            with st.spinner('Updating embeddings...'):
                success, errors = update_email_embeddings()
                if success > 0:
                    st.success(f'✅ Updated {success} embeddings successfully!')
                if errors > 0:
                    st.error(f'❌ Failed to update {errors} embeddings')
    
    with col2:
        if st.button('🗑️ Clear Vector Store', type='secondary'):
            if st.checkbox('I understand this will delete all vector embeddings'):
                if clear_vector_store():
                    st.success('✅ Vector store cleared successfully!')
                    st.rerun()
                else:
                    st.error('❌ Failed to clear vector store')
            else:
                st.warning('Please confirm that you understand this action cannot be undone')
    
    with col3:
        if st.button('🔄 Recreate Index', type='secondary'):
            if st.checkbox('I understand this will recreate the index with correct settings'):
                if recreate_opensearch_index():
                    st.success('✅ Index recreated successfully!')
                    st.rerun()
                else:
                    st.error('❌ Failed to recreate index')
            else:
                st.warning('Please confirm that you understand this action cannot be undone')
    
    # Show vector store statistics
    stats = get_vector_store_stats()
    if stats:
        st.markdown("#### Vector Store Statistics")
        st.markdown(f"""
        - Total Documents: {stats['total_documents']}
        - Collection Name: {stats['collection_name']}
        - Embedding Function: {stats['embedding_function']}
        """)

def get_incident_type_distribution():
    """Get distribution of incident types from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "incident_types": {
                        "terms": {
                            "field": "incident_type",
                            "size": 10
                        }
                    }
                }
            }
        )
        
        buckets = results['aggregations']['incident_types']['buckets']
        return {bucket['key']: bucket['doc_count'] for bucket in buckets}
    except Exception as e:
        logger.error(f"Error getting incident type distribution: {str(e)}")
        return {}

def get_severity_distribution():
    """Get distribution of severity levels from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "severity_levels": {
                        "terms": {
                            "field": "severity",
                            "size": 10
                        }
                    }
                }
            }
        )
        
        buckets = results['aggregations']['severity_levels']['buckets']
        return {bucket['key']: bucket['doc_count'] for bucket in buckets}
    except Exception as e:
        logger.error(f"Error getting severity distribution: {str(e)}")
        return {}

def get_trends_over_time():
    """Get trends of incidents over time from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "trends_over_time": {
                        "date_histogram": {
                            "field": "created_at",
                            "calendar_interval": "day"
                        },
                        "aggs": {
                            "incident_types": {
                                "terms": {
                                    "field": "incident_type",
                                    "size": 10
                                }
                            }
                        }
                    }
                }
            }
        )
        
        buckets = results['aggregations']['trends_over_time']['buckets']
        return buckets
    except Exception as e:
        logger.error(f"Error getting trends over time: {str(e)}")
        return []

def get_incident_correlations():
    """Get correlations between incident types and severity levels from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "incident_types": {
                        "terms": {
                            "field": "incident_type",
                            "size": 10
                        },
                        "aggs": {
                            "severity_distribution": {
                                "terms": {
                                    "field": "severity",
                                    "size": 10
                                }
                            }
                        }
                    }
                }
            }
        )
        
        buckets = results['aggregations']['incident_types']['buckets']
        correlations = {}
        for bucket in buckets:
            incident_type = bucket['key']
            severity_dist = {
                sub_bucket['key']: sub_bucket['doc_count']
                for sub_bucket in bucket['severity_distribution']['buckets']
            }
            correlations[incident_type] = severity_dist
        return correlations
    except Exception as e:
        logger.error(f"Error getting incident correlations: {str(e)}")
        return {}

def get_systemic_trends():
    """Get systemic trends from OpenSearch"""
    try:
        # Get all analyzed emails
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": {
                            "term": {
                                "is_analyzed": True
                            }
                        }
                    }
                },
                "size": 1000  # Adjust based on your needs
            }
        )
        
        # Process results to identify trends
        trends = {}
        for hit in results['hits']['hits']:
            incident_type = hit['_source']['incident_type']
            severity = hit['_source']['severity']
            
            if incident_type not in trends:
                trends[incident_type] = {
                    'count': 0,
                    'severity_counts': {},
                    'recent_count': 0
                }
            
            trends[incident_type]['count'] += 1
            trends[incident_type]['severity_counts'][severity] = trends[incident_type]['severity_counts'].get(severity, 0) + 1
            
            # Check if this is a recent incident (last 7 days)
            created_at = datetime.fromisoformat(hit['_source']['created_at'])
            if (datetime.now() - created_at).days <= 7:
                trends[incident_type]['recent_count'] += 1
        
        return trends
    except Exception as e:
        logger.error(f"Error getting systemic trends: {str(e)}")
        return {}

def get_incident_details(incident_type):
    """Get detailed information about a specific incident type from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": {
                            "term": {
                                "incident_type": incident_type
                            }
                        }
                    }
                },
                "size": 100,  # Adjust based on your needs
                "sort": [
                    {"created_at": {"order": "desc"}}
                ]
            }
        )
        
        incidents = []
        for hit in results['hits']['hits']:
            source = hit['_source']
            incidents.append({
                'email_id': source['email_id'],
                'subject': source['subject'],
                'severity': source['severity'],
                'created_at': source['created_at'],
                'summary': source['summary']
            })
        
        return incidents
    except Exception as e:
        logger.error(f"Error getting incident details: {str(e)}")
        return []

def get_severity_details(severity):
    """Get detailed information about a specific severity level from OpenSearch"""
    try:
        results = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={
                "query": {
                    "bool": {
                        "filter": {
                            "term": {
                                "severity": severity
                            }
                        }
                    }
                },
                "size": 100,  # Adjust based on your needs
                "sort": [
                    {"created_at": {"order": "desc"}}
                ]
            }
        )
        
        incidents = []
        for hit in results['hits']['hits']:
            source = hit['_source']
            incidents.append({
                'email_id': source['email_id'],
                'subject': source['subject'],
                'incident_type': source['incident_type'],
                'created_at': source['created_at'],
                'summary': source['summary']
            })
        
        return incidents
    except Exception as e:
        logger.error(f"Error getting severity details: {str(e)}")
        return []

# Add after the model configurations and before the analysis functions
def get_model_response(messages, model_size='small'):
    """Get response from the configured model provider with DeepSeek as primary and Anthropic as fallback"""
    try:
        # Initialize DeepSeek model in session state if not present
        if 'deepseek_model' not in st.session_state:
            try:
                available_models = get_available_deepseek_models()
                if available_models:
                    st.session_state.deepseek_model = available_models[0]
                    logger.info(f"Initialized DeepSeek model to: {st.session_state.deepseek_model}")
                else:
                    st.session_state.deepseek_model = DEEPSEEK_MODEL
                    logger.info(f"No available DeepSeek models found, using default: {DEEPSEEK_MODEL}")
            except Exception as e:
                logger.error(f"Error initializing DeepSeek model: {str(e)}")
                st.session_state.deepseek_model = DEEPSEEK_MODEL
                logger.info(f"Using default DeepSeek model: {DEEPSEEK_MODEL}")

        # For email analysis, try DeepSeek first
        try:
            # Get DeepSeek config
            deepseek_config = ANALYSIS_MODEL_CONFIG[model_size]['deepseek'].copy()
            # Ensure max_tokens is set and within limits
            if 'max_tokens' not in deepseek_config or not isinstance(deepseek_config['max_tokens'], int):
                deepseek_config['max_tokens'] = 1024 if model_size == 'small' else 2048
            # Cap max_tokens to ensure we don't exceed limits
            deepseek_config['max_tokens'] = min(deepseek_config['max_tokens'], 2048)
            
            logger.info(f"Attempting to use DeepSeek model for {model_size} task with max_tokens={deepseek_config['max_tokens']}")
            actual_model_name = st.session_state.deepseek_model
            return get_deepseek_response(messages, deepseek_config, actual_model_name)
            
        except Exception as deepseek_error:
            logger.error(f"DeepSeek model failed: {str(deepseek_error)}")
            # Fallback to Anthropic
            try:
                # Get Anthropic config
                anthropic_config = ANALYSIS_MODEL_CONFIG[model_size]['anthropic'].copy()
                # Ensure max_tokens is set and within limits
                if 'max_tokens' not in anthropic_config or not isinstance(anthropic_config['max_tokens'], int):
                    anthropic_config['max_tokens'] = 1024 if model_size == 'small' else 2048
                anthropic_config['max_tokens'] = min(anthropic_config['max_tokens'], 2048)
                
                # Ensure all required fields are present
                required_fields = ['name', 'max_tokens', 'temperature']
                for field in required_fields:
                    if field not in anthropic_config:
                        if field == 'name':
                            anthropic_config[field] = ANTHROPIC_MODEL
                        elif field == 'temperature':
                            anthropic_config[field] = 0.2
                        elif field == 'max_tokens':
                            anthropic_config[field] = 1024 if model_size == 'small' else 2048
                
                logger.info(f"Falling back to Anthropic model for {model_size} task with max_tokens={anthropic_config['max_tokens']}")
                return get_anthropic_response(messages, anthropic_config)
                
            except Exception as anthropic_error:
                raise Exception(f"Both DeepSeek and Anthropic failed. DeepSeek error: {str(deepseek_error)}, Anthropic error: {str(anthropic_error)}")
                
    except Exception as e:
        logger.error(f"Error in get_model_response: {str(e)}")
        raise

def get_anthropic_response(messages, model_config):
    """Get response from Anthropic Claude API"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Convert OpenAI-style messages to Anthropic format
        system_prompt = ""
        user_content = ""
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_content += msg['content'] + '\n'
        
        response = client.messages.create(
            model=model_config['name'],
            max_tokens=model_config['max_tokens'],
            temperature=model_config.get('temperature', 0.2),
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        
        # Handle the response based on its type
        if hasattr(response, 'content'):
            # For newer Anthropic API versions
            if isinstance(response.content, list):
                # Extract text from the first content block
                for block in response.content:
                    if block.type == 'text':
                        return block.text
                return ''  # Return empty string if no text block found
            # For older API versions or direct text content
            return str(response.content)
        # Fallback to string representation if content attribute not found
        return str(response)
    except Exception as e:
        logger.error(f"Error calling Anthropic API: {str(e)}")
        raise

# Streamlit UI
st.title('📧 Email Analysis Dashboard')

# Create tabs for different sections
tab1, tab2 = st.tabs(["📨 Email Analysis", "📊 Analytics"])

with tab1:
    # Original content will go here
    pass

with tab2:
    st.header("📈 Email Analytics Dashboard")
    
    # Load email data from OpenSearch
    try:
        # Get incident type distribution
        incident_dist = get_incident_type_distribution()
        if incident_dist:
            st.subheader("Incident Type Distribution")
            fig = px.pie(
                values=list(incident_dist.values()),
                names=list(incident_dist.keys()),
                title="Distribution of Incident Types"
            )
            st.plotly_chart(fig)
        
        # Get severity distribution
        severity_dist = get_severity_distribution()
        if severity_dist:
            st.subheader("Severity Level Distribution")
            fig = px.bar(
                x=list(severity_dist.keys()),
                y=list(severity_dist.values()),
                title="Distribution of Severity Levels",
                labels={'x': 'Severity Level', 'y': 'Count'}
            )
            st.plotly_chart(fig)
        
        # Get trends over time
        trends = get_trends_over_time()
        if trends:
            st.subheader("Incident Trends Over Time")
            # Convert to DataFrame for easier plotting
            trend_data = []
            for bucket in trends:
                date = bucket['key_as_string']
                for incident in bucket['incident_types']['buckets']:
                    trend_data.append({
                        'date': date,
                        'incident_type': incident['key'],
                        'count': incident['doc_count']
                    })
            
            if trend_data:
                df = pd.DataFrame(trend_data)
                fig = px.line(
                    df,
                    x='date',
                    y='count',
                    color='incident_type',
                    title="Incident Trends Over Time",
                    labels={'date': 'Date', 'count': 'Number of Incidents'}
                )
                st.plotly_chart(fig)
        
        # Get incident correlations
        correlations = get_incident_correlations()
        if correlations:
            st.subheader("Incident Type vs Severity Correlations")
            # Convert to DataFrame for easier plotting
            corr_data = []
            for incident_type, severity_dist in correlations.items():
                for severity, count in severity_dist.items():
                    corr_data.append({
                        'incident_type': incident_type,
                        'severity': severity,
                        'count': count
                    })
            
            if corr_data:
                df = pd.DataFrame(corr_data)
                fig = px.bar(
                    df,
                    x='incident_type',
                    y='count',
                    color='severity',
                    title="Incident Type vs Severity Distribution",
                    labels={'incident_type': 'Incident Type', 'count': 'Count', 'severity': 'Severity Level'}
                )
                st.plotly_chart(fig)
        
        # Get systemic trends
        systemic_trends = get_systemic_trends()
        if systemic_trends:
            st.subheader("Systemic Trends Analysis")
            
            # Create a DataFrame for the trends
            trend_data = []
            for incident_type, data in systemic_trends.items():
                trend_data.append({
                    'incident_type': incident_type,
                    'total_count': data['count'],
                    'recent_count': data['recent_count'],
                    'trend': 'Increasing' if data['recent_count'] > data['count'] / 4 else 'Stable'
                })
            
            if trend_data:
                df = pd.DataFrame(trend_data)
                
                # Display trend summary
                st.write("### Trend Summary")
                for _, row in df.iterrows():
                    st.write(f"**{row['incident_type']}**:")
                    st.write(f"- Total Incidents: {row['total_count']}")
                    st.write(f"- Recent Incidents (Last 7 days): {row['recent_count']}")
                    st.write(f"- Trend: {row['trend']}")
                    st.write("---")
                
                # Plot trend comparison
                fig = px.bar(
                    df,
                    x='incident_type',
                    y=['total_count', 'recent_count'],
                    title="Total vs Recent Incidents by Type",
                    labels={'value': 'Count', 'variable': 'Time Period'},
                    barmode='group'
                )
                st.plotly_chart(fig)
        
        # Detailed Analysis Section
        st.subheader("Detailed Analysis")
        
        # Incident Type Details
        selected_incident = st.selectbox(
            "Select Incident Type for Details",
            options=list(incident_dist.keys()) if incident_dist else []
        )
        
        if selected_incident:
            incidents = get_incident_details(selected_incident)
            if incidents:
                st.write(f"### Details for {selected_incident}")
                for incident in incidents:
                    with st.expander(f"{incident['subject']} ({incident['severity']})"):
                        st.write(f"**Date:** {incident['created_at']}")
                        st.write(f"**Summary:** {incident['summary']}")
        
        # Severity Level Details
        selected_severity = st.selectbox(
            "Select Severity Level for Details",
            options=list(severity_dist.keys()) if severity_dist else []
        )
        
        if selected_severity:
            incidents = get_severity_details(selected_severity)
            if incidents:
                st.write(f"### Details for {selected_severity} Severity Incidents")
                for incident in incidents:
                    with st.expander(f"{incident['subject']} ({incident['incident_type']})"):
                        st.write(f"**Date:** {incident['created_at']}")
                        st.write(f"**Summary:** {incident['summary']}")
        
    except Exception as e:
        logger.error(f"Error in analytics dashboard: {str(e)}")
        st.error("An error occurred while loading the analytics dashboard. Please check the logs for details.")

# Create three main columns for the original content
with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])

    # Left Column - Insights
    with col1:
        st.subheader('🔍 Process Analysis')
        
        # Add options for analysis
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            include_analyzed = st.checkbox('Include previously analyzed emails', value=False)
        with col1_2:
            batch_size = st.number_input('Batch Size', min_value=5, max_value=20, value=10, step=5)
        
        # Add similarity batching options
        col1_3, col1_4 = st.columns(2)
        with col1_3:
            use_similarity = st.checkbox('Use Similarity Batching', value=False)
        with col1_4:
            similarity_threshold = st.slider(
                'Similarity Threshold',
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.1,
                help='Higher values mean emails need to be more similar to be grouped together'
            )
        
        if st.button('Update Insights'):
            with st.spinner('Analyzing...'):
                insights = get_insights(
                    include_analyzed=include_analyzed,
                    batch_size=batch_size,
                    use_similarity_batching=use_similarity,
                    similarity_threshold=similarity_threshold
                )
                st.session_state['insights'] = insights
                logger.info("Insights updated in session state")

        if 'insights' in st.session_state:
            st.markdown(f"**Last Updated:** {st.session_state['insights']['last_updated']}")
            
            # Display analysis statistics
            stats = st.session_state['insights']['analysis_stats']
            st.markdown("### 📊 Analysis Statistics")
            st.markdown(f"""
            - Total Emails: {stats.get('total_emails', 0)}
            - Emails Analyzed: {stats.get('analyzed_emails', 0)}
            - Emails Referenced: {stats.get('referenced_emails', 0)}
            - Batches Processed: {stats.get('batches_processed', 0)}
            - Batching Method: {stats.get('batching_method', 'N/A')}
            """)
            
            with st.expander("Procedural Deviations", expanded=True):
                st.write(st.session_state['insights']['procedural_deviations'])
            
            with st.expander("Recurrence Indicators", expanded=True):
                st.write(st.session_state['insights']['recurrence_indicators'])
            
            with st.expander("Systemic Trends", expanded=True):
                st.write(st.session_state['insights']['systemic_trends'])

    # Middle Column - Data Management
    with col2:
        st.subheader('📥 Data Management')
        
        # Import Section
        bulk_email_json = st.text_area(
            label="Paste email JSON",
            height=100,
            key="bulk_email_json"
        )
        
        if st.button('Import Emails'):
            if bulk_email_json:
                try:
                    emails_data = json.loads(bulk_email_json)
                    if not isinstance(emails_data, list):
                        emails_data = [emails_data]
                    
                    with st.spinner('Importing emails...'):
                        success_count, error_count, error_messages = store_multiple_emails(emails_data)
                        st.success(f'✅ Imported {success_count} emails!')
                        if error_count > 0:
                            st.error(f'❌ Failed to import {error_count} emails')
                            for error in error_messages:
                                st.error(error)
                except Exception as e:
                    st.error(f'Error processing import: {str(e)}')
            else:
                st.warning('Please paste email JSON data to import.')
        
        # Clear Section
        if st.button('🗑️ Clear All Data', type='primary', help='Warning: This will permanently delete all emails and analysis data'):
                if clear_emails_table():
                    st.success('✅ All data cleared successfully!')
                    st.rerun()
                else:
                    st.error('❌ Failed to clear data')
        
        add_reinitialize_button()  # Now this will work because the function is defined above
        
        # Show Analysis Results Button
        if st.button('📊 Show Analysis Results'):
            try:
                # Query analysis results joined with emails
                query = '''
                    SELECT 
                        ar.email_id,
                        e.email_subject,
                        ar.procedural_deviations,
                        ar.recurrence_indicators,
                        ar.systemic_trends,
                        ar.created_at as analysis_date
                    FROM email_analysis ar
                    JOIN emails e ON ar.email_id = e.id
                    ORDER BY ar.created_at DESC
                '''
                analysis_results = conn.execute(query).fetchdf()

                if not analysis_results.empty:
                    # Format datetime column
                    analysis_results['analysis_date'] = analysis_results['analysis_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    # Rename columns for better display
                    analysis_results = analysis_results.rename(columns={
                        'email_id': 'Email ID',
                        'email_subject': 'Subject',
                        'procedural_deviations': 'Procedural Deviations',
                        'recurrence_indicators': 'Recurrence Indicators',
                        'systemic_trends': 'Systemic Trends',
                        'analysis_date': 'Analysis Date'
                    })

                    # Display the analysis results table
                    st.dataframe(analysis_results, use_container_width=True)
                else:
                    st.info("No analysis results found in the database.")
            except Exception as e:
                logger.error(f"Error displaying analysis results: {str(e)}")
                st.error("Error displaying analysis results. Please check the logs for details.")

        # Add recategorize button to UI
        add_recategorize_button()  # Add this line after other buttons
        add_vector_store_management()  # Add this line

    # Right Column - RAG Query Interface
    with col3:
        st.subheader('🤖 AI Query Interface')
        
        # Create a container for model selection with a border
        with st.container():
            st.markdown("### Model Selection")
            model_choice = st.radio(
                "Select Query Model",
                options=['groq', 'deepseek'],
                format_func=lambda x: 'Groq (Fast)' if x == 'groq' else 'DeepSeek (High Quality)',
                help="Groq is faster but DeepSeek may provide more detailed analysis",
                horizontal=True  # Make it horizontal for better space usage
            )
            
            # Show model details in an expander
            with st.expander("Model Details", expanded=False):
                if model_choice == 'groq':
                    st.markdown(f"**Current Model:** {GROQ_MODEL}")
                    st.markdown("**Features:**")
                    st.markdown("- Fast response times")
                    st.markdown("- Good for quick analysis")
                    st.markdown("- Suitable for most queries")
                else:
                    st.markdown(f"**Current Model:** {DEEPSEEK_MODEL}")
                    st.markdown("**Features:**")
                    st.markdown("- High-quality responses")
                    st.markdown("- Better for complex analysis")
                    st.markdown("- More detailed insights")
        
        # Create a container for query input with a border
        with st.container():
            st.markdown("### Query Input")
            query = st.text_area(
                label="Ask a question about the emails",
                height=100,
                placeholder="Example: What are the most common maintenance issues reported?",
                key="rag_query"
            )
            
            # Get total analyzed emails for dynamic slider
            total_analyzed = get_total_analyzed_emails()
            max_emails = max(total_analyzed, 100)  # At least 100, or total analyzed if higher
            
            # Context size control with dynamic limits
            st.markdown("### Context Settings")
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                context_size = st.slider(
                    "Number of recent emails",
                    min_value=5,
                    max_value=max_emails,
                    value=min(20, max_emails),
                    step=5,
                    help="How many recent emails to include in the context"
                )
            with col3_2:
                days_back = st.slider(
                    "Days to look back",
                    min_value=1,
                    max_value=90,
                    value=30,
                    step=1,
                    help="How far back to look for relevant emails"
                )
        
        # Query button in its own container
        with st.container():
            if st.button('🔍 Analyze Emails', use_container_width=True):
                if query:
                    with st.spinner(f'Analyzing emails using {model_choice}...'):
                        # Get context from analyzed emails - pass st.session_state
                        context, num_emails = get_email_context(context_size, query, days_back=days_back, st_session=st.session_state)
                        
                        # Show context information in an expander
                        with st.expander("Context Information", expanded=False):
                            st.info(f"Using {num_emails} emails for context (requested: {context_size})")
                            st.markdown(f"Total analyzed emails available: {total_analyzed}")
                        
                        # Query LLM with context and selected model
                        response = query_llm_with_context(query, context, model_choice)
                        
                        # Store in session state
                        st.session_state['last_query'] = {
                            'query': query,
                            'response': response,
                            'model': model_choice,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'context_size': len(context.split('\n')),
                            'requested_size': context_size,
                            'total_available': total_analyzed,
                            'num_emails_used': num_emails,
                            'days_back': days_back
                        }
                else:
                    st.warning('Please enter a question to analyze.')
        
        # Display last query result in a clean container
        if 'last_query' in st.session_state:
            with st.container():
                st.markdown("### Analysis Results")
                
                # Query details in an expander
                with st.expander("Query Details", expanded=False):
                    st.markdown(f"**Timestamp:** {st.session_state['last_query']['timestamp']}")
                    st.markdown(f"**Model Used:** {st.session_state['last_query']['model'].upper()}")
                    st.markdown(f"**Context:** {st.session_state['last_query']['num_emails_used']} emails, {st.session_state['last_query']['days_back']} days back")
                
                # Display the query and response in a clean format
                st.markdown("#### Question")
                st.markdown(f"_{st.session_state['last_query']['query']}_")
                
                st.markdown("#### Response")
                # Ensure we're displaying the actual response text, not the object
                response_text = st.session_state['last_query']['response']
                if isinstance(response_text, str):
                    st.markdown(response_text)
                else:
                    # Handle case where response might be an object
                    try:
                        if hasattr(response_text, 'choices') and len(response_text.choices) > 0:
                            st.markdown(response_text.choices[0].message.content)
                        else:
                            st.markdown(str(response_text))
                    except Exception as e:
                        logger.error(f"Error formatting response: {str(e)}")
                        st.markdown("Error displaying response. Please try again.")

    # Footer with metrics - make it more compact
    st.markdown("---")
    st.markdown("### 📊 Dashboard Status")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        try:
            conn = get_db_connection()
            if conn:
                try:
                    count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
                    analyzed = conn.execute("SELECT COUNT(*) FROM emails WHERE is_analyzed = TRUE").fetchone()[0]
                    st.metric("Total Emails", count, f"{analyzed} analyzed")
                except Exception as e:
                    if "Table with name emails does not exist" in str(e):
                        st.metric("Total Emails", "0", "0 analyzed")
                    else:
                        logger.error(f"Error getting email count: {str(e)}")
                        st.metric("Total Emails", "Error")
            else:
                st.metric("Total Emails", "Error")
        except Exception as e:
            logger.error(f"Error getting email count: {str(e)}")
            st.metric("Total Emails", "Error")

    with status_col2:
        st.metric("Last Update", datetime.now().strftime('%H:%M:%S'))

    with status_col3:
        if st.button('📋 View Emails', use_container_width=True):
            try:
                conn = get_db_connection()
                query = '''
                    SELECT 
                        e.id,
                        e.email_subject,
                        e.email_to,
                        e.email_from,
                        e.incident_type,
                        e.severity,
                        e.email_text_body,
                        e.is_analyzed,
                        e.analyzed_at,
                        e.created_at
                    FROM emails e
                    ORDER BY e.id DESC
                '''
                emails = conn.execute(query).fetchdf()
                
                if not emails.empty:
                    # Format the dataframe for display
                    emails['analyzed_at'] = emails['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    emails['created_at'] = emails['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    emails['is_analyzed'] = emails['is_analyzed'].map({True: '✅', False: '❌'})
                    emails['email_text_body'] = emails['email_text_body'].apply(
                        lambda x: x[:100] + '...' if len(str(x)) > 100 else x
                    )
                    
                    # Rename and reorder columns
                    emails = emails.rename(columns={
                        'id': 'ID',
                        'email_subject': 'Subject',
                        'email_to': 'To',
                        'email_from': 'From',
                        'incident_type': 'Incident Type',
                        'severity': 'Severity',
                        'email_text_body': 'Body Preview',
                        'is_analyzed': 'Analyzed',
                        'analyzed_at': 'Analyzed At',
                        'created_at': 'Created At'
                    })
                    
                    # Display with better formatting
                    st.dataframe(
                        emails[['ID', 'Subject', 'To', 'From', 'Incident Type', 'Severity', 'Body Preview', 'Analyzed', 'Analyzed At', 'Created At']],
                        use_container_width=True,
                        column_config={
                            "Body Preview": st.column_config.TextColumn(
                                "Body Preview",
                                width="large",
                                help="First 100 characters of the email body"
                            ),
                            "Subject": st.column_config.TextColumn(
                                "Subject",
                                width="medium"
                            ),
                            "To": st.column_config.TextColumn(
                                "To",
                                width="medium"
                            ),
                            "From": st.column_config.TextColumn(
                                "From",
                                width="medium"
                            ),
                            "Incident Type": st.column_config.TextColumn(
                                "Incident Type",
                                width="small"
                            ),
                            "Severity": st.column_config.TextColumn(
                                "Severity",
                                width="small"
                            ),
                            "ID": st.column_config.NumberColumn(
                                "ID",
                                width="small"
                            ),
                            "Analyzed": st.column_config.TextColumn(
                                "Analyzed",
                                width="small"
                            ),
                            "Analyzed At": st.column_config.TextColumn(
                                "Analyzed At",
                                width="medium"
                            ),
                            "Created At": st.column_config.TextColumn(
                                "Created At",
                                width="medium"
                            )
                        }
                    )
                else:
                    st.info("No emails found in the database.")
            except Exception as e:
                logger.error(f"Error displaying table: {str(e)}")
                st.error("Error displaying table. Please check the logs for details.")

with status_col4:
    if st.button('🤖 Model Info', use_container_width=True):
        with st.expander("Available Models", expanded=True):
            # Show DeepSeek models
            st.markdown("**DeepSeek Models:**")
            deepseek_models = get_available_deepseek_models()
            if deepseek_models:
                for model in sorted(deepseek_models):
                    st.markdown(f"- {model}")
                st.markdown(f"**Current:** {DEEPSEEK_MODEL}")
            else:
                st.error("Could not fetch DeepSeek models")
            
            # Show Groq model
            st.markdown("**Groq Model:**")
            st.markdown(f"- {GROQ_MODEL}")

