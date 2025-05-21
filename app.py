import streamlit as st
import duckdb
import json
from datetime import datetime, timedelta
import os
from groq import Groq
import requests
from dotenv import load_dotenv
import logging
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from functools import partial
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables only once
if 'env_loaded' not in st.session_state:
    load_dotenv()
    st.session_state.env_loaded = True
    logger.info("Environment variables loaded")

# Model configurations
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')  # Default DeepSeek model
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

# Query model configurations
QUERY_MODEL_CONFIGS = {
    'deepseek': {
        'name': DEEPSEEK_MODEL,
        'max_tokens': int(os.getenv('DEEPSEEK_MAX_TOKENS', 8192)),
        'temperature': float(os.getenv('DEEPSEEK_TEMPERATURE', 0.5))
    },
    'groq': {
        'name': GROQ_MODEL,
        'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', 8192)),
        'temperature': float(os.getenv('GROQ_TEMPERATURE', 0.5)),
        'top_p': float(os.getenv('GROQ_TOP_P', 1)),
        'frequency_penalty': float(os.getenv('GROQ_FREQUENCY_PENALTY', 0))
    }
}

# Analysis model configuration (using DeepSeek)
ANALYSIS_MODEL_CONFIG = {
    'large': {
        'name': DEEPSEEK_MODEL,
        'max_tokens': 4096,
        'temperature': 0.3,
        'use_cases': ['complex_analysis', 'summarization', 'pattern_detection']
    },
    'small': {
        'name': DEEPSEEK_MODEL,
        'max_tokens': 2048,
        'temperature': 0.2,
        'use_cases': ['simple_queries', 'factual_lookup', 'basic_summary']
    }
}

# Add at the top with other global variables
DB_PATH = 'emails.db'  # Define database path globally

def get_db_connection():
    """Get or create database connection"""
    if 'conn' not in st.session_state:
        st.session_state.conn = duckdb.connect(DB_PATH)  # Use global DB_PATH
    return st.session_state.conn

def init_database():
    """Initialize database tables"""
    try:
        # Start a transaction
        conn = get_db_connection()
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Create emails table with only essential columns
            conn.execute('''
                CREATE TABLE IF NOT EXISTS emails (
                    id INTEGER PRIMARY KEY,
                    email_subject VARCHAR,
                    email_text_body TEXT,
                    is_analyzed BOOLEAN DEFAULT FALSE,
                    analyzed_at TIMESTAMP DEFAULT NULL
                )
            ''')

            # Create analysis results table with new column names
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results_new (
                    id INTEGER PRIMARY KEY,
                    email_id INTEGER,
                    procedural_deviations TEXT,
                    recurrence_indicators TEXT,
                    systemic_trends TEXT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (email_id) REFERENCES emails(id)
                )
            ''')

            # Drop and recreate query cache table with model information
            conn.execute('DROP TABLE IF EXISTS query_cache')
            conn.execute('''
                CREATE TABLE query_cache (
                    id INTEGER PRIMARY KEY,
                    query_text TEXT,
                    response_text TEXT,
                    context_size INTEGER,
                    model_used VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            # Commit the transaction
            conn.execute("COMMIT")
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            # Rollback on any error
            conn.execute("ROLLBACK")
            logger.error(f"Error initializing database schema: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in database transaction: {str(e)}")
        return False

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
    if not init_database():
        logger.error("Failed to initialize database")
        st.error("Failed to initialize database. Please check the logs for details.")
        st.stop()
    st.session_state.db_initialized = True
    logger.info("Database initialized")

# Initialize clients only once using session state
if 'clients_initialized' not in st.session_state:
    st.session_state.groq_client = Groq(
        api_key=os.getenv('GROQ_API_KEY')
    )
    st.session_state.clients_initialized = True
    
    # Get available DeepSeek models
    available_models = get_available_deepseek_models()
    if available_models:
        logger.info(f"Found {len(available_models)} available DeepSeek models")
        # Update DEEPSEEK_MODEL if the current one isn't available
        if DEEPSEEK_MODEL not in available_models and available_models:
            DEEPSEEK_MODEL = available_models[0]  # Use the first available model
            logger.info(f"Updated DeepSeek model to: {DEEPSEEK_MODEL}")
    else:
        logger.warning("Could not fetch available DeepSeek models")
    
    logger.info("Clients initialized")

# Use session state clients
groq_client = st.session_state.groq_client

# Use session state connection
conn = st.session_state.conn

# Increase context size limit for better coverage
MAX_CONTEXT_SIZE = 30000  # Increased from 15000
MAX_EMAILS_PER_CONTEXT = 50  # Increased from 16

def get_deepseek_model_config():
    """Get the DeepSeek model configuration, ensuring we use a valid model"""
    return {
        'name': DEEPSEEK_MODEL,  # Always use the configured DeepSeek model
        'max_tokens': int(os.getenv('DEEPSEEK_MAX_TOKENS', 8192)),
        'temperature': float(os.getenv('DEEPSEEK_TEMPERATURE', 0.5))
    }

def get_groq_model_config():
    """Get the Groq model configuration"""
    return {
        'name': GROQ_MODEL,
        'max_tokens': int(os.getenv('GROQ_MAX_TOKENS', 8192)),
        'temperature': float(os.getenv('GROQ_TEMPERATURE', 0.5)),
        'top_p': float(os.getenv('GROQ_TOP_P', 1)),
        'frequency_penalty': float(os.getenv('GROQ_FREQUENCY_PENALTY', 0))
    }

# Query model configurations - use functions to ensure fresh configs
QUERY_MODEL_CONFIGS = {
    'deepseek': get_deepseek_model_config,
    'groq': get_groq_model_config
}

def get_deepseek_response(messages, model_config):
    """Make a request to DeepSeek API directly"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Fixed API endpoint URL - ensure no duplicate /v1/
        api_url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
        logger.info(f"Making DeepSeek API request to {api_url}")
        
        # Always use the DeepSeek model
        model_name = DEEPSEEK_MODEL
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
    """Make a request to Groq API for queries"""
    try:
        response = groq_client.chat.completions.create(
            model=model_config['name'],
            messages=messages,
            temperature=model_config['temperature'],
            max_tokens=model_config['max_tokens'],
            top_p=model_config['top_p'],
            frequency_penalty=model_config['frequency_penalty']
        )
        return response.choices[0].message.content
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

def summarize_emails_bulk(batch):
    """Summarize multiple emails in a single LLM call - Uses DeepSeek API directly"""
    try:
        # Prepare context with all emails in the batch
        context = "Summarize these maintenance/service emails in ONE sentence each, focusing on the most critical issue or action needed:\n\n"
        for id, text_body, subject in batch:
            context += f"Email ID {id}:\nSubject: {subject}\nBody: {text_body}\n\n"
        
        # Use large model for summarization as it's a complex task
        model_config = select_model('summarization')
        # Reduce max tokens for summary
        model_config['max_tokens'] = min(model_config['max_tokens'], 500)
        
        # Use DeepSeek API directly for analysis
        logger.info("Using DeepSeek API for email summarization")
        response_text = get_deepseek_response(
            messages=[
                {"role": "system", "content": "You are an expert at summarizing maintenance and service-related emails. For each email, provide ONE concise sentence focusing on the most critical issue or action needed. Be specific but brief. Format your response with 'Email ID X:' before each summary."},
                {"role": "user", "content": context}
            ],
            model_config=model_config
        )
        
        # Parse the response to extract summaries
        summaries = {}
        response_text = response_text.strip()
        
        # Split response into individual email summaries
        email_blocks = response_text.split('Email ID')
        for block in email_blocks[1:]:  # Skip the first empty block
            try:
                id_str, summary = block.split(':', 1)
                email_id = int(id_str.strip())
                # Ensure summary is a single sentence
                summary = summary.strip().split('.')[0] + '.'
                summaries[email_id] = summary
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse summary for block: {block}")
                continue
        
        # Fill in any missing summaries with truncated text
        for id, text_body, subject in batch:
            if id not in summaries:
                logger.warning(f"Using fallback summary for email {id}")
                # Create a very brief fallback summary
                summaries[id] = f"Critical issue: {subject[:50]}..."
        
        return summaries
    except Exception as e:
        logger.error(f"Error in bulk summarization: {str(e)}")
        # Return very brief fallback summaries
        return {
            id: f"Critical issue: {subject[:50]}..."
            for id, text_body, subject in batch
        }

def select_model(query_type, query_text=None):
    """
    Select the appropriate model based on query type and complexity.
    For analysis tasks, returns DeepSeek model config.
    For queries, returns Groq model config.
    """
    if query_type in ['simple_queries', 'factual_lookup', 'basic_summary']:
        return ANALYSIS_MODEL_CONFIG['small'].copy()  # Return a copy to avoid modifying the original
    return ANALYSIS_MODEL_CONFIG['large'].copy()  # Return a copy to avoid modifying the original

def store_email(email_data):
    """Store email in database"""
    try:
        email = email_data['Email']
        text_body = email['TextBody'][0] if isinstance(email['TextBody'], list) else email['TextBody']
        
        # Get database connection
        conn = get_db_connection()
        
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM emails").fetchone()[0]
        
        # Only store essential information
        conn.execute('''
            INSERT INTO emails (
                id, email_subject, email_text_body, is_analyzed
            ) VALUES (?, ?, ?, FALSE)
        ''', (
            next_id,
            email['Subject'],
            text_body
        ))
        
        logger.info(f"Stored email {next_id}: {email['Subject'][:50]}...")
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

def process_batch(batch, batch_num, total_batches):
    """Process a single batch of emails - Uses DeepSeek API directly"""
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
        summaries = summarize_emails_bulk(batch)
        
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

        # Use large model for complex analysis
        model_config = select_model('complex_analysis')
        # Reduce max tokens for analysis
        model_config['max_tokens'] = min(model_config['max_tokens'], 1000)

        # Use DeepSeek API directly for analysis
        thread_logger.info("Using DeepSeek API for batch analysis")
        response = get_deepseek_response(
            messages=[
                {"role": "system", "content": "You are an expert at analyzing maintenance and service-related emails. You MUST follow the exact format specified in the prompt, including exactly 1 bullet point per section. Keep each bullet point to ONE sentence and focus on the most critical issue."},
                {"role": "user", "content": prompt}
            ],
            model_config=model_config
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

        # Store analysis results for each email in the batch
        for id, _, _ in batch:
            # Map old section names to new ones for database storage
            pd = section_map.get("PROCEDURAL DEVIATIONS", "") or ""
            ri = section_map.get("RECURRENCE INDICATORS", "") or ""
            st = section_map.get("SYSTEMIC TRENDS", "") or ""
            
            # Use a separate database connection for each thread
            try:
                thread_conn = duckdb.connect(DB_PATH)
                thread_conn.execute('''
                    INSERT INTO analysis_results_new 
                    (id, email_id, procedural_deviations, recurrence_indicators, systemic_trends)
                    VALUES (
                        (SELECT COALESCE(MAX(id), 0) + 1 FROM analysis_results_new),
                        ?, ?, ?, ?
                    )
                ''', (id, pd, ri, st))
                thread_logger.info(f"Stored analysis results for email {id}")
                thread_conn.close()
            except Exception as e:
                thread_logger.error(f"Error storing analysis results for email {id}: {str(e)}")

        return section_map
    except Exception as e:
        thread_logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return None

def get_insights(include_analyzed=False, batch_size=10, use_similarity_batching=False, similarity_threshold=0.7):
    """Get insights for emails using parallel batch processing"""
    logger.info("Starting insights generation")
    
    # Get emails based on include_analyzed flag
    query = '''
        SELECT id, email_text_body, email_subject
        FROM emails 
        WHERE is_analyzed = FALSE OR ? = TRUE
        ORDER BY id DESC
    '''
    all_emails = conn.execute(query, [include_analyzed]).fetchall()
    
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
        # Create a partial function with the total number of batches
        process_batch_partial = partial(process_batch, total_batches=len(batches))
        
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
                    
                    # Mark unanalyzed emails in this batch as processed using a separate connection
                    try:
                        thread_conn = duckdb.connect(DB_PATH)
                        unanalyzed_ids = [id for id, _, _ in batch if not include_analyzed]
                        if unanalyzed_ids:
                            thread_conn.execute('''
                                UPDATE emails 
                                SET is_analyzed = TRUE, analyzed_at = CURRENT_TIMESTAMP 
                                WHERE id IN ({})
                            '''.format(','.join('?' * len(unanalyzed_ids))), unanalyzed_ids)
                        thread_conn.close()
                    except Exception as e:
                        logger.error(f"Error updating email status: {str(e)}")
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
    
    analysis_stats = {
        'total_emails': len(all_emails),
        'analyzed_emails': len([id for id, _, _ in all_emails if not include_analyzed]),
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

def clear_emails_table():
    """Clear the emails table and all dependent tables"""
    try:
        # Close current connection
        if 'conn' in st.session_state and st.session_state.conn is not None:
            st.session_state.conn.close()
            st.session_state.conn = None
        
        # Delete the database file to fully clear data and reduce file size
        if os.path.exists(DB_PATH):  # Use global DB_PATH
            os.remove(DB_PATH)
            logger.info(f"Deleted database file {DB_PATH}")
        
        # Reconnect and reinitialize database
        st.session_state.conn = duckdb.connect(DB_PATH)  # Use global DB_PATH
        if not init_database():
            logger.error("Failed to reinitialize database after clearing")
            return False
        
        logger.info("Database cleared by deleting file and reinitializing")
        return True
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return False

def get_total_analyzed_emails():
    """Get the total number of analyzed emails"""
    try:
        conn = get_db_connection()
        result = conn.execute("""
            SELECT COUNT(*) 
            FROM emails 
            WHERE is_analyzed = TRUE
        """).fetchone()[0]
        return result
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

def get_email_context(limit=10, query_text=None, days_back=30, min_analysis_quality=0.5):
    """Get context from analyzed emails for RAG using their analysis summaries"""
    try:
        # Get database connection
        conn = get_db_connection()
        
        # Log the incoming parameters
        logger.info(f"Getting email context - Query: '{query_text}', Limit: {limit}, Days back: {days_back}")
        
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days_back)
        
        # Determine which categories are relevant to the query
        relevant_categories = determine_relevant_categories(query_text) if query_text else [
            'procedural_deviations', 'recurrence_indicators', 'systemic_trends'
        ]
        
        # Build dynamic SELECT clause based on relevant categories
        select_clause = ', '.join([
            f"COALESCE({category}, '') as {category}"
            for category in relevant_categories
        ])
        
        # Get analyzed emails with their analysis results within date range
        query = f'''
            WITH ranked_emails AS (
                SELECT 
                    ar.email_id,
                    ar.analysis_date,
                    {select_clause},
                    -- Calculate analysis quality based on content length and category coverage
                    (
                        LENGTH(COALESCE(procedural_deviations, '')) +
                        LENGTH(COALESCE(recurrence_indicators, '')) +
                        LENGTH(COALESCE(systemic_trends, ''))
                    ) / 3.0 as analysis_quality
                FROM analysis_results_new ar
                WHERE ar.analysis_date >= ?
                ORDER BY ar.analysis_date DESC
            )
            SELECT * FROM ranked_emails
            WHERE analysis_quality >= ?
            LIMIT ?
        '''
        results = conn.execute(query, [date_threshold, min_analysis_quality, MAX_EMAILS_PER_CONTEXT]).fetchall()
        
        if not results:
            return f"No analyzed emails found in the last {days_back} days with sufficient analysis quality.", 0
        
        # Get email subjects in a separate query
        email_ids = [r[0] for r in results]
        subjects = {id: subject for id, subject in conn.execute(
            'SELECT id, email_subject FROM emails WHERE id IN ({})'.format(
                ','.join('?' * len(email_ids))
            ), email_ids).fetchall()}
        
        # Process results with increased context size
        total_chars = 0
        used_emails = set()
        context_parts = []
        
        for email_id, analysis_date, *analysis_values in results:
            if email_id in used_emails or total_chars >= MAX_CONTEXT_SIZE:
                break
                
            subject = subjects.get(email_id, "Subject not found")
            analysis_summary = []
            
            for category, value in zip(relevant_categories, analysis_values):
                if value:
                    display_name = ' '.join(word.capitalize() for word in category.split('_'))
                    analysis_summary.append(f"{display_name}: {value}")
            
            if analysis_summary:
                email_content = f"Email {email_id} (Analyzed: {analysis_date}):\nSubject: {subject}\nAnalysis: {' | '.join(analysis_summary)}"
                email_size = len(email_content)
                
                if total_chars + email_size <= MAX_CONTEXT_SIZE:
                    used_emails.add(email_id)
                    context_parts.append(email_content)
                    total_chars += email_size
        
        # Build final context
        categories_str = ', '.join(category.replace('_', ' ').title() for category in relevant_categories)
        context = f"Analyzed email summaries (showing {len(used_emails)} emails, {total_chars} chars)\n"
        context += f"Categories included: {categories_str}\n"
        context += f"Analysis period: Last {days_back} days\n"
        context += "\n".join(context_parts)
        
        logger.info(f"Context summary: {len(used_emails)} emails, {total_chars} chars")
        return context, len(used_emails)
        
    except Exception as e:
        logger.error(f"Error getting email context: {str(e)}")
        return "Error retrieving email context. Please check the logs for details.", 0

def get_cached_query(query_text, context_size, model_name):
    """Get cached query result if available and not expired"""
    try:
        conn = get_db_connection()
        # Add cache expiration (24 hours)
        cache_expiry = datetime.now() - timedelta(hours=24)
        
        result = conn.execute('''
            SELECT response_text, access_count, created_at
            FROM query_cache
            WHERE query_text = ? 
            AND context_size = ?
            AND model_used = ?
            AND created_at > ?
            ORDER BY last_accessed DESC
            LIMIT 1
        ''', (query_text, context_size, model_name, cache_expiry)).fetchone()
        
        if result:
            response_text, access_count, created_at = result
            # Update access count and last accessed time
            conn.execute('''
                UPDATE query_cache
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE query_text = ? AND context_size = ? AND model_used = ?
            ''', (query_text, context_size, model_name))
            return response_text
    except Exception as e:
        logger.error(f"Error retrieving cached query: {str(e)}")
    return None

def cache_query_result(query_text, response_text, context_size, model_name):
    """Cache query result with model information"""
    try:
        conn = get_db_connection()
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM query_cache").fetchone()[0]
        
        conn.execute('''
            INSERT INTO query_cache (id, query_text, response_text, context_size, model_used)
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
    """Get similar emails using TF-IDF and cosine similarity"""
    try:
        conn = get_db_connection()
        # Get all emails
        emails = conn.execute('''
            SELECT id, email_subject, email_text_body
            FROM emails
        ''').fetchall()
        
        if not emails:
            return []
        
        # Prepare texts for TF-IDF
        texts = [f"Subject: {subject}\nBody: {body}" for _, subject, body in emails]
        texts.insert(0, query_text)  # Add query text at the beginning
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarities with query
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Create results list
        results = []
        for i, (id, subject, text) in enumerate(emails):
            similarity = similarities[0][i]
            results.append((id, subject, text, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:limit]
    except Exception as e:
        logger.error(f"Error finding similar emails: {str(e)}")
        return []

def query_llm_with_context(query_text, context, model_name='groq'):
    """
    Query the LLM with email context using the specified model.
    
    Args:
        query_text (str): The user's query
        context (str): The email context to use for answering
        model_name (str): The model to use ('groq' or 'deepseek')
        
    Returns:
        str: The LLM's response
    """
    try:
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
            7. Format your response in clear, readable sections if appropriate"""},
            {"role": "user", "content": f"""Context from analyzed emails:
            {context}
            
            Question: {query_text}
            
            Please provide a clear, concise answer based on the email context above."""}
        ]

        # Get response from selected model
        logger.info(f"Querying {model_name} with context")
        if model_name == 'groq':
            response = get_groq_response(messages, model_config)
        else:  # deepseek
            response = get_deepseek_response(messages, model_config)
        
        # Cache the response
        cache_query_result(query_text, response, len(context.split('\n')), model_name)
        
        return response
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

# Streamlit UI
st.set_page_config(layout="wide", page_title="Email Analysis Dashboard")
st.title('📧 Email Analysis Dashboard')

# Create three main columns
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
    if st.button('🗑️ Clear All'):
        if clear_emails_table():
            st.success('✅ All emails cleared!')
            st.rerun()
        else:
            st.error('❌ Failed to clear emails')

  
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
                    ar.analysis_date
                FROM analysis_results_new ar
                JOIN emails e ON ar.email_id = e.id
                ORDER BY ar.analysis_date DESC
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
                    # Get context from analyzed emails
                    context, num_emails = get_email_context(context_size, query, days_back=days_back)
                    
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
            st.markdown(st.session_state['last_query']['response'])

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
                    e.email_text_body,
                    e.is_analyzed,
                    e.analyzed_at
                FROM emails e
                ORDER BY e.id DESC
            '''
            emails = conn.execute(query).fetchdf()
            
            if not emails.empty:
                # Format the dataframe for display
                emails['analyzed_at'] = emails['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                emails['is_analyzed'] = emails['is_analyzed'].map({True: '✅', False: '❌'})
                emails['email_text_body'] = emails['email_text_body'].apply(
                    lambda x: x[:100] + '...' if len(str(x)) > 100 else x
                )
                
                # Rename and reorder columns
                emails = emails.rename(columns={
                    'id': 'ID',
                    'email_subject': 'Subject',
                    'email_text_body': 'Body Preview',
                    'is_analyzed': 'Analyzed',
                    'analyzed_at': 'Analyzed At'
                })
                
                # Display with better formatting
                st.dataframe(
                    emails[['ID', 'Subject', 'Body Preview', 'Analyzed', 'Analyzed At']],
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

