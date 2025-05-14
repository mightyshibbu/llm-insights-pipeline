import streamlit as st
import duckdb
import json
from datetime import datetime
import os
from openai import OpenAI
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

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1"
)
logger.info("DeepSeek client initialized")

# Initialize DuckDB connection
db_path = 'emails.db'

def get_db_connection():
    """Get a database connection, creating a new one if needed"""
    try:
        # Check if connection exists and is valid
        if 'conn' in st.session_state and st.session_state.conn is not None:
            try:
                # Test the connection
                st.session_state.conn.execute("SELECT 1").fetchone()
                return st.session_state.conn
            except Exception:
                # If test fails, close the connection
                try:
                    st.session_state.conn.close()
                except:
                    pass
                st.session_state.conn = None
        
        # Create new connection
        conn = duckdb.connect(db_path)
        st.session_state.conn = conn
        logger.info(f"New DuckDB connection established to {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return None

# Initialize connection
conn = get_db_connection()
if not conn:
    st.error("Failed to connect to database. Please check the logs for details.")
    st.stop()

logger.info(f"DuckDB connection established to {db_path}")

# Create tables if they don't exist
def init_database():
    """Initialize database tables"""
    try:
        # Create emails table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY,
                tenant_id INTEGER,
                email_to VARCHAR,
                email_from VARCHAR,
                email_subject VARCHAR,
                email_text_body TEXT,
                email_html_body TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analyzed_at TIMESTAMP DEFAULT NULL,
                is_analyzed BOOLEAN DEFAULT FALSE
            )
        ''')

        # Create analysis results table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY,
                email_id INTEGER,
                process_mistakes TEXT,
                failure_patterns TEXT,
                common_themes TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email_id) REFERENCES emails(id)
            )
        ''')

        # Create query cache table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY,
                query_text TEXT,
                response_text TEXT,
                context_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        logger.info("Database schema initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing database schema: {str(e)}")
        return False

# Initialize database
if not init_database():
    st.error("Failed to initialize database. Please check the logs for details.")
    st.stop()

def store_email(email_data):
    """Store email in database"""
    try:
        email = email_data['Email']
        text_body = email['TextBody'][0] if isinstance(email['TextBody'], list) else email['TextBody']
        
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM emails").fetchone()[0]
        
        conn.execute('''
            INSERT INTO emails (
                id, tenant_id, email_to, email_from, email_subject,
                email_text_body, email_html_body, is_analyzed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)
        ''', (
            next_id,
            email_data['TenantID'],
            email['To'],
            email['From'],
            email['Subject'],
            text_body,
            email['HtmlBody']
        ))
        
        logger.info(f"Stored email {next_id} from {email['From']} to {email['To']}")
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
    for email in emails:
        id, text_body, subject, created_at = email
        full_text = f"Subject: {subject}\nBody: {text_body}"
        email_texts[id] = {
            'text': full_text,
            'email': email
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

def summarize_emails_bulk(batch):
    """Summarize multiple emails in a single LLM call"""
    try:
        # Prepare context with all emails in the batch
        context = "Summarize these maintenance/service emails in 2-3 sentences each, focusing on key issues and actions needed:\n\n"
        for id, text_body, subject, _ in batch:
            context += f"Email ID {id}:\nSubject: {subject}\nBody: {text_body}\n\n"
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at summarizing maintenance and service-related emails. For each email, provide a concise 2-3 sentence summary focusing on key issues and actions needed. Format your response with 'Email ID X:' before each summary."},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse the response to extract summaries
        summaries = {}
        response_text = response.choices[0].message.content.strip()
        
        # Split response into individual email summaries
        email_blocks = response_text.split('Email ID')
        for block in email_blocks[1:]:  # Skip the first empty block
            try:
                id_str, summary = block.split(':', 1)
                email_id = int(id_str.strip())
                summaries[email_id] = summary.strip()
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse summary for block: {block}")
                continue
        
        # Fill in any missing summaries with truncated text
        for id, text_body, subject, _ in batch:
            if id not in summaries:
                logger.warning(f"Using fallback summary for email {id}")
                summaries[id] = f"Subject: {subject}\nBody: {text_body[:200]}..."
        
        return summaries
    except Exception as e:
        logger.error(f"Error in bulk summarization: {str(e)}")
        # Return fallback summaries for all emails
        return {
            id: f"Subject: {subject}\nBody: {text_body[:200]}..."
            for id, text_body, subject, _ in batch
        }

def process_batch(batch, batch_num, total_batches):
    """Process a single batch of emails"""
    try:
        logger.info(f"Processing batch {batch_num} of {total_batches}")
        
        # Get bulk summaries for all emails in the batch
        summaries = summarize_emails_bulk(batch)
        
        # Prepare context from summarized emails
        context = "Maintenance and service emails:\n"
        for id, text_body, subject, _ in batch:
            summary = summaries.get(id, f"Subject: {subject}\nBody: {text_body[:200]}...")
            context += f"\nEmail {id}:\n{summary}\n"
        
        prompt = f"""You are an expert at analyzing maintenance and service-related emails. Analyze these summarized emails and provide specific insights.

        {context}

        You MUST provide analysis in EXACTLY these three sections, with these EXACT headers:

        PROCESS MISTAKES:
        [Your analysis here]

        FAILURE PATTERNS:
        [Your analysis here]

        COMMON THEMES:
        [Your analysis here]

        For each section:
        1. Start with the EXACT header shown above
        2. Provide EXACTLY 2 bullet points (no more, no less)
        3. Each bullet point MUST start with "• Email ID X:" where X is the actual email ID
        4. Be specific and detailed
        5. Include examples from the emails"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing maintenance and service-related emails. You MUST follow the exact format specified in the prompt, including exactly 2 bullet points per section."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the response
        parts = re.split(r'(PROCESS MISTAKES:|FAILURE PATTERNS:|COMMON THEMES:)', response_text)
        section_map = {}
        for idx in range(1, len(parts), 2):
            header = parts[idx].rstrip(':')
            content = parts[idx+1].strip()
            section_map[header] = content

        # Store analysis results for each email in the batch
        for id, _, _, _ in batch:
            store_analysis_results(
                id,
                section_map.get("PROCESS MISTAKES", ""),
                section_map.get("FAILURE PATTERNS", ""),
                section_map.get("COMMON THEMES", "")
            )

        return section_map
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return None

def get_insights(include_analyzed=False, batch_size=10, use_similarity_batching=False, similarity_threshold=0.7):
    """Get insights for emails using parallel batch processing"""
    logger.info("Starting insights generation")
    
    # Get emails based on include_analyzed flag
    query = '''
        SELECT id, email_text_body, email_subject, created_at
        FROM emails 
        WHERE is_analyzed = FALSE OR ? = TRUE
        ORDER BY created_at DESC
    '''
    all_emails = conn.execute(query, [include_analyzed]).fetchall()
    
    if not all_emails:
        logger.info("No emails found in database")
        return {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'process_mistakes': 'No emails found in database',
            'failure_patterns': 'No emails found in database',
            'common_themes': 'No emails found in database',
            'email_count': 0,
            'analysis_stats': {
                'total_emails': 0,
                'analyzed_emails': 0,
                'referenced_emails': 0,
                'date_range': 'N/A',
                'batches_processed': 0
            }
        }
    
    # Calculate date range
    dates = [created_at for _, _, _, created_at in all_emails]
    date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
    
    logger.info(f"Found {len(all_emails)} emails to analyze")
    logger.info(f"Date range of emails: {date_range}")
    
    # Create batches based on similarity or chronological order
    if use_similarity_batching:
        batches = create_similarity_batches(all_emails, batch_size, similarity_threshold)
    else:
        batches = [all_emails[i:i + batch_size] for i in range(0, len(all_emails), batch_size)]
    
    # Process batches in parallel
    all_insights = {
        'PROCESS MISTAKES': [],
        'FAILURE PATTERNS': [],
        'COMMON THEMES': []
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
                    all_insights["PROCESS MISTAKES"].extend(section_map["PROCESS MISTAKES"].splitlines())
                    all_insights["FAILURE PATTERNS"].extend(section_map["FAILURE PATTERNS"].splitlines())
                    all_insights["COMMON THEMES"].extend(section_map["COMMON THEMES"].splitlines())
                    
                    # Mark unanalyzed emails in this batch as processed
                    unanalyzed_ids = [id for id, _, _, _ in batch if not include_analyzed]
                    if unanalyzed_ids:
                        conn.execute('''
                            UPDATE emails 
                            SET is_analyzed = TRUE, analyzed_at = CURRENT_TIMESTAMP 
                            WHERE id IN ({})
                        '''.format(','.join('?' * len(unanalyzed_ids))), unanalyzed_ids)
            except Exception as e:
                logger.error(f"Error processing batch result: {str(e)}")
    
    # Combine insights from all batches
    combined_insights = {
        'PROCESS MISTAKES': '\n'.join(all_insights['PROCESS MISTAKES']),
        'FAILURE PATTERNS': '\n'.join(all_insights['FAILURE PATTERNS']),
        'COMMON THEMES': '\n'.join(all_insights['COMMON THEMES'])
    }
    
    # Count references to specific emails in the analysis
    email_references = {}
    for section in combined_insights.values():
        for id in [id for id, _, _, _ in all_emails]:
            if str(id) in section:
                email_references[id] = email_references.get(id, 0) + 1
    
    analysis_stats = {
        'total_emails': len(all_emails),
        'analyzed_emails': len([id for id, _, _, _ in all_emails if not include_analyzed]),
        'referenced_emails': len(email_references),
        'date_range': date_range,
        'batches_processed': len(batches),
        'batching_method': 'similarity' if use_similarity_batching else 'chronological'
    }
    
    logger.info(f"Analysis complete. Stats: {analysis_stats}")
    
    return {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'process_mistakes': combined_insights['PROCESS MISTAKES'] or 'No process mistakes identified',
        'failure_patterns': combined_insights['FAILURE PATTERNS'] or 'No failure patterns identified',
        'common_themes': combined_insights['COMMON THEMES'] or 'No common themes identified',
        'email_count': len(all_emails),
        'analysis_stats': analysis_stats
    }

def clear_emails_table():
    """Clear the emails table and all dependent tables"""
    global conn
    try:
        # Close current connection
        conn.close()
        # Delete the database file to fully clear data and reduce file size
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Deleted database file {db_path}")
        # Reconnect and reinitialize database
        conn = duckdb.connect(db_path)
        if not init_database():
            logger.error("Failed to reinitialize database after clearing")
            return False
        logger.info("Database cleared by deleting file and reinitializing")
        return True
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return False

def get_email_context(limit=50):
    """Get context from recent emails for RAG"""
    query = '''
        SELECT id, email_text_body, email_subject, created_at
        FROM emails
        ORDER BY created_at DESC
        LIMIT ?
    '''
    emails = conn.execute(query, [limit]).fetchall()
    
    context = "Recent maintenance and service emails:\n"
    for id, text_body, subject, created_at in emails:
        context += f"\nEmail {id} ({created_at.strftime('%Y-%m-%d')}):\nSubject: {subject}\nBody: {text_body}\n"
    
    return context

def query_llm_with_context(query_text, context):
    """Query LLM with email context using RAG approach"""
    # Check cache first
    cached_response = get_cached_query(query_text, len(context.split('\n')))
    if cached_response:
        logger.info("Using cached query result")
        return cached_response

    # Get similar emails using embeddings
    similar_emails = get_similar_emails(query_text)
    if similar_emails:
        context += "\n\nMost relevant emails:\n"
        for id, subject, text, similarity in similar_emails:
            context += f"\nEmail {id} (Relevance: {similarity:.2f}):\nSubject: {subject}\nBody: {text}\n"

    prompt = f"""You are an expert at analyzing maintenance and service-related emails. Use the following context to answer the user's question.

    Context:
    {context}

    User Question: {query_text}

    Instructions:
    1. Use ONLY the information from the provided context
    2. If the context doesn't contain relevant information, say so
    3. Reference specific email IDs when making points
    4. Be specific and detailed in your response
    5. Format your response in clear paragraphs with bullet points where appropriate

    Response:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing maintenance and service-related emails. Use the provided context to answer questions accurately and specifically."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        response_text = response.choices[0].message.content
        
        # Cache the result
        cache_query_result(query_text, response_text, len(context.split('\n')))
        
        return response_text
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return f"Error generating response: {str(e)}"

def get_cached_query(query_text, context_size):
    """Get cached query result if available"""
    try:
        result = conn.execute('''
            SELECT response_text, access_count
            FROM query_cache
            WHERE query_text = ? AND context_size = ?
            ORDER BY last_accessed DESC
            LIMIT 1
        ''', (query_text, context_size)).fetchone()
        
        if result:
            # Update access count and last accessed time
            conn.execute('''
                UPDATE query_cache
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE query_text = ? AND context_size = ?
            ''', (query_text, context_size))
            return result[0]
    except Exception as e:
        logger.error(f"Error retrieving cached query: {str(e)}")
    return None

def cache_query_result(query_text, response_text, context_size):
    """Cache query result"""
    try:
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM query_cache").fetchone()[0]
        
        conn.execute('''
            INSERT INTO query_cache (id, query_text, response_text, context_size)
            VALUES (?, ?, ?, ?)
        ''', (next_id, query_text, response_text, context_size))
        logger.info(f"Cached query result for: {query_text[:50]}...")
    except Exception as e:
        logger.error(f"Error caching query result: {str(e)}")

def store_analysis_results(email_id, process_mistakes, failure_patterns, common_themes):
    """Store LLM analysis results in database"""
    try:
        # Get the next available ID
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM analysis_results").fetchone()[0]
        
        conn.execute('''
            INSERT INTO analysis_results 
            (id, email_id, process_mistakes, failure_patterns, common_themes)
            VALUES (?, ?, ?, ?, ?)
        ''', (next_id, email_id, process_mistakes, failure_patterns, common_themes))
        logger.info(f"Stored analysis results for email {email_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing analysis results: {str(e)}")
        return False

def get_similar_emails(query_text, limit=5):
    """Get similar emails using TF-IDF and cosine similarity"""
    try:
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

def get_db_size():
    """Get the current size of the database file"""
    try:
        if os.path.exists(db_path):
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
            size_bytes = os.path.getsize(db_path)
            
            # Reconnect after getting size
            conn = duckdb.connect(db_path)
            st.session_state.conn = conn
            
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
        - Date Range: {stats.get('date_range', 'N/A')}
        - Batches Processed: {stats.get('batches_processed', 0)}
        - Batching Method: {stats.get('batching_method', 'N/A')}
        """)
        
        with st.expander("Process Mistakes", expanded=True):
            st.write(st.session_state['insights']['process_mistakes'])
        
        with st.expander("Failure Patterns", expanded=True):
            st.write(st.session_state['insights']['failure_patterns'])
        
        with st.expander("Common Themes", expanded=True):
            st.write(st.session_state['insights']['common_themes'])

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

# Right Column - RAG Query Interface
with col3:
    st.subheader('🤖 AI Query Interface')
    
    # Query input
    query = st.text_area(
        label="Ask a question about the emails",
        height=100,
        placeholder="Example: What are the most common maintenance issues reported?",
        key="rag_query"
    )
    
    # Context size control
    context_size = st.slider(
        "Number of recent emails to consider",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Adjust how many recent emails to include in the context"
    )
    
    if st.button('🔍 Query Emails'):
        if query:
            with st.spinner('Analyzing emails...'):
                # Get context from recent emails
                context = get_email_context(context_size)
                
                # Query LLM with context
                response = query_llm_with_context(query, context)
                
                # Store in session state
                st.session_state['last_query'] = {
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        else:
            st.warning('Please enter a question to analyze.')
    
    # Display last query result
    if 'last_query' in st.session_state:
        st.markdown(f"**Last Query:** {st.session_state['last_query']['timestamp']}")
        st.markdown("### Question")
        st.write(st.session_state['last_query']['query'])
        st.markdown("### Response")
        st.write(st.session_state['last_query']['response'])

# Footer with metrics
st.markdown("---")
st.markdown("### 📈 Dashboard Status")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    try:
        # Ensure we have a valid connection
        conn = get_db_connection()
        if conn:
            try:
                count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
                st.metric("Total Emails", count)
            except Exception as e:
                if "Table with name emails does not exist" in str(e):
                    st.metric("Total Emails", "0")
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
    if st.button('📋 Show Table'):
        try:
            # Get all emails with their analysis status
            query = '''
                SELECT 
                    e.id,
                    e.email_subject,
                    e.email_from,
                    e.email_to,
                    e.created_at,
                    e.is_analyzed,
                    e.analyzed_at
                FROM emails e
                ORDER BY e.created_at DESC
            '''
            emails = conn.execute(query).fetchdf()
            
            if not emails.empty:
                # Format the dataframe for display
                emails['created_at'] = emails['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                emails['analyzed_at'] = emails['analyzed_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
                emails['is_analyzed'] = emails['is_analyzed'].map({True: '✅', False: '❌'})
                
                # Rename columns for better display
                emails = emails.rename(columns={
                    'id': 'ID',
                    'email_subject': 'Subject',
                    'email_from': 'From',
                    'email_to': 'To',
                    'created_at': 'Created',
                    'is_analyzed': 'Analyzed',
                    'analyzed_at': 'Analyzed At'
                })
                
                # Display the table
                st.dataframe(emails, use_container_width=True)
            else:
                st.info("No emails found in the database.")
        except Exception as e:
            logger.error(f"Error displaying table: {str(e)}")
            st.error("Error displaying table. Please check the logs for details.")

