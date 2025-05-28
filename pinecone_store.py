import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pinecone
from sentence_transformers import SentenceTransformer
import torch
import warnings
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Force CPU mode to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_model():
    """Get or initialize the sentence transformer model"""
    if 'sentence_transformer_model' not in st.session_state:
        try:
            device = 'cpu'
            logger.info(f"Using device: {device}")
            
            # Initialize model with specific settings
            st.session_state.sentence_transformer_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=device,
                cache_folder='./model_cache'
            )
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            st.session_state.sentence_transformer_model = None
    
    if st.session_state.sentence_transformer_model is None:
        raise RuntimeError("Sentence transformer model failed to initialize")
    return st.session_state.sentence_transformer_model

def init_pinecone():
    """Initialize Pinecone client and index"""
    try:
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_env = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')

        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return None

        # Initialize Pinecone client with new API
        pc = pinecone.Pinecone(
            api_key=pinecone_api_key,
            environment=pinecone_env
        )
        
        index_name = "email-analysis"
        dimension = 384  # Must match embedding model output

        # Check if index exists and create if it doesn't
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logger.info(f"Created new Pinecone index: {index_name}")

        # Get index
        index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return None

def get_embedding(text: str) -> List[float]:
    """Generate embedding using sentence-transformers"""
    try:
        if not text.strip():
            return None
            
        model = get_model()
        # Ensure text is properly encoded
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        embedding = model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        return None

def store_email(email_data: Dict[str, Any]) -> bool:
    try:
        email = email_data['Email']
        text_body = email.get('TextBody', '')
        if isinstance(text_body, list):
            text_body = text_body[0]

        # Generate a unique ID using timestamp and tenant ID
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        tenant_id = email_data.get('TenantID', '')
        doc_id = f"email_{tenant_id}_{timestamp}"  # Make ID unique with timestamp

        text_for_embedding = f"Subject: {email['Subject']}\nBody: {text_body}"
        embedding = get_embedding(text_for_embedding)

        if not embedding or all(v == 0 for v in embedding):
            logger.error(f"Invalid or empty embedding for email {doc_id}")
            return False

        metadata = {
            'email_id': str(email_data.get('TenantID', '')),
            'subject': email['Subject'],
            'from': email['From'],
            'to': email['To'],
            'created_at': datetime.now().isoformat(),
            'created_at_timestamp': int(datetime.now().timestamp()),
            'is_analyzed': False,
            'text_body': text_body[:1000],
            'html_body': email.get('HtmlBody', '')[:1000],
            'incident_type': email_data.get('incident_type', 'Others'),
            'severity': email_data.get('severity', 'Low'),
            'reasoning': email_data.get('reasoning', ''),
            'unique_id': doc_id  # Store the unique ID in metadata
        }

        index = init_pinecone()
        if not index:
            return False

        # Format vector data according to new API
        vector_data = {
            'id': doc_id,
            'values': embedding,
            'metadata': metadata
        }

        # Upsert vector using new API format
        index.upsert(vectors=[vector_data])
        logger.info(f"Successfully stored email {doc_id} in Pinecone")
        return True
    except Exception as e:
        logger.error(f"Error storing email in Pinecone: {str(e)}")
        return False

def query_similar_emails(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    try:
        index = init_pinecone()
        if not index:
            return []

        embedding = get_embedding(query_text)
        if not embedding:
            logger.error("Failed to get embedding for query text")
            return []

        # Query using new API format
        results = index.query(
            vector=embedding,
            top_k=n_results,
            include_metadata=True
        )

        # Format results according to new API response
        return [{
            'id': match.id,
            'metadata': match.metadata,
            'score': match.score
        } for match in results.matches]
    except Exception as e:
        logger.error(f"Error querying similar emails: {str(e)}")
        return []

def get_email_context(limit: int = 10, query_text: str = None, days_back: int = 30) -> str:
    try:
        index = init_pinecone()
        if not index:
            return "Failed to connect to Pinecone.", 0

        if query_text:
            results = query_similar_emails(query_text, limit * 2)
        else:
            results = index.query(vector=[0] * 384, top_k=limit, include_metadata=True).matches

        if not results:
            return "No emails found.", 0

        context_parts = []
        used_emails = set()
        total_chars = 0

        for result in results:
            if hasattr(result, 'id'):
                email_id = result.id
                metadata = result.metadata
                score = getattr(result, 'score', None)
            else:
                email_id = result['id']
                metadata = result['metadata']
                score = result.get('score', None)

            if email_id in used_emails or total_chars >= 30000:
                continue

            email_content = f"""Email {email_id} (Similarity: {score:.2f} if score else 'N/A'):
Subject: {metadata['subject']}
From: {metadata['from']}
To: {metadata['to']}
Incident Type: {metadata['incident_type']}
Severity: {metadata['severity']}
Body Preview: {metadata['text_body'][:200]}..."""

            if total_chars + len(email_content) <= 30000:
                context_parts.append(email_content)
                used_emails.add(email_id)
                total_chars += len(email_content)

        context = "\n".join(context_parts)
        return context, len(used_emails)
    except Exception as e:
        logger.error(f"Error getting email context: {str(e)}")
        return "Error retrieving email context.", 0

def clear_pinecone_index():
    try:
        index = init_pinecone()
        if not index:
            return False
        index.delete(delete_all=True)
        logger.info("Successfully cleared Pinecone index")
        return True
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {str(e)}")
        return False

def get_pinecone_stats():
    try:
        index = init_pinecone()
        if not index:
            return None
        stats = index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_name': 'email-analysis',
            'metric': stats.metric
        }
    except Exception as e:
        logger.error(f"Error getting Pinecone stats: {str(e)}")
        return None

def fetch_all_pinecone_emails(limit=1000):
    index = init_pinecone()
    if not index:
        return []
    # Pinecone doesn't have a "fetch all", so we use a zero vector and high top_k
    results = index.query(vector=[0]*384, top_k=limit, include_metadata=True)
    emails = []
    for match in results.matches:
        meta = match.metadata
        emails.append({
            'id': match.id,
            'subject': meta.get('subject', ''),
            'body': meta.get('text_body', ''),
            'incident_type': meta.get('incident_type', 'Others'),
            'severity': meta.get('severity', 'Low'),
            'created_at': meta.get('created_at', ''),
            'reasoning': meta.get('reasoning', ''),
        })
    return emails
