# Email Analysis Dashboard

This Streamlit application provides an interface to store and analyze email data using DuckDB and DeepSeek LLM. It helps track maintenance information and failure patterns from email content using advanced AI analysis.

## Features

- Store email data in a DuckDB database
- Advanced email content analysis using DeepSeek LLM
- Intelligent insights generation for maintenance and failure patterns
- Real-time insights updates
- Simple and intuitive user interface

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your DeepSeek API credentials:
```
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Usage

1. The application will open in your default web browser
2. To add a new email:
   - Paste the email JSON data in the text area
   - Click "Add Email"
3. To view insights:
   - Click "Update Insights" to refresh the analysis
   - View the AI-generated insights about maintenance and failure patterns

## Data Structure

The application stores email data in a DuckDB table with the following structure:
- Basic email metadata (TenantID, Type, Subtype, etc.)
- Email content (To, From, Subject, Body)
- Timestamps for tracking

## Insights

The application uses DeepSeek LLM to provide two main insights:
1. Process Mistakes: Detailed analysis of maintenance/service history and issues
2. Failure Patterns: AI-powered detection of potential issues or problems in the email content

## Technical Details

- Uses DuckDB for efficient data storage and querying
- Integrates with DeepSeek LLM for advanced text analysis
- Implements RAG (Retrieval Augmented Generation) for better insights
- Provides real-time analysis with loading indicators 