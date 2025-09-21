# Vernacular AI Chatbot

## Pain Point
A language barrier hinders support for sellers in Tier 2/3 cities. Many sellers are more comfortable communicating in their local languages (Hindi, Tamil, Bengali, etc.) but support systems primarily use English, creating accessibility issues and reducing support effectiveness.

## Proposed Solution
A multilingual chatbot using Retrieval-Augmented Generation (RAG) to provide local language support. The system detects the user's language, retrieves relevant information from a knowledge base, and generates responses in the seller's preferred language.

## Architecture
- **Language Detection**: Automatically identifies user's preferred language
- **RAG System**: Retrieves relevant documents from knowledge base
- **Multilingual LLM**: Generates responses in local languages
- **Knowledge Base**: Vector database containing seller support information
- **Translation Layer**: Handles language conversion when needed

## Files
- `src/chatbot.py`: Core chatbot implementation with RAG
- `requirements.txt`: Python dependencies

## Usage
Sellers can interact with the chatbot in their preferred language. The system automatically detects the language, retrieves relevant support information, and provides helpful responses in the seller's local language, improving accessibility and support quality.
