# YouTube Video RAG System

A Retrieval-Augmented Generation (RAG) system that extracts transcripts from YouTube videos and enables intelligent question-answering based on the video content.

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate answers based on the retrieved information

## 🔧 System Architecture

```
YouTube Video → Transcript Extraction → Text Chunking → Vector Embeddings → Vector Store
                                                                                    ↓
User Question → Similarity Search → Retrieved Context → LLM Prompt → Final Answer
```

## 📋 Implementation Overview

This notebook demonstrates a complete RAG pipeline with the following components:

### 1. **YouTube Transcript Extraction**
```python
video_id = "bMgfsdYoEEo"  # Horror movie trailer transcript
transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en-US'])
```
- Extracts transcript from a specific YouTube video
- Handles errors for videos without transcripts
- Preserves timestamp information

### 2. **File Export Capabilities**
- Saves transcript as `.txt` file
- Generates PDF using FPDF library
- Raw transcript data with timestamps preserved

### 3. **Timestamp-Based Search Function**
```python
def search_transcript_with_timestamps(transcript_chunks, query, window=30):
```
- Direct keyword search with precise timestamps
- Returns highlighted matches with context windows
- Example: Searching "They're both hiding" returns match at 0:01

### 4. **Text Chunking**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
```
- Breaks transcript into 2 chunks (for this specific video)
- 1000 character chunks with 200 character overlap

### 5. **Vector Store Creation**
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```
- Creates embeddings using OpenAI's model
- Uses FAISS for similarity search
- Configured to retrieve top 4 similar chunks

### 6. **LLM Integration**
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```
- Uses GPT-4o-mini for answer generation
- Low temperature (0.2) for factual accuracy

### 7. **Custom Prompt Template**
```python
prompt = PromptTemplate(
    template="""
You are a precise and factual assistant. Your job is to answer questions strictly based on the video transcript provided.
- Only use the transcript above to answer.
- If the transcript does not contain the answer, respond exactly with: "I'm sorry, the answer is not available in the video."
- If the transcript only partially answers, say: "The transcript only mentions XYZ but does not provide a full answer."
""")
```

## 📊 Tested Examples

### Example 1: Content Summary
- **Question**: "What is this video about? Can you give me a brief summary of the transcript?"
- **Result**: Successfully identifies the video is about Jack and Janet Smurl's supernatural experiences in Pennsylvania
- **Retrieved Context**: Combines both transcript chunks for comprehensive answer

### Example 2: Unrelated Query
- **Question**: "What is the RAG?"
- **Result**: "I'm sorry, the answer is not available in the video."
- **Behavior**: Correctly identifies that RAG information is not in the horror movie trailer transcript

## 🔍 Sample Transcript Content

The notebook processes a horror movie trailer transcript containing:
- Character dialogue about supernatural events
- References to the Smurl family and their haunted house
- Religious prayers and exorcism attempts
- Sound effects and music cues marked in brackets

## 🏗️ Technical Stack

### Required Libraries
```python
youtube-transcript-api
langchain-community 
langchain-openai
faiss-cpu
tiktoken
python-dotenv
fpdf
```

### Models Used
- **Embeddings**: OpenAI `text-embedding-3-small`
- **LLM**: OpenAI `gpt-4o-mini`
- **Vector Store**: FAISS

## 🔧 Key Functions Implemented

1. **YouTube transcript fetching** with error handling
2. **File export** (TXT and PDF formats)
3. **Timestamp search** with highlighted results
4. **Vector-based semantic search**
5. **Fact-constrained answer generation**

## 📈 System Characteristics

- **Chunks Created**: 2 (for the sample video)
- **Search Results**: Returns relevant passages with timestamps
- **Answer Accuracy**: Strictly limited to transcript content
- **Error Handling**: Graceful responses when information unavailable

This implementation demonstrates a complete RAG pipeline specifically designed for YouTube video content analysis, with robust fact-checking and timestamp integration.