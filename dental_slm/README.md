Setup for SLM:

1. Overview
This is a retrieval augmented small language model (SLM) that generates dental referral notes from malocclusion and gingivitis model outputs from NDCS guidelines.

2. Files


Ndcs_kb.txt: Knowledge base containing:
Malocclusion labels from OMNI Dataset and 10 classes split into Condition, Appliance (like braces) and normal groups
Gingivitis MGI Scoring based on the Mendeley dataset on a 0-4 scale
Mouth region map (Maxilla, mandible)
Referral criteria and form template

ingest.py: Loads ndcs_kb.txt and splits it into chunks, then persists them to a Chroma database at ./chrome_db. Basically processes the ndcs_kb.txt into readable information for the model to use.

Chroma: Open source vector database for storing and searching embeddings rather than rows of data. Chroma finds data which has a similar meaning to another set of data.
Stores each text chunk and its vector from all-MiniLM-L6-v2
Embeds the query to rank stored chunks by its relevance to the query’s vector (basically how related is chunk is to the input)
Returns top k closest matches

test_slm.py: Takes detection outputs from the malocclusion and gingivitis models, applies referral logic from ndcs_kb.txt, builds a pre-filled referral note and retries relevant chunks from Chroma based on the findings. It then prompts a local SLM (phi3) to write a short combined clinical summary. 

Code for .py files with annotations

Content of ndcs_kb.txt

3. Setup
Install dependencies
langchain-community, chromadb, sentence-transformers (for HuggingFace embeddings), ollama Python client.

Install and run ollama locally and pull the phi3 model. (ollama pull phi3)

Place ndcs_kb.txt in a new folder called data/ndcs_kb.txt
Fix ingest.py to persist the Chroma DB (Chroma.from_documents() , then run it once to create ./chrome_db
Run test_slm.py for every screening session and replace the hardcoded detected_labels, mgi_scores, priority, and patient_age_group with real model outputs.

4. Running it in terminal

python ingest.py # one-time / re-run after KB edits
 python test_slm.py # per patient screening

Output: Referral note printed to console with referral field and SLM-generated clinical summary using info from ndcs_kb.txt chunks

Important to note:
 This model follows a RAG pipeline (Retrieval, Augmented, Generation), where:

Test_slm.py pulls 4 most relevant chunks of ndcs_kb.txt from Chroma database
Joined the chunks into context and inserted into the summary prompt sent to phi3
Generates output, calling the phi3 SLM 


