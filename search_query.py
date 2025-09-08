
import base64
import json
import re
import os
import textwrap
from dotenv import load_dotenv
from quart import request, jsonify
import asyncpg

from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from openai import AsyncAzureOpenAI

load_dotenv()
def safe_base64_decode(data):
    if data.startswith("https"):
        return data
    try:
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        data = data.rstrip()
        while data and data[-1] not in valid_chars:
            data = data[:-1]
        while len(data) % 4 == 1:
            data = data[:-1]
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        decoded = base64.b64decode(data).decode("utf-8", errors="ignore")
        decoded = decoded.strip().rstrip("\uFFFD").rstrip("?").strip()
        decoded = re.sub(r'\.(docx|pdf|pptx|xlsx)[0-9]+$', r'.\1', decoded, flags=re.IGNORECASE)
        return decoded
    except Exception as e:
        return f"[Invalid Base64] {data} - {str(e)}"

# -------------------------
# Async search and answer
# -------------------------
async def ask_query(user_query, user_id, conversation_store, clanguage="english"):
    # Async Azure credential
    credential = AsyncDefaultAzureCredential()

    AZURE_SEARCH_SERVICE = "https://aiconciergeserach.search.windows.net"
    deployment_name = "ocm-gpt-4o"

    # Language-based index and prompts
    if clanguage == "french_canadian":
        index_name = "index-evolve-french-sep08"
        answer_prompt_template = """Vous êtes un assistant IA parlant français canadien. Utilisez les extraits de sources les plus pertinents et informatifs ci-dessous pour répondre à la question de l’utilisateur.

Directives :
- Concentrez-vous principalement sur les extraits contenant la réponse la plus directe et complète.
- N’extrayez que les informations factuelles présentes dans les extraits.
- Chaque fait doit être immédiatement suivi de la citation entre crochets, par ex. [3].
- N’ajoutez aucune information qui n’est pas explicitement présente dans les extraits sources.
- Fournissez un résumé suivi de détails de soutien. Utilisez des **mots en gras** pour les titres et les termes importants.

Historique de la conversation :
{conversation_history}

Sources :
{sources}

Question de l’utilisateur : {query}

Répondez avec :
- Une réponse en français canadien, citant les sources entre crochets comme [1], [2], surtout là où la réponse est clairement soutenue."""  # (French prompt here, same as before)
        followup_prompt_template = """En vous basant uniquement sur les extraits suivants, générez 3 questions de suivi que l’utilisateur pourrait poser.
N’utilisez que le contenu des sources. N’inventez pas de nouveaux faits.

Format :
Q1 : <question>
Q2 : <question>
Q3 : <question>

SOURCES :
{citations}

- Toutes les questions doivent être formulées en français canadien.
"""  # (French followup)
    else:
        index_name = "index-evolve-mo"
        answer_prompt_template = """You are an AI assistant. Use the most relevant and informative source chunks below to answer the user's query.

Guidelines:
- Focus your answer primarily on the chunk(s) that contain the most direct and complete answer.
- Extract only factual information present in the chunks.
- Each fact must be followed immediately by the citation in square brackets, e.g., [3].
- Do not add any information not explicitly present in the source chunks.
- Provide a summary followed by supporting details. Use **bold words** for titles and important words.

Conversation History:
{conversation_history}

Sources:
{sources}

User Question: {query}

Respond with:
- An answer citing sources inline like [1], [2], especially where the answer is clearly supported."""  # (English prompt)
        followup_prompt_template = """Based only on the following chunks of source material, generate 3 follow-up questions the user might ask.
Only use the content in the sources. Do not invent new facts.

Format:
Q1: <question>
Q2: <question>
Q3: <question>

SOURCES:
{citations}"""  # (English followup)

    openai_client = AsyncAzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint="https://ai-hubdevaiocm273154123411.cognitiveservices.azure.com/",
    api_key="1inOabIDqV45oV8EyGXA4qGFqN3Ip42pqA5Qd9TAbJFgUdmTBQUPJQQJ99BCACHYHv6XJ3w3AAAAACOGuszT"  # <-- Hardcoded key
)


    search_client = AsyncSearchClient(
        endpoint=AZURE_SEARCH_SERVICE,
        index_name=index_name,
        credential=credential
    )

    # Conversation tracking
    if user_id not in conversation_store:
        conversation_store[user_id] = {"history": [], "chat": ""}

    conversation_store[user_id]["history"].append(user_query)
    if len(conversation_store[user_id]["history"]) > 3:
        conversation_store[user_id]["history"] = conversation_store[user_id]["history"][-3:]

    history_queries = " ".join(conversation_store[user_id]["history"])
    conversation_history = conversation_store[user_id]["chat"]

    # -------------------------
    # Async fetch chunks
    # -------------------------
    async def fetch_chunks(query_text, k_value, start_index):
        vector_query = VectorizableTextQuery(text=query_text, k_nearest_neighbors=5, fields="text_vector")
        results = await search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            select=["title", "chunk", "parent_id"],
            top=k_value,
            semantic_configuration_name=f"{index_name}-semantic-configuration",
            query_type="semantic"
        )
        chunks, sources = [], []
        i = 0
        async for doc in results:
            title = doc.get("title", "N/A")
            chunk_content = doc.get("chunk", "N/A").replace("\n", " ").replace("\t", " ").strip()
            parent_id_decoded = safe_base64_decode(doc.get("parent_id", "Unknown Document"))
            chunk_id = start_index + i
            chunks.append({"id": chunk_id, "title": title, "chunk": chunk_content, "parent_id": parent_id_decoded})
            sources.append(f"Source ID: [{chunk_id}]\nContent: {chunk_content}\nDocument: {parent_id_decoded}")
            i += 1
        return chunks, sources

    history_chunks, history_sources = await fetch_chunks(history_queries, 5, 1)
    standalone_chunks, standalone_sources = await fetch_chunks(user_query, 5, 6)

    # Deduplicate chunks
    combined_chunks = history_chunks + standalone_chunks
    seen = set()
    all_chunks = []
    for chunk in combined_chunks:
        if chunk["chunk"] not in seen:
            seen.add(chunk["chunk"])
            all_chunks.append(chunk)
    sources_formatted = "\n\n---\n\n".join([f"Source ID: [{c['id']}]\nContent: {c['chunk']}\nDocument: {c['parent_id']}" for c in all_chunks])

    # -------------------------
    # Build AI prompt
    # -------------------------
    prompt = answer_prompt_template.format(conversation_history=conversation_history, sources=sources_formatted, query=user_query)

    response = await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=deployment_name,
        temperature=0.7
    )
    full_reply = response.choices[0].message.content.strip()

    # Standardize citations
    flat_ids = [int(n.strip()) for match in re.findall(r"\[(.*?)\]", full_reply) for n in match.split(",") if n.strip().isdigit()]
    unique_ids = []
    for i in flat_ids:
        if i not in unique_ids:
            unique_ids.append(i)
    id_mapping = {old_id: new_id+1 for new_id, old_id in enumerate(unique_ids)}

    def replace_citations(text, mapping):
        def repl(match):
            nums = [mapping.get(int(n.strip()), int(n.strip())) for n in match.group(1).split(",") if n.strip().isdigit()]
            return f"[{', '.join(map(str, sorted(set(nums))))}]"
        return re.sub(r"\[(.*?)\]", repl, text)

    ai_response = replace_citations(full_reply, id_mapping)

    # Update conversation and citations
    citations = []
    seen = set()
    for old_id in unique_ids:
        new_id = id_mapping[old_id]
        for chunk in all_chunks:
            if chunk["id"] == old_id and old_id not in seen:
                seen.add(old_id)
                updated_chunk = chunk.copy()
                updated_chunk["id"] = new_id
                citations.append(updated_chunk)

    conversation_store[user_id]["chat"] += f"\nUser: {user_query}\nAI: {ai_response}"

    # Follow-up questions
    follow_up_prompt = followup_prompt_template.format(citations=citations)
    follow_up_resp = await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": follow_up_prompt}],
        model=deployment_name
    )
    follow_ups_raw = follow_up_resp.choices[0].message.content.strip()

    return {"query": user_query, "ai_response": ai_response, "citations": citations, "follow_ups": follow_ups_raw}



