import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic
from dotenv import load_dotenv

# Charger les variables du fichier .env (notamment ANTHROPIC_API_KEY)
load_dotenv()

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": i + 1})
    return pages

def chunk_text(pages, size=800, overlap=150):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
            start += size - overlap
    return chunks

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

def index_chunks(chunks):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts).tolist()
    collection.upsert(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"page": c["page"]} for c in chunks],
    )

# ÉTAPE 5 : Le Retrieval (retrouver les chunks les plus proches de la question)
def retrieve(question, k=2):
    q_vec = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    return [
        {"text": doc, "page": meta["page"]}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]

# ÉTAPE 6 : Appel au LLM (Claude) pour générer la réponse à partir du contexte
llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM = """Tu es un assistant qui répond aux questions à partir de documents.
Règles :
- Réponds UNIQUEMENT à partir du contexte fourni entre les balises <contexte>.
- Si la réponse n'y figure pas, dis exactement : "Je ne trouve pas cette information dans le document."
- Cite toujours les pages utilisées, sous la forme (p. X).
- Réponds en français, de façon concise."""

def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)
    
    msg = llm.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"<contexte>\n{context}\n</contexte>\n\nQuestion : {question}",
        }],
    )
    return msg.content[0].text

if __name__ == "__main__":
    # Indexation au démarrage si la base est vide
    if collection.count() == 0:
        print("Indexation du document...")
        pages = load_pdf("document.pdf")
        index_chunks(chunk_text(pages))
        print(f"{collection.count()} chunks indexés.")
    
    # Boucle de discussion interactive avec le LLM
    while True:
        q = input("\nPosez votre question sur le PDF (ou 'q' pour quitter) : ")
        if q.strip().lower() == "q":
            break
        print("\nRéponse de Claude :")
        print(answer(q))