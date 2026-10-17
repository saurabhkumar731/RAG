from src.ytrag.data_loader import load_all_documents
from src.ytrag.vectorstore import FaissVectorStore
from src.ytrag.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    #docs = load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?",top_k=3))
    rag_serach=RAGSearch()
    query="What is attention mechanism?"
    summary=rag_serach.search_and_summarize(query,top_k=3)
    print("Summary:",summary)
    