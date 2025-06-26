from coordinate import get_qa_chain_llama as get_qa_chain

def chatbot():
    qa_chain = get_qa_chain()
    print("🤖 Chatbot RAG (LLAMA local) sẵn sàng. Gõ 'exit' để thoát.")

    while True:
        query = input("❓ Bạn: ")
        if query.lower() == 'exit':
            break
        result = qa_chain({"query": query})
        print("🤖 Bot:", result["result"])

if __name__ == "__main__":
    chatbot()
