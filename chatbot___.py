from coordinate import LlamaCpp, get_retriever, RetrievalQA


class ChatBot:
    def __init__(self):
        self.qa_chain = self.get_qa_chain_llama()

    def chatbot(self):
        print("🤖 Chatbot RAG (LLAMA local) sẵn sàng. Gõ 'exit' để thoát.")

        while True:
            query = input("❓ Bạn: ")
            if query.lower() == 'exit':
                break
            result = self.qa_chain({"query": query})
            print("🤖 Bot:", result["result"])

    def chatbot_gra(self, query):
        result = self.qa_chain({"query": query})
        return result["result"]

    def get_qa_chain_llama(self):
        llm = LlamaCpp(
            model_path="models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            n_ctx=2048,
            verbose=False,
            n_threads=1,  # TODO: change
            n_batch=64
        )

        retriever = get_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        return chain

if __name__ == "__main__":
    ChatBot().chatbot()
