from langchain_community.vectorstores import FAISS
from vectordb.store_data import DataStorage
from ollama import chat
from ollama import ChatResponse
import logging
import numpy as np
import faiss
import multiprocessing
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

path = "/tmp/.faiss_index"

class Retrieval():
    def __init__(self):
        self.model_id = "llama3.1:8b"
        self.datastorage = DataStorage()
        self.index = faiss.read_index(f"{path}/index.faiss")

        self.vectordb = FAISS.load_local(
            path, self.datastorage.model, allow_dangerous_deserialization=True)
        self.docs = self.vectordb.docstore._dict

    def generate_questions(self, query):
        ai_role_prompt = r"""
            You are a smart and helpful assistant specialized in generating follow-up and deeper questions based on an initial question provided by the user.

            Your role:
            - Read and understand the user's original question.
            - Identify its topic, scope, intent, and any ambiguity.
            - Generate 2 to 5 new questions that explore, clarify, or deepen the original question.

            Instructions:
            - Assume the user is seeking to learn, analyze, or reflect more deeply on the topic.
            - Do NOT answer or rephrase the original question.
            - Do NOT provide introductions, explanations, or summaries—only list the new questions.
            - Do NOT go beyond the scope of the original topic.
            - Keep questions directly relevant, open-ended when appropriate, and thought-provoking.
            - If the original question is vague or broad, include a clarifying question as part of your output.

            Output format:
            - Output only the questions, numbered from 1 to 5.
            - Do not include any headings, greetings, comments, or filler content.

            Example input:
            Question: "How does artificial intelligence impact education?"

            Example output:
            1. In what specific ways can AI enhance personalized learning?
            2. What challenges do schools face when integrating AI tools?
            3. How might AI change the role of teachers in the classroom?
            4. Are there ethical concerns regarding AI-based decision-making in education?
        """
       
        messages = [
            {"role": "system", "content": ai_role_prompt},
            {"role": "user", "content": query},
        ]

        logging.info("Generating questions")
        response: ChatResponse = chat(model=self.model_id, messages=messages)
        questions = []
        for question in response.message.content.split("\n"):
            print(question)
            questions.append(question)
        logging.info("Questions generated")
        return questions

    def generate_answers(self, context_text, query):
        ai_role_prompt = f"""
            You are an assistant that answers questions based ONLY on a list of context documents.

            Context documents:
            {context_text}

            Question:
            {query}

            Instructions:
            - Use ONLY the information provided in the context documents to answer the question.
            - Do not use any knowledge that is not explicitly found in the context documents, even if you know the answer.
            - It is strictly forbidden to make assumptions, inferences, or deductions based on general or prior knowledge.
            - If the answer is not clearly available in the context, respond exactly with: "The context does not provide enough information to answer this question."
            - Do not act as an expert or use knowledge outside of the provided context.
            - Keep the answer clear, concise, and based only on the documents.

            IMPORTANT: Do not deviate from the provided context under any circumstances and ONLY use the information from the provided context.
        """


        messages = [
            {"role": "system", "content": ai_role_prompt},
            {"role": "user", "content": query},
        ]

        logging.info("Generating answer")
        response: ChatResponse = chat(model=self.model_id, messages=messages)
        logging.info("Answer generated")
        return response.message.content

    def search_documents(self, context, k=10):
        def similarity(context, results):
            vector = self.datastorage.model.embed_query(context)
            D, I = self.index.search(np.array([vector]).astype('float32'), k)
            for index, distance in zip(I[0], D[0]):
                if index not in results or results[index] > distance:
                    results[index] = distance
            return results

        results = {}
        if type(context) == str:
            similarity(context, results)
        else: 
            for question in context:
                similarity(question, results)
                
        return results

    def no_context(self, query):
        results = self.search_documents(query)
        documents = [list(self.docs.values())[idx].page_content for idx in results]
        logging.info(documents)
        answer = self.generate_answers(documents, query)
        return answer

    def multi_query(self, query):
        questions = self.generate_questions(query)
        results = self.search_documents(questions)
        documents = [list(self.docs.values())[idx].page_content for idx in results]
        answer = self.generate_answers(documents, query)
        return answer

    def rag_fusion(self, query):
        questions = self.generate_questions(query)
        results = self.search_documents(questions)
        sorted_indices = sorted(results, key=results.get)
        documents = [list(self.docs.values())[idx].page_content for idx in sorted_indices]
        answer = self.generate_answers(documents, query)
        return answer
    
    def query_decomposition(self, query):
        questions = self.generate_questions(query)
        context = []
        for question in questions:
            documents = self.search_documents(question)
            documents = [list(self.docs.values())[idx].page_content for idx in documents]
            for document in documents:
                if document not in context:
                    context.append(document)

        answer = None
        for ctx, question in zip(context, questions):
            print(f"{ctx} and {question}")
            if answer is not None:
                print("new context added")
                ctx = ctx + "\n" + answer
                print(answer)
            answer = self.generate_answers(ctx, question)

    def step_back(self, query):
        examples = [
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?"
            }, 
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?"
            }
        ]

        ai_role_prompt = rf"""
            You are a helpful AI assistant. Your task is to help improve the document retrieval step in a Retrieval-Augmented Generation (RAG) system by applying a "step-back" strategy.
            Given a user's specific question, take a step back and rewrite it as a more general or explanatory version that reflects the broader topic or intent behind the question.

            Then, use that broader question to retrieve background documents or context before answering the original query.

            Instructions:
            1. Read the original question carefully.
            2. Reformulate it as a more general question that provides helpful context for retrieval.
            3. Return only the reformulated broader query.

            Original question: {query}
            These are few examples: {examples}
        """

        messages = [
            {
                "role": "system", 
                "content": ai_role_prompt
            }, 
            {
                "role": "user",
                "content": query
            }
        ]

        logging.info(f"Generating question from {query}")
        response: ChatResponse = chat(model=self.model_id, messages=messages)
        logging.info("Question generated")
        questions = self.search_documents(response.message.content)
        documents = [list(self.docs.values())[idx].page_content for idx in questions]
        answer = self.generate_answers(documents, query)
        return answer

    def hyde_query(self, query):
        ai_role_prompt = rf"""Please write a scientific paper passage to answer the question
            Question: {query}
            Passage:
        """

        messages = [
            {
                "role": "system", 
                "content": ai_role_prompt
            }, 
            {
                "role": "user",
                "content": query
            }
        ]

        logging.info(f"Generating paper from question: {query}")
        paper: ChatResponse = chat(model=self.model_id, messages=messages)
        logging.info("Paper generated")
        print(f"Lenght: {len(paper.message.content)}\tPaper: {paper.message.content}")
        near_documents = self.search_documents(paper.message.content)   
        documents = [list(self.docs.values())[idx].page_content for idx in near_documents]
        answer = self.generate_answers(documents, query)
        return answer
    
if __name__ == "__main__":
    print(Retrieval().no_context("Why does Elizabeth Bennet feel “no very cordial feelings” toward Mr. Darcy after their first meeting at the ball?"))