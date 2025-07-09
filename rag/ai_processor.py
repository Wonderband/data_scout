import os
from openai import OpenAI
from db.retriever import search_hybrid, search_hybrid_multi

from openai import OpenAI

  # make sure OPENAI_API_KEY is in your env


def clean_prompt(query: str) -> str:
    """
    Extracts and normalizes only the essential data elements
    (dates, names, numbers, etc.) from the user’s raw prompt.
    """
    system_prompt = (
        "You are an expert information extractor and normalizer for a document database."
    )
    user_instructions = """
    When given a user’s query, you must:
    - Fix all mistakes in the text.
    - Identify and extract any of these items in the order they appear:
        • Company or organization names  
        • Dates (e.g. “квітень 2024”, “04/2024”, “2024-04-15”)  
        • Bank account numbers (IBANs like UA...)  
        • Monetary sums or prices (with or without currency symbols)  
        • Quantities (with or without units)  
        • Document types (e.g. invoice, contract, report, рахунок, виписка, акт)  
        • Any other essential data
    - Normalize dates to ISO format:
        • If only year+month are given, output YYYY‑MM.
        • If a full date is given, output YYYY‑MM‑DD.
        • If only a year is given, output YYYY.
    - Normalize numbers/sums: remove spaces/thousands‑separators, use a dot for decimals.
    - Output **only** the extracted, normalized values as a comma‑separated string, with no labels or extra words.
    - If no values found in the user query - return an empty string (`""`).

    Example:
    User’s prompt:
        Знайдіть мені рахунок-фактуру від Some Company за квітень 2024 
        на суму 1234,56 грн і банківський рахунок UA26 0076 5345 111
    Extractor’s output:
        рахунок-фактура, Some Company, 2024-04-01, 1234.56, UA2600765345111

    Now process the following user query and return just the comma‑separated, normalized essentials (or `""` if no essentials found):
    """
    # combine instructions and actual user query
    full_user_content = user_instructions + "\n\n" + query.strip()
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",  # or whichever model you prefer
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_content},
        ],
        max_tokens=500,
        temperature=0.0,
    )

    # now you can safely index into choices[0].message.content
    cleaned = response.choices[0].message.content.strip()
    return cleaned


def generate_final_response(user_prompt: str, context: str) -> str:
    """
    Combine the original user prompt and the retrieved context and ask OpenAI (or another LLM)
    to generate the final answer.
    """
    client = OpenAI()
    system_prompt = "You are a finance expert excelling in parsing finance documents and filtering relevant to the prompt."
    instructions = (
        "You are provided with the original user prompt and context information retrieved from "
        "a knowledge base. Process the context together with the user prompt to generate a concise and coherent answer.\n\n"
        "Insert into your answer the names of the files, relevant to user prompt and filter out irrelevant search results"
        "If you see that prompt contains mistakes when comparing with context information - fix that mistakes in your response and point at them"
        "User prompt: {0}\n\nContext:\n{1}\n\nAnswer:".format(user_prompt, context)
    )
    response = client.chat.completions.create(
        model="gpt-4o",  # or whichever model you prefer
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instructions},
        ],
        max_tokens=500,
        temperature=0.0,
    )

    # now you can safely index into choices[0].message.content
    final_answer = response.choices[0].message.content.strip()
    return final_answer


def perform_rag(user_prompt: str, base_dir: str):
    """
    Perform the custom Retrieval Augmented Generation:
      1. Clean the user prompt via OpenAI.
      2. Use the cleaned query to retrieve top results from Chroma DB.
      3. Provide both the original prompt and the context to OpenAI to generate the final answer.
    """
    # Step 1: Clean the prompt.
    cleaned_query = clean_prompt(user_prompt)
    if cleaned_query == "" or len(cleaned_query) < 3:
        return "Sorry, but your query contains no data filters to apply"

    # # Step 2: Retrieve top records from Chroma DB using the cleaned query.
    # # Assume search_hybrid returns a list of tuples with at least the document text.
    results = search_hybrid_multi(cleaned_query, base_dir, top_k=20)
    context_chunks = []
    for doc in results:
        # Adjust this formatting as needed; here we simply convert each returned tuple to a string.
        context_chunks.append(str(doc))
    context_str = "\n".join(context_chunks)

    # Step 3: Generate the final answer using the original prompt + retrieved context.
    final_response = generate_final_response(user_prompt, context_str)
    return final_response


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_module.py <user_prompt>")
    else:
        user_prompt = sys.argv[1]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        answer = perform_rag(user_prompt, base_dir)
        print("Final answer:")
        print(answer)
