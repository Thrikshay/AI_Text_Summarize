import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter



# 1. LOAD API KEY


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not found.")



# 2. INITIALIZE THE LLM


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=groq_api_key,
)



# 3. CREATE PROMPT TEMPLATE


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI Text Summarizer.

Rules:
- Summarize the text provided by the user.
- Do not add new information or opinions.
- Keep the important ideas, facts, and conclusions.
- Make the summary clear and easy to understand.
- Return the summary in 4 to 6 bullet points.
        """,
    ),

    (
        "human",
        """
Summarize the following text:

{text}
        """,
    ),
])


# 4. CREATE OUTPUT PARSER


parser = StrOutputParser()



# 5. CREATE LANGCHAIN CHAIN


summary_chain = prompt | llm | parser



# 6. CREATE TEXT SPLITTER


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)


# 7. GET TEXT FROM USER


def get_user_text():

    print("\nAI TEXT SUMMARIZER")
    print("----------------------")

    print("\nEnter or paste the text you want to summarize.")
    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()

    if not text:

        print("\nError: Please enter some text.")

        return None

    return text



# 8. SUMMARIZE TEXT


def summarize_text(text):

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Store summaries of every chunk
    summaries = []

    print(f"\nNumber of chunks created: {len(chunks)}")

    # Summarize every chunk
    for i, chunk in enumerate(chunks, start=1):

        print(f"Summarizing chunk {i}...")

        summary = summary_chain.invoke({
            "text": chunk
        })

        summaries.append(summary)

    # If only one chunk was created,
    # directly return its summary
    if len(summaries) == 1:

        return summaries[0]

    # Combine summaries of all chunks
    combined_summary = "\n".join(summaries)

    # Summarize the combined summaries again
    final_summary = summary_chain.invoke({
        "text": combined_summary
    })

    return final_summary



# 9. MAIN PROGRAM


def main():

    user_text = get_user_text()

    if user_text is None:
        return

    print("\nGenerating Summary...\n")

    summary = summarize_text(user_text)

    print("\n==============================")
    print("        FINAL SUMMARY")
    print("==============================\n")

    print(summary)



# 10. RUN PROGRAM


if __name__ == "__main__":
    main()