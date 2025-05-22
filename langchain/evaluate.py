# Step 0: Set env
import langchain
import os

from dotenv import load_dotenv

load_dotenv()
persist_directory = os.getenv("PERSIST_DIRECTORY")
os.environ["OPENAI_API_KEY"] = os.getenv("SECRET_KEY")

# Step 1: Bring text files
# (local 임시 텍스트 -> 나중에 DB에서 정답 txt, 답안 txt 받아오기)

with open("../dataset/answer_key.txt", "r", encoding="utf-8") as f:
    docs_answer_key = f.read()

with open("../dataset/student_answer.txt", "r", encoding="utf-8") as f:
    docs_student_answer = f.read()

# Step 2: Retrieval ~ Generation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """Evaluate <student answer> based on the <answer key>.

- In three areas: Missing, Incorrect, and Summary. 
- Feedback is based on the answer key.
- Don't evaluate information that's not in the answer key.
- Recognize that foreign proper nouns may be spelled differently.
- Please write in markdown format and Korean.

<Answer key>:
{answer_key}

<Student answer>:
{student_answer}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)

rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()

result = rag_chain.invoke(
    {"answer_key": docs_answer_key, "student_answer": docs_student_answer}
)
print("\nresult:\n", result)
