# create question out of chunks of text
import json

from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

with open('src/bert.md', 'r') as f:
    text = f.read()
    
chunks = split_text_into_chunks(text)
print("Number of chunks: ", len(chunks))

OPENAI_API_KEY=input("Enter your OpenAI API Key: ")
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def prompt_template(chunk: str):
    prompt = f"""Create a question whose answer is the following text: \n\n {chunk}
    
    Instructions:
    - The question should be easy to answer given the text.
    - Don't make the question too long.
    - Only ask for information that is in the text. 
    - External knowledge is not required.
    
    Question: """
    
    return prompt

questions = []
answers = []
for chunk in chunks:
    prompt = prompt_template(chunk)    
    question = llm.invoke(prompt).strip()
    questions.append(question)
    answers.append(chunk)

# prepare test data, create a list of dictionaries
test_data = []
for i in range(len(questions)):
    test_data.append({
        'question': questions[i],
        'answer': answers[i]
    })
    
with open('src/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)