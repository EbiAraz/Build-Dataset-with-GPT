from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm



load_dotenv()

client = OpenAI(api_key = os.getenv("METIS_API_KEY") ,
                 base_url="https://api.metisai.ir/openai/v1")


pdf_path = "data/book.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(documents)


def create_prompt(text):

    prompt = f"""
    You are a dataset Generator.

    Convert the following text into Alpaca instruction format.

    Rules:

    -Return ONLY JSON
    -No markdown
    -No Explanation

    Each item must contain:

    instruction
    input
    output

    Generate 5 examples

    text : 
    {text}
    """
    return prompt


def generate_examples(text):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
            messages=[
                {
                "role":"user",
                "content":create_prompt(text)
                }
            ],
        temperature=0.3
        )

    return response.choices[0].message.content

    

def clean_json(output):

    output = output.replace("```json","")
    output = output.replace("```","")

    output = output.strip()

    return json.loads(output)





dataset = []

for chunk in tqdm(chunks):

    text = chunk.page_content

    try:
        result = generate_examples(text)

        examples = clean_json(result)

        dataset.extend(examples)

    except Exception as e:
        print("Error : ", e)



with open(
    "output/alpaca.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(dataset,f,indent=2,ensure_ascii=False)

print(
    "Dataset Size : ",
    len(dataset)
)




print("Number of chunks : ", len(chunks))

print("Pages : ",len(documents))
print(documents[0].page_content[:500])