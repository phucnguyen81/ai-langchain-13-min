#Note: The openai-python library support for Azure OpenAI is in preview.
import os

from dotenv import load_dotenv, find_dotenv
import openai

load_dotenv(find_dotenv())

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

response = openai.ChatCompletion.create(
    engine="gpt35-16k",  # engine is the deployment name on Azure Portal
    max_tokens=4000,
    temperature=0,
    messages=[
{
    "role": "system", "content": """
You are an assistant designed to extract entities from medical prescription. Users will paste in a string of prescription text and you will respond with entities you've extracted from the text as a JSON object. Here's an example of your output format:
{
   "drug_name": "",
   "drug_strength": "",
   "drug_amount": "",
   "drug_unit": "",
   "drug_frequency": "",
   "drug_time_unit": "",
}
"""
},
{
    "role": "user", "content": """
YESOM 40 40mg (Esomeprazol 40mg)
Ngày uống 3 lần, mỗi lần 1 Viên
"""
},
{
    "role": "assistant", "content": """
{
   "drug_name": "YESOM 40",
   "drug_strength": "40mg",
   "drug_amount": "1",
   "drug_unit": "Viên",
   "drug_frequency": "3",
   "drug_time_unit": "Ngày",
}
"""
},
{
    "role": "user", "content": """
SUCRATE GEL (Sucralfate 1g (goi))
Ngày uống 3 lần, mỗi lần 1 GÓI
"""
},
])

print(response)
print(response['choices'][0]['message']['content'])
