#Note: The openai-python library support for Azure OpenAI is in preview.
import os

from dotenv import load_dotenv, find_dotenv
import openai

load_dotenv(find_dotenv())

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

response = openai.ChatCompletion.create(
    max_tokens=1000,
    engine="gpt35", # engine = "deployment_name".
    messages=[
{
    "role": "system", "content": """
You are an assistant designed to extract entities from medical prescription. Users will paste in a string of prescription text and you will respond with entities you've extracted from the text as a JSON object. Here's an example of your output format:
{
   "drug_name": "",
   "drug_strength": "",
   "drug_amount": "",
   "drug_frequency_per_day": ""
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


