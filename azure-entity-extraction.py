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
You are an assistant designed to extract entities from text. Users will paste in a string of text and you will respond with entities you've extracted from the text as a JSON object. Here's an example of your output format:
{
   "name": "",
   "company": "",
   "phone_number": ""
}
"""
},
{
    "role": "user", "content": """
Hello. My name is Robert Smith. I'm calling from Contoso Insurance, Delaware. My colleague mentioned that you are interested in learning about our comprehensive benefits policy. Could you give me a call back at (555) 346-9322 when you get a chance so we can go over the benefits?
"""
}
])

print(response)
print(response['choices'][0]['message']['content'])


