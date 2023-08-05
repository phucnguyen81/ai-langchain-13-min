""" Chat model of OpenAI. """
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

from factory import Models


if __name__ == '__main__':
    models = Models()

    # Default chat model (gpt model)
    chat = models.openai_chat()
    messages = [
        SystemMessage(content='You are an expert data scientist'),
        HumanMessage(content=(
            'Write a Python script that trains a neural network'
            ' on simulated data'
        ))
    ]
    response=chat(messages)
    print(response.content, end='\n')
