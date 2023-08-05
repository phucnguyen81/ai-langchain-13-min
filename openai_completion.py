""" Completion model of OpenAI. """
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from factory import Models


if __name__ == '__main__':
    models = Models()

    # Default completion model (davinci model)
    llm = models.openai()
    response = llm('explain large language models in one sentence')
    print(response)

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

    # Use PromptTemplate
    template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""
    prompt = PromptTemplate(
        input_variables=['concept'],
        template=template,
    )
    llm(prompt.format(concept='autoencoder'))

    # Define a chain with language model and prompt as arguments.
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    print(chain.run('autoencoder'))

    # Define a second chain
    second_prompt = PromptTemplate(
        input_variables=['ml_concept'],
        template=(
            "Turn the concept description of {ml_concept} and "
            "explain it to me like I'm five in 500 words"
        )
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)
    print(chain_two.run('algorithm_concept'))

    # Define a sequential chain using the two chains above: the second
    # chain takes the output of the first chain as input
    overall_chain = SimpleSequentialChain(
        chains=[chain, chain_two], verbose=True)

    # Run the chain specifying only the input variable for the first chain.
    explanation = overall_chain.run('autoencoder')
    print(explanation)
