"""
Factory for creating different models: OpenAI, AzureOpenAI, Llama, etc.
"""
import os

from dotenv import dotenv_values

from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAI, AzureOpenAI, LlamaCpp
from langchain.chat_models import ChatOpenAI

# Directory of this file
DIRNAME = os.path.dirname(__file__)


class Models:
    """ Factory for creating different models. """

    def __init__(self):
        # Load variables from .env file without setting environment variables
        env_file = os.path.join(DIRNAME, '.env')
        self._config = dotenv_values(env_file)

    def __str__(self):
        return f'Models({self._config})'

    def openai(self, model_name='text-davinci-003') -> OpenAI:
        """ Create OpenAI completion model. """
        return OpenAI(
            openai_api_key=self._config.get('OPENAI_API_KEY'),
            model_name=model_name,
        )

    def openai_chat(self, model_name='gpt-3.5-turbo') -> ChatOpenAI:
        """ Create OpenAI chat model. """
        return ChatOpenAI(
            openai_api_key=self._config.get('OPENAI_API_KEY'),
            model_name=model_name,
        )

    def openai_embeddings(self) -> OpenAIEmbeddings:
        """ Create OpenAI embeddings model. """
        return OpenAIEmbeddings(
            openai_api_key=self._config.get('OPENAI_API_KEY'),
        )

    def azure_openai_gpt(self) -> OpenAI:
        """ Create Azure OpenAI GPT model. """
        return AzureOpenAI(
            openai_api_key=self._config.get('AZURE_OPENAI_API_KEY'),
            openai_api_base=self._config.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_version=self._config.get('AZURE_OPENAI_API_VERSION'),
            deployment_name=self._config.get('AZURE_OPENAI_GPT_DEPLOYMENT'),
            model=self._config.get('AZURE_OPENAI_GPT_MODEL'),
            temperature=0,
        )

    def azure_openai_ada(self) -> OpenAI:
        """ Create Azure OpenAI Ada model. """
        return AzureOpenAI(
            openai_api_key=self._config.get('AZURE_OPENAI_API_KEY'),
            openai_api_base=self._config.get('AZURE_OPENAI_ENDPOINT'),
            openai_api_version=self._config.get('AZURE_OPENAI_API_VERSION'),
            deployment_name=self._config.get('AZURE_OPENAI_ADA_DEPLOYMENT'),
            model=self._config.get('AZURE_OPENAI_ADA_MODEL'),
            temperature=0,
        )

    def llama(self) -> LlamaCpp:
        """ Create wrapper of a Llama compatible model. """
        model_path = self._config.get('LLAMA_MODEL_PATH')
        return LlamaCpp(model_path=model_path)

    def llama_embeddings(self) -> LlamaCppEmbeddings:
        """ Create Llama embeddings. """
        model_path = self._config.get('LLAMA_MODEL_PATH')
        return LlamaCppEmbeddings(model_path=model_path)
