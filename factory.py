"""
Factory for creating different models: OpenAI, AzureOpenAI, Llama, etc.
"""
import os

from dotenv import dotenv_values

from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.llms import OpenAI

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

    def openai(self) -> OpenAI:
        """ Create OpenAI model. """
        return OpenAI(
            openai_api_key=self._config.get('OPENAI_API_KEY'),
        )

    def llama(self) -> LlamaCpp:
        """ Create wrapper of a Llama compatible model. """
        model_path = self._config.get('LLAMA_MODEL_PATH')
        return LlamaCpp(model_path=model_path)

    def llama_embeddings(self) -> LlamaCppEmbeddings:
        """ Create Llama embeddings. """
        model_path = self._config.get('LLAMA_MODEL_PATH')
        return LlamaCppEmbeddings(model_path=model_path)
