"""LLM backend: OpenRouter (the only provider).

The client implements the two-step dialogue: step 1 asks for a
human-readable decision tree; step 2 sends the step-1 response back as
assistant context together with a follow-up prompt asking for the Python
function. The raw response is the concatenation of the two responses
(three newlines, 25 hashes, three newlines).
"""

import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from ._tree import DIALOGUE_SEPARATOR

# Load the environment variables from the .env file closest to the working
# directory. The default search starts at this module's file location, which
# for an installed package is site-packages — never the user's project — so
# a project-local .env would be missed when the environment lives outside
# the project.
load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

class OpenRouterClient:
    """OpenRouter via its OpenAI-compatible API.

    Configuration through the environment variables ``OPENROUTER_API_KEY`` and
    ``OPENROUTER_MODEL``; ``OPENROUTER_BASE_URL`` optionally points the client at
    another OpenAI-compatible server (vLLM, LiteLLM, Ollama); checked lazily at call time so that importing the
    package never requires credentials.
    """

    def __init__(self, temperature: float, seed: int | None = None):
        self.temperature = temperature
        self.seed = seed

    def two_step(self, first_prompt: str, second_prompt: str) -> str:
        if "OPENROUTER_API_KEY" not in os.environ:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set; it is required "
                "for backend='openrouter'. Set it in the environment or a .env file."
            )
        if "OPENROUTER_MODEL" not in os.environ:
            raise ValueError(
                "OPENROUTER_MODEL environment variable is not set; it is required "
                "for backend='openrouter'. Set it in the environment or a .env file."
            )

        client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        openrouter_model = os.environ["OPENROUTER_MODEL"]

        # Initialize conversation history
        messages = [
            {"role": "user", "content": first_prompt},
        ]

        first_response = client.chat.completions.create(
            model=openrouter_model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
        )

        # Add assistant response to the conversation history
        messages.append({"role": "assistant", "content": first_response.choices[0].message.content})

        # User's next input
        messages.append({"role": "user", "content": second_prompt})

        # Second response
        second_response = client.chat.completions.create(
            model=openrouter_model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
        )

        return (
            first_response.choices[0].message.content
            + 3 * "\n"
            + DIALOGUE_SEPARATOR
            + 3 * "\n"
            + second_response.choices[0].message.content
        )


def make_client(backend: str, temperature: float, seed: int | None = None):
    if backend == "openrouter":
        return OpenRouterClient(temperature, seed)
    raise ValueError(f"Unknown backend: {backend}")
