import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

load_dotenv()


class OpenRouterLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://tokenharbor.ai/v1",
            api_key=os.getenv("TOKENHARBOR_API_KEY"),
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content
        if schema is None:
            return text
        # Pull the JSON object out of the reply and validate it against DeepEval's schema
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        return schema.model_validate_json(json_str)

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


judge = OpenRouterLLM("deepseek-v4.1-flash:free")

test_case = LLMTestCase(
    input="What do penguins eat?",
    actual_output="Penguins primarily feed on grass and ice cubes",
    expected_output="Penguins eat marine animals such as krill, fish and squid",
)

metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
metric.measure(test_case)

print("Score :", metric.score)
print("Reason:", metric.reason)
print("Passed:", metric.is_successful())

correctness = GEval(
    name="Correctness",
    criteria="Actual output is factually correct when compared to expected output.",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge,
)
correctness.measure(test_case)

print("Correctness score :", correctness.score)
print("Correctness reason:", correctness.reason)
print("Correctness passed:", correctness.is_successful())