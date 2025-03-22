"""
1. describe what the [CLASS NAME] looks like:
2. describe the appearance of the [CLASS NAME]:
3. describe the color of the [CLASS NAME]:
4. describe the pattern of the [CLASS NAME]:
5. describe the shape of the [CLASS NAME]:
"""

from pydantic import BaseModel
from typing import List
import ollama
import json
import os
from openai import OpenAI

client = OpenAI()


MODEL_TO_USE = "OPENAI-4O"

# (1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8,
# Pneumothorax; 9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13,
# Pleural_Thickening; 14 Hernia)
x_ray_classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]


class LLMOptions:
    """
    Ensure that these are installed in the system
    """

    LLAMA = "llama3.1:8b"
    DEEPSEEK = "deepseek-r1:7b"
    MEDITRON = "meditron:7b"


class ConceptOutput(BaseModel):
    concepts: list[str]


feature_dimensions = ["appearance", "color", "pattern", "shape"]

class2concepts = {}

for class_name in x_ray_classes:
    class2concepts[class_name] = []
    for feature in feature_dimensions:
        for i in range(1):
            if MODEL_TO_USE == "OPENAI-4O":
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract the concepts from the class to be used in a Concept Bottleneck Model.",
                        },
                        {
                            "role": "user",
                            "content": "Describe the {} of the {} disease in X-Ray that can be used as concepts in my Concept Bottleneck Model for X-Ray disease classification.".format(
                                feature, class_name
                            ),
                        },
                    ],
                    response_format=ConceptOutput,
                )
                print(response.choices[0].message.content)
                os._exit(0)
            else:
                response = ollama.generate(
                    model=LLMOptions.LLAMA,
                    prompt=f"Describe the {feature} of the {class_name} disease in X-Ray that can be used as concepts in my Concept Bottleneck Model for X-Ray disease classification.",
                    format=ConceptOutput.model_json_schema(),
                )

                try:
                    response = ConceptOutput.model_validate_json(response["response"])

                    print(response.concepts)

                    class2concepts[class_name].extend(response.concepts)
                except Exception as e:
                    print("Unable to parse response, skipping")
                    print(e)

# make each value unique
for key, value in class2concepts.items():
    class2concepts[key] = list(set(value))

# for each class, filter out the concepts that are overly similar using a similarity threshold
# and keep only the most unique concepts


print(class2concepts)

# save in /home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts.json
with open(
    "/home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts.json",
    "w",
) as f:
    json.dump(class2concepts, f)
