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


class LLMOptions:
    """
    Ensure that these are installed in the system
    """

    LLAMA = "llama3.1:8b"
    DEEPSEEK = "deepseek-r1:7b"
    MEDITRON = "meditron:7b"
    OPENAI_4O = "gpt-4o-2024-08-06"


DATASET_TO_USE = "COVIDX"
MODEL_TO_USE = LLMOptions.OPENAI_4O


class ConceptOutput(BaseModel):
    concepts: list[str]


feature_dimensions = ["appearance", "color", "pattern", "shape"]


if DATASET_TO_USE == "HAM10000":
    print("Extracting concepts for HAM10000 dataset")
    class2concepts = {}
    class2concepts_granular = {}

    ham10000_classes = [
        "actinic keratoses",
        "basal cell carcinoma",
        "benign keratosis-like lesions",
        "dermatofibroma",
        "melanocytic nevi",
        "melanoma",
        "vascular lesions",
    ]

    for class_name in ham10000_classes:
        class2concepts[class_name] = []
        class2concepts_granular[class_name] = {}
        for feature in feature_dimensions:
            class2concepts_granular[class_name][feature] = []
            print(
                "Extracting concepts for class: {} and feature: {}".format(
                    class_name, feature
                )
            )
            for i in range(5):
                if MODEL_TO_USE == LLMOptions.OPENAI_4O:
                    response = client.beta.chat.completions.parse(
                        model=LLMOptions.OPENAI_4O,
                        messages=[
                            {
                                "role": "system",
                                "content": "Extract the concepts from the class for Skin Cancer classification from dermatoscopic images.",
                            },
                            {
                                "role": "user",
                                "content": "Describe the {} of the {} disease that can be used as visual concepts for Skin Cancer classification.".format(
                                    feature, class_name
                                ),
                            },
                        ],
                        response_format=ConceptOutput,
                    )
                    print(response.choices[0].message.parsed.concepts)
                    class2concepts[class_name].extend(
                        response.choices[0].message.parsed.concepts
                    )
                    class2concepts_granular[class_name][feature].extend(
                        response.choices[0].message.parsed.concepts
                    )
                else:
                    response = ollama.generate(
                        model=LLMOptions.LLAMA,
                        prompt=f"Describe the {feature} of the {class_name} disease for Skin Cancer classification.",
                        format=ConceptOutput.model_json_schema(),
                    )

                    try:
                        response = ConceptOutput.model_validate_json(
                            response["response"]
                        )

                        print(response.concepts)

                        class2concepts[class_name].extend(response.concepts)
                        class2concepts_granular[class_name][feature].extend(
                            response.concepts
                        )
                    except Exception as e:
                        print("Unable to parse response, skipping")
                        print(e)
        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/HAM10000/concepts_generalist/class2concepts_v2.json",
            "w",
        ) as f:
            json.dump(class2concepts, f)

        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/HAM10000/concepts_generalist/class2concepts_granular_v2.json",
            "w",
        ) as f:
            json.dump(class2concepts_granular, f)

    # make each value unique
    for key, value in class2concepts.items():
        class2concepts[key] = list(set(value))

    for key, value in class2concepts_granular.items():
        for feature, concepts in value.items():
            class2concepts_granular[key][feature] = list(set(concepts))

    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/HAM10000/concepts_generalist/class2concepts_v2.json",
        "w",
    ) as f:
        json.dump(class2concepts, f)

    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/HAM10000/concepts_generalist/class2concepts_granular_v2.json",
        "w",
    ) as f:
        json.dump(class2concepts_granular, f)

    print(f"class2concepts", class2concepts)
    print(f"class2concepts_granular", class2concepts_granular)

elif DATASET_TO_USE == "COVIDX":
    covidx_classes = ["Lung_Opacity", "Normal", "Viral Pneumonia", "COVID"]
    class2concepts = {}
    class2concepts_granular = {}

    for class_name in covidx_classes:
        class2concepts[class_name] = []
        class2concepts_granular[class_name] = {}
        for feature in feature_dimensions:
            class2concepts_granular[class_name][feature] = []
            print(
                "Extracting concepts for class: {} and feature: {}".format(
                    class_name, feature
                )
            )
            for i in range(5):
                if MODEL_TO_USE == LLMOptions.OPENAI_4O:
                    response = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {
                                "role": "system",
                                "content": "Extract the concepts from the class to be used in a Concept Bottleneck Model.",
                            },
                            {
                                "role": "user",
                                "content": "Describe the {} of the {} disease in X-Ray that can be used as concepts for X-Ray classification.".format(
                                    feature.replace("_", " "), class_name
                                ),
                            },
                        ],
                        response_format=ConceptOutput,
                    )
                    print(response.choices[0].message.parsed.concepts)
                    class2concepts[class_name].extend(
                        response.choices[0].message.parsed.concepts
                    )
                    class2concepts_granular[class_name][feature].extend(
                        response.choices[0].message.parsed.concepts
                    )

                else:
                    response = ollama.generate(
                        model=LLMOptions.LLAMA,
                        prompt=f"Describe the {feature} of the {class_name} disease in X-Ray that can be used as concepts for X-Ray classification.",
                        format=ConceptOutput.model_json_schema(),
                    )

                    try:
                        response = ConceptOutput.model_validate_json(
                            response["response"]
                        )

                        print(response.concepts)

                        class2concepts[class_name].extend(response.concepts)
                        class2concepts_granular[class_name][feature].extend(
                            response.concepts
                        )
                    except Exception as e:
                        print("Unable to parse response, skipping")
                        print(e)
        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/COVIDX/concepts/concepts_generalist/class2concepts.json",
            "w",
        ) as f:
            json.dump(class2concepts, f)

        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/COVIDX/concepts/concepts_generalist/class2concepts_granular.json",
            "w",
        ) as f:
            json.dump(class2concepts_granular, f)

    # make each value unique
    for key, value in class2concepts.items():
        class2concepts[key] = list(set(value))

    for key, value in class2concepts_granular.items():
        for feature, concepts in value.items():
            class2concepts_granular[key][feature] = list(set(concepts))

    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/COVIDX/concepts/concepts_generalist/class2concepts.json",
        "w",
    ) as f:
        json.dump(class2concepts, f)

    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/COVIDX/concepts/concepts_generalist/class2concepts_granular.json",
        "w",
    ) as f:
        json.dump(class2concepts_granular, f)

    print(f"class2concepts", class2concepts)
    print(f"class2concepts_granular", class2concepts_granular)


elif DATASET_TO_USE == "XRAY_NIH":
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

    for class_name in x_ray_classes:
        class2concepts[class_name] = []
        class2concepts_granular[class_name] = {}
        for feature in feature_dimensions:
            class2concepts_granular[class_name][feature] = []
            print(
                "Extracting concepts for class: {} and feature: {}".format(
                    class_name, feature
                )
            )
            for i in range(5):
                if MODEL_TO_USE == LLMOptions.OPENAI_4O:
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
                    print(response.choices[0].message.parsed.concepts)
                    class2concepts[class_name].extend(
                        response.choices[0].message.parsed.concepts
                    )
                    class2concepts_granular[class_name][feature].extend(
                        response.choices[0].message.parsed.concepts
                    )
                else:
                    response = ollama.generate(
                        model=LLMOptions.LLAMA,
                        prompt=f"Describe the {feature} of the {class_name} disease in X-Ray that can be used as concepts in my Concept Bottleneck Model for X-Ray disease classification.",
                        format=ConceptOutput.model_json_schema(),
                    )

                    try:
                        response = ConceptOutput.model_validate_json(
                            response["response"]
                        )

                        print(response.concepts)

                        class2concepts[class_name].extend(response.concepts)
                        class2concepts_granular[class_name][feature].extend(
                            response.concepts
                        )
                    except Exception as e:
                        print("Unable to parse response, skipping")
                        print(e)
        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts.json",
            "w",
        ) as f:
            json.dump(class2concepts, f)

        with open(
            "/home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts_granular.json",
            "w",
        ) as f:
            json.dump(class2concepts_granular, f)

    # make each value unique
    for key, value in class2concepts.items():
        class2concepts[key] = list(set(value))

    for key, value in class2concepts_granular.items():
        for feature, concepts in value.items():
            class2concepts_granular[key][feature] = list(set(concepts))

    # save in /home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts.json and class2concepts_granular.json
    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts.json",
        "w",
    ) as f:
        json.dump(class2concepts, f)

    with open(
        "/home/sn666/explainable_ai/LaBo/datasets/XRAY_NIH/concepts/class2concepts_granular.json",
        "w",
    ) as f:
        json.dump(class2concepts_granular, f)

    print(f"class2concepts", class2concepts)
    print(f"class2concepts_granular", class2concepts_granular)
