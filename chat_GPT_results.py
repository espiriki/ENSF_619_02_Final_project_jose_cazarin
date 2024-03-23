from openai import OpenAI
import os
import re
from os import listdir
from os.path import isfile, join
from pathlib import Path
import json
import sys

API_KEY = "ADD_YOUR_API_HERE"
MODEL = "gpt-4-0125-preview"
# MODEL = "gpt-3.5-turbo-0125"
# MODEL = "gpt-4-0613"

def pre_process_text(text):
    text = text.lower()

    text = text.replace("_", " ")

    # Remove the digits
    pattern_remove_digits = r"[0-9]"
    text = re.sub(pattern_remove_digits, "", text)

    # Remove any symbols that are left, keeping spaces
    pattern_remove_symbols = r"[^a-zA-Z ]+"
    text = re.sub(pattern_remove_symbols, "", text)

    return text.strip()


chat_gpt_model = MODEL
client = OpenAI(api_key=API_KEY)
chat_history = []

PATH_TO_DATASET = "<PATH_TO_DATASET>/CVPR_2024_dataset_Test"

initial_prompts = [
    "You are an specialist in recycling now.",
    "I want you to classify an object into 4 categories of recycling.\n\
blue: recyclable\n\
green: compostable\n\
black: non-recyclable, will go to a landfill\n\
ttr: everything else that needs special treatment and cannot be disposed of in a regular bin.\n\
When I send you an comma-separated list of object descriptions, you will classify each description into one of those four categories. \
Your response will be a comma-separated list of category names, which is blue, green, black or ttr. Your response cannot be empty.",
]


def get_response(sample):
    response = get_chat_gpt_response(sample)
    response = response.split()
    response = response[0]
    return response.lower()


def get_chat_gpt_response(prompt, model=chat_gpt_model):
    global chat_history

    chat_history.append(prompt)

    messages = []
    for msg in chat_history:
        messages.append({"role": "user", "content": msg})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
    )

    return response.choices[0].message.content


for prompt in initial_prompts:
    response = get_chat_gpt_response(prompt)

print("Chunk: ", sys.argv[1])
print("Class: ", sys.argv[2])
start_chunk_idx = int(sys.argv[1])
current_class = [str(sys.argv[2])]

for _class in current_class:
    path = os.path.join(PATH_TO_DATASET, _class)
    samples_for_current_class = [
        f for f in listdir(path) if isfile(join(path, f))]

    samples_for_current_class.sort()
    full_prompt = ""
    for sample in samples_for_current_class[
        (start_chunk_idx * 10): (start_chunk_idx * 10) + 10
    ]:
        sample = pre_process_text(Path(sample).stem)
        full_prompt = full_prompt + sample + ", "

    full_prompt = full_prompt[:-1]
    print("full prompt: ", full_prompt)
    response = get_chat_gpt_response(full_prompt)
    print("response: ", response)

    list_of_responses = response.split(",")

    print("num of responses: ", len(list_of_responses))

output_dict = {}

full_prompt = full_prompt.split(",")

out_json = []
for prompt, chat_gpt in zip(full_prompt, list_of_responses):
    output_dict = {}
    output_dict["description"] = prompt
    output_dict["GT"] = _class.lower()
    output_dict["chat_gpt_response"] = chat_gpt
    out_json.append(output_dict)

file_name = os.path.join(
    "results_chat_gpt",
    "results_chat_gpt_chunk_{}_GT_{}.txt".format(start_chunk_idx, sys.argv[2]),
)

try:

    f = open(file_name, "w")
    f.write(json.dumps(out_json, indent=2))
    f.close()
except Exception as e:
    print(e)
    sys.exit(1)
