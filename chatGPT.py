import sys
from chatgpt_wrapper import ChatGPT

bot = ChatGPT()
prompt = \
"""
What are the 4 steps for the robotic arm to place the rubbish in the bin?
The instructions must be short, informative, and concise and must include corrdinates.
Coordinates are given by (x, y, z).


Position: 
%s

Instruction:
"""

def post_processing(response):
    output_list = []
    sentenses = response.split("\n")
    for sentence in sentenses:
        if len(sentence) <= 0:
            continue
        if sentence[0].isdigit():
            output_list += [sentence]
    return '\n'.join(output_list)

def load_description(description_path):
    with open(description_path, "r") as f:
        description = f.read()
    return description

def save_instruction(instruction, save_path):
    with open(save_path, "w") as f:
        f.write(instruction)

description_path = sys.argv[1]
save_path = sys.argv[2]
# description_path = "../dataset2/raw/0/put_rubbish_in_bin/variation0/episodes/episode1/description.txt"
# save_path = "./test.txt"
description = load_description(description_path)
prompt = prompt % description

response = bot.ask(prompt)
instruction = post_processing(response)
save_instruction(instruction, save_path)

print("Prompt:")
print(prompt)
print("Answer")
print(instruction)


