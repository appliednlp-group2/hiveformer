import langchain
import openai
from langchain.llms import OpenAIChat
import json

import os
os.environ["OPENAI_API_KEY"] = ""

class InstructionLLM():
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = OpenAIChat(model_name=model_name)    
        with open('episodes.json', 'r') as f:
            self.eposodes = json.load(f)['max_episode_length']

    def get_instruction(self, task_name, task_descriptions):
        task_descriptions = "\n".join(task_descriptions)
        num_steps = self.eposodes[task_name]
        text =f"""
What are the {num_steps} steps for the robotic arm to do the following task?
The instructions must be short and informative and concise.

Task:
{task_descriptions}

Instruction:
"""
        # print(text)
        return self.llm(text)


if __name__ == '__main__':
    task_name = 'basketball_in_hoop'
    task_descriptions = ['put the ball in the hoop', 'play basketball', 'shoot the ball through the net', 'pick up the basketball and put it in the hoop', 'throw the basketball through the hoop', 'place the basket ball through the hoop']

    llm = InstructionLLM()
    instruction = llm.get_instruction(task_name, task_descriptions)
    print(repr(instruction))