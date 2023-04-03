from pathlib import Path
import pickle
from chatgpt import InstructionLLM

instr_path = Path('hiveformer_dataset/multi_task_dataset/instructions')
tasks = ["basketball_in_hoop", "slide_block_to_target", "wipe_desk", "lamp_off", "close_drawer", "turn_tap", "take_usb_out_of_computer", "turn_oven_on", "close_microwave"]

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    # 各タスクのinstructionを生成して表示するテスト
    # llm = InstructionLLM()
    # for task in tasks:
    #     path = instr_path / task / 'instructions.pkl'
    #     data = load_pickle(path)
    #     task_descriptions = data[task]['raw']
    #     print(task)
    #     for _ in range(5):
    #         print("-"*30)
    #         instruction = llm.get_instruction(task, task_descriptions)
    #         print(instruction)
    #     print()
    
    # eval logファイルから{task)name Reward Variation ...の行を抜き出すテスト
    path = "test.log"
    task_name = tasks[1]
    print(task_name)
    with open(path, 'r') as f:
        for line in f:
            string = task_name+' '
            if string in line:
                index = line.index(string)
                data = line[index:].split()
                step, reward, success_rate = int(data[6]), int(float(data[2])), int(float(data[-1]))
                print(f"Step: {step}, Success: {reward}, Success Rate: {success_rate}")
