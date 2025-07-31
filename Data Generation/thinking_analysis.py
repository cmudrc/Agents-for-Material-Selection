# Force cuda usage
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

##########################################################################################

import llama_cpp
from transformers import ReactCodeAgent
import regex as re
import pandas as pd
from data_generation_helper import WIKIPEDIA_SEARCH_PROMPT, compile_question, append_results
import logging
from transformers import Tool
import langchain_community.utilities.wikipedia

##########################################################################################

class WikipediaSearch(Tool):
    name = "wikipedia_search"
    description = "Search Wikipedia, the free encyclopedia."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search terms",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, query: str) -> str:
        wikipedia_api = langchain_community.utilities.wikipedia.WikipediaAPIWrapper(top_k_results=5)
        answer = wikipedia_api.run(query)
        return answer

##########################################################################################
# Code adapted from: https://github.com/grndnl/llm_material_selection_jcise/blob/main/generation/generation.py

materials = ['aluminum', 'glass', 'wood', 'elastomer']

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s',
handlers=[logging.FileHandler(f'thinking_logs.log'), logging.StreamHandler()]
)
logger = logging.getLogger()

def extract_content_from_memory(memory):
    return [entry['content'] for entry in memory if 'content' in entry]

MAX_TRIES = 5

for modelsize in [
                  '1.7',
                  '4',
                  '8',
                  '14',
                  '32'
                  ]:

    # Create LLM object
    llm = llama_cpp.Llama.from_pretrained(
        repo_id=f'bartowski/Qwen_Qwen3-{modelsize}B-GGUF',
        filename=f'Qwen_Qwen3-{modelsize}B-Q4_K_M.gguf',
        n_ctx=4096,
        gpu=True,
        metal=True,
        n_gpu_layers=-1
    )

    # Define a function to call the LLM and return output
    def llm_engine(messages, max_tokens=1000, stop_sequences=['Task']) -> str:
        global completion_tokens, total_tokens
        response = llm.create_chat_completion(
            messages=messages,
            stop=stop_sequences,
            max_tokens=max_tokens,
            temperature=0.6
        )
        answer = response['choices'][0]['message']['content']
        return answer
    
    # Create agent equipped with search tool
    websurfer_agent = ReactCodeAgent(
    system_prompt=WIKIPEDIA_SEARCH_PROMPT,
    tools=[WikipediaSearch()],
    llm_engine=llm_engine,
    add_base_tools = False,
    verbose = False,
    max_iterations=10
    )
    
    for question_type in [
        'agentic', 
        'zero-shot', 
        'few-shot', 
        'parallel', 
        'chain-of-thought'
        ]:
        results = pd.DataFrame(columns=['design', 'criteria', 'material', 'response'])
        for design, criterion in [('kitchen utensil grip', 'lightweight'), ('safety helmet', 'heat resistant'), ('underwater component', 'corrosion resistant'), ('spacecraft component', 'high strength')]:
            if question_type == 'parallel':
                material = ', '.join(materials)
                question = compile_question(design, criterion,  material, question_type)
                tries = 0
                response = None
                while tries < MAX_TRIES:
                    thinking_mode = False
                    run_response = llm_engine([{'role': 'user', 'content': question}])
                    if '<think' in run_response: thinking_mode = True
                    run_response = re.findall(r'\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+', run_response)
                    if not run_response:
                        tries += 1
                        continue
                    response = run_response[-1]
                    logging.info(f'{modelsize}, {question_type}, thinking mode: {thinking_mode}')
                    break
                results = append_results(results, design, criterion, material, response, question_type)
            else:
                for material in materials:
                    if question_type == 'agentic':
                        question = compile_question(design, criterion, material, question_type)
                        tries = 0
                        response = None
                        while tries < MAX_TRIES:
                            thinking_mode = False
                            run_response = websurfer_agent.run(question)
                            if isinstance(run_response, str):
                                if '<think' in run_response: thinking_mode = True
                                run_response = re.findall(r'\d+', run_response)[-1]
                            memory = websurfer_agent.write_inner_memory_from_logs()
                            content = extract_content_from_memory(memory)
                            final_answer_called = False
                            count = 0
                            for item in content[1:]:
                                count += 1
                                if '<think' in item: thinking_mode = True
                                if 'final_answer' in item: final_answer_called = True
                            last_item = content[-1]
                            if (('OUTPUT OF STEP' in last_item and 'Error' in last_item) or (not final_answer_called) 
                                or (response and (response < 0 or response > 10))): 
                                tries += 1
                                continue
                            response = run_response
                            logging.info(f'{modelsize}, {question_type}, {material}, thinking mode: {thinking_mode}')
                            break
                    elif question_type in ['zero-shot', 'few-shot']:
                        tries = 0
                        response = None
                        while tries < MAX_TRIES:
                            thinking_mode = False
                            question = compile_question(design, criterion, material, question_type)
                            run_response = llm_engine([{'role': 'user', 'content': question}])
                            if '<think' in run_response: thinking_mode = True
                            run_response = re.findall(r'\d+', run_response)
                            if not run_response:
                                tries += 1
                                continue
                            run_response = run_response[-1]
                            if int(run_response) < 0 or int(run_response) > 10:
                                tries += 1
                                continue
                            response = run_response
                            logging.info(f'{modelsize}, {question_type}, {material}, thinking mode: {thinking_mode}')
                            break
                    elif question_type == 'chain-of-thought':
                        tries = 0
                        response = None
                        while tries < MAX_TRIES:
                            thinking_mode = False
                            question = compile_question(design, criterion, material, question_type, count=0)
                            reasoning = llm_engine([{'role': 'user', 'content': question}], max_tokens=200)
                            question = compile_question(design, criterion, material, question_type, count=1, reasoning=reasoning)
                            run_response = llm_engine([{'role': 'user', 'content': question}])
                            if '<think' in run_response: thinking_mode = True
                            run_response = re.findall(r'\d+', run_response)
                            if not run_response:
                                tries += 1
                                continue
                            run_response = run_response[-1]
                            if int(run_response) < 0 or int(run_response) > 10:
                                tries += 1
                                continue
                            response = run_response
                            logging.info(f'{modelsize}, {question_type}, {material}, thinking mode: {thinking_mode}')
                            break
                    results = append_results(results, design, criterion, material, response, question_type)