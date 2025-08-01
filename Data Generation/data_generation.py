# Force cuda usage
import os
os.environ['FORCE_CUDA'] = '1'

##########################################################################################

import llama_cpp
from transformers import ReactCodeAgent
import regex as re
import pandas as pd
import logging
from transformers import Tool
import langchain_community.utilities.wikipedia
from data_generation_helper import WIKIPEDIA_SEARCH_PROMPT, compile_question, append_results

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
        searches.append(query)
        wikipedia_api = langchain_community.utilities.wikipedia.WikipediaAPIWrapper(top_k_results=5)
        answer = wikipedia_api.run(query)
        return answer

##########################################################################################
# Code adapted from: https://github.com/grndnl/llm_material_selection_jcise/blob/main/generation/generation.py

materials = ['steel', 'aluminum', 'titanium', 'glass', 'wood', 'thermoplastic', 'thermoset', 'elastomer', 'composite']

# Configure logging
log_dir = 'agent_logs'
os.makedirs(log_dir, exist_ok=True)

def setup_logger(modelsize):
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f'{log_dir}/qwen3_{modelsize}B_searches.log'), logging.StreamHandler()]
    )

def extract_content_from_memory(memory):
    return [entry['content'] for entry in memory if 'content' in entry]

searches = []
total_tokens = []
completion_tokens = []
MAX_TRIES = 5

for modelsize in ['1.5', '3', '7', '14', '32', '72']:
    setup_logger(modelsize)
    logger = logging.getLogger()
    
    # Create LLM object
    llm = llama_cpp.Llama.from_pretrained(
        repo_id=f'bartowski/Qwen2.5-{modelsize}B-Instruct-GGUF',
        filename=f'Qwen2.5-{modelsize}B-Instruct-Q4_K_M.gguf',
        n_ctx=2048,
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
        completion_tokens.append(response["usage"]["completion_tokens"])
        total_tokens.append(response["usage"]["total_tokens"])
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
    
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        results = pd.DataFrame(columns=['design', 'criteria', 'material', 'response'])
        for design in ['kitchen utensil grip', 'safety helmet', 'underwater component', 'spacecraft component']:
            for criterion in ['lightweight', 'heat resistant', 'corrosion resistant', 'high strength']:
                if question_type == 'parallel':
                    material = ', '.join(materials)
                    question = compile_question(design, criterion,  material, question_type)
                    response = llm_engine([{'role': 'user', 'content': question}])
                    results = append_results(results, design, criterion, material, response, question_type)
                else:
                    for material in materials:
                        if question_type == 'agentic':
                            question = compile_question(design, criterion, material, question_type)
                            tries = 0
                            response = None
                            total_tokens = []
                            completion_tokens = []
                            while tries < MAX_TRIES:
                                run_response = websurfer_agent.run(question)
                                if isinstance(run_response, str): run_response = re.findall(r'\d+', run_response)[0]
                                memory = websurfer_agent.write_inner_memory_from_logs()
                                content = extract_content_from_memory(memory)
                                final_answer_called = False
                                count = 0
                                for item in content[1:]:
                                    count += 1
                                    if 'final_answer' in item: final_answer_called = True
                                last_item = content[-1]
                                if (('OUTPUT OF STEP' in last_item and 'Error' in last_item) or (not final_answer_called) 
                                    or (response and (response < 0 or response > 10))): 
                                    tries += 1
                                    logging.info(f"Retrying run for {design}, {criterion}, {material}: {tries}/{MAX_TRIES}")
                                    searches = []
                                    total_tokens = []
                                    completion_tokens = []
                                else:
                                    logging.info(f"Keeping result for {design}, {criterion}, {material}")
                                    logging.info(f"Searches: {', '.join(searches)}")
                                    logging.info(f"Total tokens: {', '.join([str(tokens) for tokens in total_tokens])}")
                                    logging.info(f"Completion tokens: {', '.join([str(tokens) for tokens in completion_tokens])}")
                                    response = run_response
                                    searches = []
                                    total_tokens = []
                                    completion_tokens = []
                                    break
                        elif question_type in ['zero-shot', 'few-shot']:
                            question = compile_question(design, criterion, material, question_type)
                            response = llm_engine([{'role': 'user', 'content': question}])
                            response = re.findall(r'\d+', response)[0]
                        elif question_type == 'parallel':
                            question = compile_question(design, criterion, [', '.join(materials)], question_type)
                            response = llm_engine([{'role': 'user', 'content': question}])
                        elif question_type == 'chain-of-thought':
                            count = 0
                            question = compile_question(design, criterion, material, question_type, count=count, )
                            reasoning = llm_engine([{'role': 'user', 'content': question}], max_tokens=200)
                            count = 1
                            question = compile_question(design, criterion, material, question_type, count=count, reasoning=reasoning)
                            response = llm_engine([{'role': 'user', 'content': question}])
                        results = append_results(results, design, criterion, material, response, question_type)
        results.to_csv(f'Results/Data/qwen_{modelsize}B_{question_type}.csv', index=False)
