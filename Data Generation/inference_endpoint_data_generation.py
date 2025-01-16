import os
from transformers import ReactCodeAgent
import regex as re
import pandas as pd
import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_generation_helper import WIKIPEDIA_SEARCH_PROMPT, WikipediaSearch, compile_question, append_results

##########################################################################################
# running 14B and 32B models using huggingface inference endpoint

materials = ['steel', 'aluminum', 'titanium', 'glass', 'wood', 'thermoplastic', 'thermoset', 'elastomer', 'composite']

# creating mapping for model size to inference endpoint URL
INFERENCE_URLS = {
    '14': os.getenv('14B_INFERENCE_URL'),
    '32': os.getenv('32B_INFERENCE_URL')
}

# creating mapping for model size to model
MODEL_IDS = {
    '14': 'tgi',
    '32': None
}

# def test_inference_url():
#     headers = {
#         'Authorization': f'Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}',
#         'Content-Type': 'application/json'
#     }
#     payload = {
#         "messages": [
#             {"role": "user", "content": "Hi!"}
#         ],
#         "max_tokens": 100,
#         "temperature": 0.7,
#         "parameters": {"n_gpu_layers": 28}
#     }
#     response = requests.post(
#             url=os.getenv('14B_INFERENCE_URL'),
#             headers=headers,
#             json=payload
#         )
#     if response.status_code == 200:
#         print("Inference URL is accessible.")
#         print("Response:", response.json())  # Optionally print the response to understand the structure
#     else:
#         print(f"Error accessing the inference URL: {response.status_code} - {response.text}")
# test_inference_url()

# print(torch.cuda.is_available())

for modelsize in [
                '14',
                # '32'
                ]:
    
    inference_url=INFERENCE_URLS[modelsize]
    model_id=MODEL_IDS[modelsize]
    
    # # define a function to call the inference endpoint and return output
    # def llm_engine(messages, max_tokens=1000, stop_sequences=['Task'], temperature=0.6):
    #     client = OpenAI(
    #         base_url=inference_url,
    #         api_key=os.getenv('HUGGINGFACE_API_TOKEN')
    #     )
    #     chat_completion = client.chat.completions.create(
    #         model=model_id,
    #         messages=messages,
    #         max_tokens=max_tokens,
    #         temperature=temperature,
    #         stop=stop_sequences
    #     )
    #     return chat_completion.choices[0].message.content

    # define a function to call the inference endpoint and return output
    def llm_engine(messages, max_tokens=1000, stop_sequences=['Task'], temperature=0.6):
        headers = {
            'Authorization': f'Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model_id,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stop': stop_sequences,
            'device': 'cuda'
        }
        response = requests.post(
            url=inference_url,
            headers=headers,
            json=payload
        )
        answer = response.json()['choices'][0]['message']['content']
        return answer
        
    # create agent with web search tool
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
                            response = websurfer_agent.run(question)
                            if isinstance(response, str): response = int(re.findall(r'\d+', response)[0])
                        elif question_type in ['zero-shot', 'few-shot']:
                            question = compile_question(design, criterion, material, question_type)
                            response = llm_engine([{'role': 'user', 'content': question}])
                            response = re.findall(r'\d+', response)[0]
                        elif question_type == 'parallel':
                            question = compile_question(design, criterion, [', '.join(materials)], question_type)
                            response = llm_engine([{'role': 'user', 'content': question}])
                        elif question_type == 'chain-of-thought':
                            count = 0
                            question = compile_question(design, criterion, material, question_type)
                            reasoning = llm_engine([{'role': 'user', 'content': question}], max_tokens=200)
                            count = 1
                            question = compile_question(design, criterion, material, question_type, reasoning=reasoning)
                            response = llm_engine([{'role': 'user', 'content': question}])
                        results = append_results(results, design, criterion, material, response, question_type)
        results.to_csv(f'Results/Data/qwen_{modelsize}B_{question_type}.csv', index=False)