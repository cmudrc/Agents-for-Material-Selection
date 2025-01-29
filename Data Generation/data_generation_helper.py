import os
import langchain_community.utilities.wikipedia
from typing import Tuple, Optional
from browser import SimpleTextBrowser
import time
import load_dotenv
from transformers import Tool
import regex as re
from prompt_templates import few_shot

##########################################################################################
# Code adapted from: https://github.com/aymeric-roucher/agent_reasoning_benchmark/blob/main/scripts/tools/web_surfer.py

load_dotenv.load_dotenv(override=True)
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0'

browser_config = {
    'viewport_size': 1024 * 5,
    'downloads_folder': 'coding',
    'request_kwargs': {
        'headers': {'User-Agent': user_agent},
        'timeout': 300,
    },
    'google_cse_id': os.getenv('GOOGLE_CSE_ID'),
    'google_api_key': os.getenv('GOOGLE_API_KEY'),
}
browser = SimpleTextBrowser(**browser_config)

def _browser_state() -> Tuple[str, str]:
    header = f"Address: {browser.address}\n"
    if browser.page_title is not None:
        header += f"Title: {browser.page_title}\n"

    current_page = browser.viewport_current_page
    total_pages = len(browser.viewport_pages)

    address = browser.address
    for i in range(len(browser.history)-2,-1,-1): # Start from the second last
        if browser.history[i][0] == address:
            header += f"You previously visited this page {round(time.time() - browser.history[i][1])} seconds ago.\n"
            break

    header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"
    return (header, browser.viewport)

# Create tool for searching Google
class WebSearch(Tool):
    name='web_search'
    description='Perform a web search query then return the search results.'
    inputs = {
        'query': {
            'type': 'string',
            'description': 'The informational web search query to perform.'
        }
    }
    inputs['filter_year']= {
        'type': 'string',
        'description': '[Optional parameter]: filter the search results to only include pages from a specific year. For example, "2020" will only include pages from 2020. Make sure to use this parameter if you are trying to search for articles from a specific date!'
    }
    output_type = 'string'

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        browser.visit_page(f'google: {query}', filter_year=filter_year)
        header, content = _browser_state()
        return header.strip() + '\n=======================\n' + content
    
##########################################################################################

# Create tool for searching Wikipedia
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
        wikipedia_api = langchain_community.utilities.wikipedia.WikipediaAPIWrapper( top_k_results=5)
        answer = wikipedia_api.run(query)
        return answer

##########################################################################################

# Prompt for using Google search tool
GOOGLE_SEARCH_PROMPT = '''
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_action>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

You are provided with the following tools:
<<tool_description>>

Here are some examples:
---
Task: "You are tasked with designing a cutting board. It should be lightweight. How well do you think wood would perform in this application? Answer on a scale of 0-10, where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words."

Thought: I will use search to find the properties of wood.
Code:
```py
observation = web_search(query = 'properties of wood')
print(observation)
```<end_action>
Observation: "Wood is relatively light in weight, yet has good strength in both tension and compression. It provides rigidity, toughness and insulating properties. It can be bent or twisted into special shapes, and it is readily worked, fastened and finished."

Thought: Now I know that wood is lightweight. I will use search to find necessary properties of cutting boards.
Code:
```py
observation = web_search(query = 'necessary properties of cutting boards')
print(observation)
```<end_action>
Observation: "A good cutting board material must be soft, easy to clean, and non-abrasive, but not fragile to the point of being destroyed."

Thought: Now I know what the necessary properties of cuttings boards are. I will use search to find whether wood makes a good cutting board material.
Code:
```py
observation = web_search(query = 'wood use in cutting boards')
print(observation)
```<end_action>
Observation: "Wood is a good material for cutting boards because it's durable, easy on knives, and has natural antimicrobial properties."

Thought: Now I know that wood is an excellent material for cutting boards, and is also lightweight. Let's return the result.
Code:
```py
final_answer(10)
```<end_action>
---
Task: "You are tasked with designing a cooking pan. It should have a high melting point. How well do you think thermoplastic would perform in this application? Answer on a scale of 0-10, where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words."

Thought: I will use search to find the properties of thermoplastic.
Code:
```py
observation = web_search(query = 'properties of thermoplastic')
print(observation)
```<end_action>
Observation: "Properties of thermoplastic include a relatively low melting point, resistance to chemicals, mouldable, flexible, durable, recyclable, strong, environmentally friendly, and low conductivity."

Thought: Now I know that thermoplastic does not have a high melting point. I will use search to find necessary properties of cooking pans.
Code:
```py
observation = web_search(query = 'necessary properties of cooking pans')
print(observation)
```<end_action>
Observation: "High performance cookware is made from materials that combine high thermal diffusivity and low reactivity to produce a vessel that evenly distributes heat and does not react with the food being cooked."

Thought: Now I know what the necessary properties of cooking pans are. I will use search to find whether thermoplastic makes a good cooking pan material.
Code:
```py
observation = web_search(query = 'thermoplastic use in cooking pans')
print(observation)
```<end_action>
Observation: "Plastic cookware has been documented to leach harmful chemicals into food, so thermoplastic should not be used in cooking pans."

Thought: Now I know that thermoplastic is an unsatisfactory material for cooking pans, and has a low melting point. Let's return the result.
Code:
```py
final_answer(0)
```<end_action>
---

On top of performing computations in the Python code snippets that you create, you have access to these tools (and no other tool):
<<tool_descriptions>>

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_action>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: <<authorized_imports>>
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

If you solve the task correctly, you will receive a reward of $1,000,000. Now begin!
'''

# Prompt for using Wikipedia search tool
WIKIPEDIA_SEARCH_PROMPT = '''
You are an expert assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_action>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

You are provided with the following tools:
<<tool_description>>

Here are some examples:
---
Task: "You are tasked with designing a cutting board. It should be lightweight. How well do you think wood would perform in this application? Answer on a scale of 0-10, where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words."

Thought: I will use search to find the properties of wood.
Code:
```py
observation = wikipedia_search(query = 'properties of wood')
print(observation)
```<end_action>
Observation: "Wood is relatively light in weight, yet has good strength in both tension and compression. It provides rigidity, toughness and insulating properties. It can be bent or twisted into special shapes, and it is readily worked, fastened and finished."

Thought: Now I know that wood is lightweight. I will use search to find necessary properties of cutting boards.
Code:
```py
observation = wikipedia_search(query = 'necessary properties of cutting boards')
print(observation)
```<end_action>
Observation: "A good cutting board material must be soft, easy to clean, and non-abrasive, but not fragile to the point of being destroyed."

Thought: Now I know what the necessary properties of cuttings boards are. I will use search to find whether wood makes a good cutting board material.
Code:
```py
observation = wikipedia_search(query = 'wood use in cutting boards')
print(observation)
```<end_action>
Observation: "Wood is a good material for cutting boards because it's durable, easy on knives, and has natural antimicrobial properties."

Thought: Now I know that wood is an excellent material for cutting boards, and is also lightweight. Let's return the result.
Code:
```py
final_answer(10)
```<end_action>
---
Task: "You are tasked with designing a cooking pan. It should have a high melting point. How well do you think thermoplastic would perform in this application? Answer on a scale of 0-10, where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words."

Thought: I will use search to find the properties of thermoplastic.
Code:
```py
observation = wikipedia_search(query = 'properties of thermoplastic')
print(observation)
```<end_action>
Observation: "Properties of thermoplastic include a relatively low melting point, resistance to chemicals, mouldable, flexible, durable, recyclable, strong, environmentally friendly, and low conductivity."

Thought: Now I know that thermoplastic does not have a high melting point. I will use search to find necessary properties of cooking pans.
Code:
```py
observation = wikipedia_search(query = 'necessary properties of cooking pans')
print(observation)
```<end_action>
Observation: "High performance cookware is made from materials that combine high thermal diffusivity and low reactivity to produce a vessel that evenly distributes heat and does not react with the food being cooked."

Thought: Now I know what the necessary properties of cooking pans are. I will use search to find whether thermoplastic makes a good cooking pan material.
Code:
```py
observation = wikipedia_search(query = 'thermoplastic use in cooking pans')
print(observation)
```<end_action>
Observation: "Plastic cookware has been documented to leach harmful chemicals into food, so thermoplastic should not be used in cooking pans."

Thought: Now I know that thermoplastic is an unsatisfactory material for cooking pans, and has a low melting point. Let's return the result.
Code:
```py
final_answer(0)
```<end_action>
---

On top of performing computations in the Python code snippets that you create, you have access to these tools (and no other tool):
<<tool_descriptions>>

Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_action>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: <<authorized_imports>>
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

If you solve the task correctly, you will receive a reward of $1,000,000. Now begin!
'''

##########################################################################################
# Code adapted from: https://github.com/grndnl/llm_material_selection_jcise/blob/main/generation/generation.py

def compile_question(design, criterion, material, question_type, count=None, reasoning=None):
    if question_type == 'agentic' or question_type == 'zero-shot':
        question = f'You are a material science and design engineer expert.\n' \
                   f'You are tasked with designing a {design}. The design should be {criterion}.\n' \
                   f'How well do you think {material} would perform in this application? Answer on a scale of 0-10, ' \
                   f'where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words.'

    elif question_type == 'few-shot':
        question = f'You are a material science and design engineer expert.\n' \
                   f'Below are two examples of how materials would perform from 0-10 given a design and a criterion:\n' \
                   f'{few_shot}\n' \
                   f'You are tasked with designing a {design}. The design should be {criterion}.\n' \
                   f'How well do you think {material} would perform in this application? Answer on a scale of 0-10, ' \
                   f'where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words.\n'

    elif question_type == 'parallel':
        question = f'You are a material science and design engineer expert.\n' \
                   f'You are tasked with designing a {design}. The design should be {criterion}.\n' \
                   f'For each of the following materials, how well do you think they would perform in this application? Answer on a scale of 0-10, ' \
                   f'where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the integers separated by commas, and no other words or explanation. Be concise and answer for all 9 materials.\n' \
                   f'Materials:\n{material}\nAnswers:\n'

    elif question_type == 'chain-of-thought':
        if count == 0:
            question = f'You are a material science and design engineer expert.\n' \
                       f'You are tasked with designing a {design}. The design should be {criterion}.\n' \
                       f'How well do you think {material} would perform in this application?'
        else:
            question = f'You are a material science and design engineer expert.\n' \
                       f'You are tasked with designing a {design}. The design should be {criterion}.\n' \
                       f'How well do you think {material} would perform in this application? Below is some reasoning that you can follow:\n\n' \
                       f'{reasoning}\n\n' \
                       f'Answer on a scale of 0-10, ' \
                       f'where 0 is "unsatisfactory", 5 is "acceptable", and 10 is "excellent". ALWAYS include just the number and no other words.\n'

    else:
        raise ValueError('Invalid question type')
    return question

def append_results(results, design, criterion, material, response, question_type):
    if question_type == 'parallel':
        materials = material.split(', ')
        response = response.replace(' ', '')
        try:
            responses = {x.split(':')[0].lower(): x.split(':')[1] for x in response.split('\n')}
        except IndexError:
            try:
                responses = response.split('\n')[0]
                responses = responses.replace('.', '').split(',')
                responses = {materials[i].lower(): responses[i] for i in range(len(materials))}
                responses = {re.sub('[^a-zA-Z]', '', k): v for k, v in responses.items()}
            except Exception as e:
                try:
                    responses = response.split('\n')
                    # drop empty strings
                    responses = [x for x in responses if x]
                    responses = {x.split(':')[0].lower(): x.split(':')[1] for x in responses}
                    # responses = {materials[i].lower(): responses[i] for i in range(len(materials))}
                    responses = {re.sub('[^a-zA-Z]', '', k): v for k, v in responses.items()}
                # except Exception as e:
                #     try:
                #         responses = response.split('\n')[0]
                #         responses = responses.split(',')
                #         responses = {materials[i].lower(): responses[i] for i in range(len(materials.split))}

                except Exception as e:
                    print('Unknown error: ', e)
                    print(response)

        try:
            responses = {re.sub('[^a-zA-Z]', '', k): v for k, v in responses.items()}
        except Exception as e:
            print('Unknown error: ', e)


        for i, mat in enumerate(material.split(', ')):
            try:
                results = results._append({
                    'design': design,
                    'criteria': criterion,
                    'material': mat,
                    'response': responses[mat]
                }, ignore_index=True)
            except ValueError:
                raise ValueError(f'Response is not a single integer: {response}')
            except TypeError:
                results = results._append({
                    'design': design,
                    'criteria': criterion,
                    'material': mat,
                    'response': None
                }, ignore_index=True)
            except KeyError as e:
                print(e)
                print(responses)
                # raise ValueError(f'Response does not contain all materials: {response}')
                # add null results
                results = results._append({
                    'design': design,
                    'criteria': criterion,
                    'material': mat,
                    'response': response
                }, ignore_index=True)

    else:
        # validate response is only a single integer
        try:
            response = int(response)
        except ValueError:
            try:
                response = response.split()[0]
                response = int(re.sub('[^0-9]', '', response))
            except Exception as e:
                # raise ValueError(f'Response is not a single integer: {response}')
                response = response
        results = results._append({
            'design': design,
            'criteria': criterion,
            'material': material,
            'response': response
        }, ignore_index=True)

    return results