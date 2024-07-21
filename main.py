import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, AgentOutputParser
from tools import extract_triple, extract_relation, search_wiki, analyze_sentence, extract_entity
from langchain.tools import StructuredTool
from langchain.tools.base import ToolException
from langchain.utilities import SerpAPIWrapper
from custom_prompt import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX#  HUMAN_MESSAGE_TEMPLATE
import json
from datetime import datetime
start_time = datetime.now()
print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
openai_api_key = ""
serp_api_key = ""

# 加载 OpenAI 模型
llm = ChatOpenAI(temperature=0, openai_api_base='', openai_api_key=openai_api_key,  model_name="gpt-4") 
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)

import requests
import json

# 使用你的API密钥和自定义搜索引擎ID
api_key = ""
cse_id = ""
def google_search(search_term, api_key, cse_id, **kwargs):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': search_term,
        'key': api_key,
        'cx': cse_id,
    }
    params.update(kwargs)
    response = requests.get(url, params=params)
    return response.json()

def handle_error(error: ToolException) -> str:
    return "An error occurred: " + str(error)

tools = [
    Tool(
        name="ExtractRelation",
        func=extract_relation,
        description="This tool is useful for extracting candidate relations from a sentence. The required input is the sentence itself, along with a list of relation candidates in string format.",
        handle_tool_error=handle_error,
    ),
    Tool.from_function(
        name="ExtractEntity",
        func=extract_entity,
        description="useful for when you need to extract candidate entities contains in the given sentence. The input should be the entity or relation in string type.",
        handle_tool_error=handle_error,
    ),
    Tool.from_function(
        name="SearchInternet",
        #func=search.run,
	func=google_search,
        description="useful for when you need to find more information about an entity or relation from the Internet search engine. The input should be the entity or relation in string type.",
        handle_tool_error=handle_error,
    ),
    Tool.from_function(
        name="SearchWikipedia",
        func=search_wiki,
        description="useful for when you need to find more information about an entity or relation from the Wikipedia. The input should be the entity or relation in string type.",
        handle_tool_error=handle_error,
    ),
    Tool.from_function(
        name="AnalyzeSentence",
        func=analyze_sentence,
        description="useful for when you need to taka a basic analyze about a sentence, conduct analyze to have "
                    "better understanding about the sentence. Input should be the sentence that need to be "
                    "extract triple.",
        handle_tool_error=handle_error,

        )
]

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, agent_kwargs={"prefix": PREFIX,"format_instructions": FORMAT_INSTRUCTIONS,"suffix": SUFFIX})

# 读取test.json文件中的数据
with open('', 'r') as file:
    data = json.load(file)

# agent.run(input="", relation="")
