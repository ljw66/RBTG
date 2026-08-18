from pydantic import BaseModel, Field
from typing import List
import json


# ==================== Pydantic 模型定义 ====================

class FunItem(BaseModel):
    """功能项（处理项）"""
    fun_id: str = Field(
        description="需求标识，格式：TR-GN-{前缀}-{序号}。TR和GN为固定标识，第三部分根据中文标题切词后缩写拼接，示例：TR-GN-HubPrTm-001"
    )
    handling: str = Field(
        description="需求描述，处理逻辑要完整，不能有缺项。示例：软件通过包路由功能获取到遥测包，通过APID判断当前收到的包类型，将遥测帧组织或完成其它功能需要的参考遥测保存至数据池。"
    )


class TestRequirementOutput(BaseModel):
    """测试需求项输出结构"""
    fun_req_summary: str = Field(
        description="对该小节的功能概述。如果内容中有功能描述内容，可以直接取出。示例：通过包路由功能获得从总线终端或其它进程传来的遥测包中获取遥测数据，并存入整星数据池。"
    )
    
    input_stream_desc: List[str] = Field(
        description="输入流说明列表，内容要简练概括，但不能缺少必要信息。没有输入流时填写['无']。示例：['上行注入中断状态寄存器。', '信道关口遥测：以10s为周期固定更新遥测。']"
    )
    
    fun_items: List[FunItem] = Field(
        description="处理项列表，处理逻辑要完整，不能有缺项。处理下面的每一子项单独生成一个测试需求项。"
    )
    
    output_stream_desc: List[str] = Field(
        description="输出流说明列表，如果没有的话请根据文档内容总结，没有输出流时填写['无']。示例：['SMU信道关口模块数字量遥测包（APID=0x404）。', 'SMU指令译码模块数字量遥测包（APID=0x405）。']"
    )


def _generate_example_json() -> str:
    """
    从 Pydantic 模型生成示例 JSON 字符串
    
    作用：
    1. 创建一个符合 TestRequirementOutput 模型的示例对象
    2. 将其转换为 JSON 字符串
    3. 作为 extract_content 的值，用于提示词中告诉模型期望的输出格式
    
    好处：
    - 自动保证示例结构与模型定义一致
    - 如果修改了模型，示例会自动更新
    - 避免手动维护 JSON 字符串导致的结构错误
    
    返回示例：
    {
        "fun_req_summary": "...",
        "input_stream_desc": [...],
        "fun_items": [...],
        "output_stream_desc": [...]
    }
    """
    # 创建一个示例对象（使用 Pydantic 模型，自动验证结构）
    example = TestRequirementOutput(
        fun_req_summary="通过包路由功能获得从总线终端或其它进程传来的遥测包中获取遥测数据，并存入整星数据池。",
        input_stream_desc=[
            "上行注入中断状态寄存器。",
            "信道关口遥测：以10s为周期固定更新遥测。"
        ],
        fun_items=[
            FunItem(
                fun_id="TR-GN-HubPrTm-001",
                handling="软件通过包路由功能获取到遥测包，通过APID判断当前收到的包类型，将遥测帧组织或完成其它功能需要的参考遥测保存至数据池。"
            ),
            FunItem(
                fun_id="TR-GN-HubPrTm-002",
                handling="处理逻辑描述示例。"
            )
        ],
        output_stream_desc=[
            "SMU信道关口模块数字量遥测包（APID=0x404）。",
            "SMU指令译码模块数字量遥测包（APID=0x405）。"
        ]
    )
    
    # 将 Pydantic 对象转换为字典，再格式化为 JSON 字符串
    example_dict = example.model_dump()
    return json.dumps(example_dict, ensure_ascii=False, indent=4)


# ==================== 提示词模板 ====================

req_prompts = {
    "system_prompt": "你是一位专业的嵌入式软件测试工程师，负责将输入内容按照以下规则转换为测试需求项。",
    
    "user_prompt": """#角色
你是一位专业的嵌入式软件测试工程师，负责将输入内容按照以下规则转换为测试需求项。你的任务是将功能需求拆解为输入流-处理-输出流，从处理里获取测试需求项，如果没有处理请根据文档内容总结，每一个小条目生成一个测试需求项（TR项），可以根据语义分成多个测试需求项。从输入中获取输入流，从输出中获取输出流。

在填写内容时，需要补足细节信息，比如文档中出现'均在规定的存储空间内'内容需要找到具体的存储空间并填入（规定的存储空间地址范围：0x20000E00~0x20003E00），如果未找到则不填，不可以自行编造。

请注意以下文档中，相似度得分为1的文档是你要使用的```主要文档```，相似度得分<1的文档为检索得到的```参考文档```，生成内容时请根据```主要文档```来生成，出现上下文内容不足时才使用参考文档进行补充。

文档：{doc_content}

# 输出格式要求
请严格按照以下 JSON 格式输出，确保所有字段都存在且类型正确：
{extract_content}

请根据上述要求，提取并生成测试需求项。""",
    
    # 从 Pydantic 模型自动生成示例，确保结构一致性
    "extract_content": _generate_example_json(),
    
    "doc_content": """功能需求"""
}

if __name__ == "__main__":
    print(req_prompts["extract_content"])