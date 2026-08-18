from pydantic import BaseModel, Field
from typing import List
import json


# ==================== Pydantic 模型定义 ====================

class PerfReqItem(BaseModel):
    """单个性能需求项"""
    perf_req_desc: str = Field(
        description="性能需求描述，示例：下行码速率：16384bps（测控下行），8192bps（中继）"
    )
    perf_req_id: str = Field(
        description="测试需求标识，示例：TR-XN-001，TR和XN固定不变，序号从1开始递增"
    )
    other_desc: str = Field(
        description="相关说明，根据性能需求描述提取，示例：速率要求"
    )


class PerfReqItemsOutput(BaseModel):
    """性能需求项输出结构（用于 structured output）"""
    perf_req_items: List[PerfReqItem] = Field(
        description="性能需求项列表，每一项包含性能需求描述、测试需求标识和相关说明"
    )



class SoftwareNameOutput(BaseModel):
    """软件名称等信息输出结构"""
    software_name: str = Field(description="软件名称")
    software_id: str = Field(description="软件标识号")
    model_name: str = Field(description="型号名称")
    doc_version: str = Field(description="文档版本号")
    entrusting_addr: str = Field(description="委托单位地址")
    model_subsystem_name: str = Field(description="型号分系统名称")
    code_version: str = Field(description="代码版本号，默认值2.04")
    req_spec_file_id: str = Field(description="需求规格文件标识号，默认值为WY-R/HYS-4A/RX016")


class InterruptItem(BaseModel):
    """单个中断使用情况项"""
    name: str = Field(description="中断名称，示例：外部中断1（0x15）")
    priority: str = Field(description="中断优先级，示例：25ms")
    cycles: str = Field(description="周期(触发频率) / 随机 (频发/偶发)")
    trigger: str = Field(description="触发方式，示例：低电平")
    function: str = Field(description="执行功能，示例：BIU中断，存储复接的数据接收缓存")


class UseOfInterruptOutput(BaseModel):
    """中断使用情况输出结构"""
    interrupt_list: List[InterruptItem] = Field(description="中断使用情况列表")


class SubsystemRelationOutput(BaseModel):
    """系统概述输出结构"""
    subsystem_relation: str = Field(description="系统概述，取自系统组成的内容")


class CpuStorageOutput(BaseModel):
    """CPU和存储器信息输出结构"""
    cpu_desc: str = Field(description="CPU类型、主频，示例：处理器采用BM3823芯片；主频：200MHz；")
    use_of_storage_io: str = Field(description="存储器，按示例格式输出，存储器的描述不要简写")


class SoftwareLevelOutput(BaseModel):
    """软件级别等信息输出结构"""
    software_level: str = Field(description="软件级别 (A, B, C)")
    programming_language: str = Field(description="软件编写语言，示例：C语言")
    dev_unit: str = Field(description="研制单位")
    subsystem_function: str = Field(description="软件的主要功能，每项功能前需要按照示例说明生成一个顺序的编号")
    subsystem_function_qd: str = Field(description="软件强度测试功能")


class DevPlatformOutput(BaseModel):
    """开发平台信息输出结构"""
    dev_platform: str = Field(description="软件开发平台、编译环境描述，按照示例输出主要的编译信息")


class GpioInterfaceItem(BaseModel):
    """单个GPIO接口项"""
    interface_desc: str = Field(description="接口描述，对接口功能的描述")
    table_data: str = Field(description="接口表格")


class GpioInterfaceOutput(BaseModel):
    """GPIO接口信息输出结构"""
    gpio_interfaces: List[GpioInterfaceItem] = Field(description="GPIO接口列表")


class DogInterfaceItem(BaseModel):
    """单个看门狗接口项"""
    interface_desc: str = Field(description="接口描述，对接口功能的描述")
    table_data: str = Field(description="接口表格")


class DogInterfaceOutput(BaseModel):
    """看门狗接口信息输出结构"""
    dog_interfaces: List[DogInterfaceItem] = Field(description="看门狗接口列表")


class OtherHardwareInterfaceItem(BaseModel):
    """单个其他硬件接口项"""
    interface_name: str = Field(description="接口名称，提取硬件接口子章节名即可，示例：启动状态数据读写接口")
    interface_type: str = Field(description="接口类型，如FPGA、FLASH等")
    interface_desc: str = Field(description="接口描述，对接口功能的描述")
    table_data: str = Field(description="接口表格")


class OtherHardwareInterfaceOutput(BaseModel):
    """其他硬件接口信息输出结构"""
    other_hardware_interfaces: List[OtherHardwareInterfaceItem] = Field(description="其他硬件接口列表")


class InterfaceReqItem(BaseModel):
    """单个接口需求项"""
    interface_req_desc: str = Field(description="接口需求描述，示例：软件通过1553B接口与各RT进行重要数据、遥控、遥测数据的采集与分发")
    interface_req_id: str = Field(description="测试需求标识，填写-")
    other_desc: str = Field(description="相关说明，根据接口描述从'接口信息格式及内容的测试'和'接口正常/异常测试'中选取一个填写")
    interface_name: str = Field(description="对应接口名称，示例：1553B接口")
    interface_id: str = Field(description="接口标识，示例：IF-001，从001开始递增")


class InterfaceReqItemsOutput(BaseModel):
    """接口需求项输出结构"""
    interface_req_summary: List[InterfaceReqItem] = Field(description="接口需求项列表")
    interface_req_count: str = Field(description="接口需求项数，接口需求中所有条目的总和")


class ReliableReqItem(BaseModel):
    """单个可靠性安全性需求项"""
    reliable_req_desc: str = Field(description="可靠性安全性需求描述")
    reliable_req_id: str = Field(description="对应测试需求标识，示例：TR-AQ-001，TR和AQ固定不变，序号从1开始递增")
    other_desc: str = Field(description="相关说明，根据描述内容从'提高安全性的容错方案'和'异常条件下软件的处理和保护能力'中选择一个填写")


class ReliableReqItemsOutput(BaseModel):
    """可靠性安全性需求项输出结构"""
    reliable_req_summary: List[ReliableReqItem] = Field(description="可靠性安全性需求项列表")
    reliable_req_count: str = Field(description="可靠性安全性需求项数，可靠性安全性需求中所有条目的总和")


class MarginReqItem(BaseModel):
    """单个余量需求项"""
    margin_req_desc: str = Field(description="余量需求描述，每一项余量需求描述中隐含余量大于20%的要求，示例：SRAM存储空间使用不超过4M，余量大于20%")
    margin_req_id: str = Field(description="对应测试需求标识，示例：TR-YL-001，TR和YL固定不变，序号从1开始递增")
    other_desc: str = Field(description="相关说明，从余量需求描述中概括，示例1：存储余量；示例2：时间余量")


class MarginReqItemsOutput(BaseModel):
    """余量需求项输出结构"""
    margin_req_summary: List[MarginReqItem] = Field(description="余量需求项列表")
    margin_req_count: str = Field(description="余量需求项数，余量需求中所有条目的总和")


class BoundaryReqItem(BaseModel):
    """单个边界需求项"""
    boundary_req_desc: str = Field(description="边界需求描述，示例：若收到指令ID为0x06的内存下卸指令包(TC_MEM_DUMP)，下卸长度限幅96字节")
    boundary_req_id: str = Field(description="对应测试需求标识，示例：TR-BJ-001，TR和BJ固定不变，序号从1开始递增")
    other_desc: str = Field(description="相关说明，从边界需求描述中概括，示例1：数据输出域边界测试；示例2：数据输入域边界测试")


class BoundaryReqItemsOutput(BaseModel):
    """边界需求项输出结构"""
    boundary_req_items: List[BoundaryReqItem] = Field(description="边界需求项列表")


class SafetyCriticalFuncItem(BaseModel):
    """单个安全关键功能项"""
    func_desc: str = Field(description="安全关键功能描述，示例：入轨段程控功能")
    func_id: str = Field(description="需求标识，填写-")
    other_desc: str = Field(description="相关说明，示例：直接导致危险发生的功能")


class SafetyCriticalReqItemsOutput(BaseModel):
    """安全关键功能需求项输出结构"""
    all_funcs: List[SafetyCriticalFuncItem] = Field(description="安全关键功能列表")
    safety_critical_req_count: str = Field(description="安全关键功能需求项数，安全关键功能需求中所有条目的总和")


class RecoverReqItem(BaseModel):
    """单个恢复性需求项"""
    recover_req_desc: str = Field(description="恢复性需求描述，示例：系统异常复位后，启动软件重新运行，进行程序映像的加载，为正常功能，对业务执行无影响")
    recover_req_id: str = Field(description="对应测试需求标识，示例：TR-HF-001，TR和HF固定不变，序号从1开始递增")
    other_desc: str = Field(description="相关说明，固定填写：故障发生时保护系统状态")


class RecoverReqItemsOutput(BaseModel):
    """恢复性需求项输出结构"""
    recover_req_summary: List[RecoverReqItem] = Field(description="恢复性需求项列表")


# ==================== 模型映射（用于 structured output）====================

def get_output_model(prompt_key: str):
    """
    根据 prompt key 获取对应的 Pydantic 模型类
    
    :param prompt_key: prompt 类型
    :return: Pydantic 模型类，如果未找到则返回 None
    """
    model_mapping = {
        "perf_req_items": PerfReqItemsOutput,
        "software_name": SoftwareNameOutput,
        "use_of_interrupt": UseOfInterruptOutput,
        "subsystem_relation": SubsystemRelationOutput,
        "cpu_storage": CpuStorageOutput,
        "software_level": SoftwareLevelOutput,
        "dev_platform": DevPlatformOutput,
        "gpio_interface": GpioInterfaceOutput,
        "dog_interface": DogInterfaceOutput,
        "other_hardware_interface": OtherHardwareInterfaceOutput,
        "interface_req_items": InterfaceReqItemsOutput,
        "reliable_req_items": ReliableReqItemsOutput,
        "margin_req_items": MarginReqItemsOutput,
        "boundary_req_items": BoundaryReqItemsOutput,
        "safety_critical_req_items": SafetyCriticalReqItemsOutput,
        "recover_req_items": RecoverReqItemsOutput,
    }
    return model_mapping.get(prompt_key)


# ==================== 独立 Prompt 配置（使用 Structured Output）====================

# 性能需求项 prompt（使用 Pydantic 模型进行 structured output）
perf_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取性能需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户输入的性能需求，从描述中提取性能需求描述，按照示例顺序生成测试需求标识，按照示例提取简要的相关说明，生成性能需求，不能随意扩写，不能随意增加项数。

请注意，看门狗中可能包含隐含的性能需求，比如这段看门狗的描述```FPGA前50s不会对CPU狗咬（启动软件不喂狗），直至50s时间到或CPU接管喂狗```,意味着软件启动时间小于50s。

文档：{doc_content}

请根据上述要求，提取并生成性能需求项。每个性能需求项应包含：
- perf_req_desc: 性能需求描述（示例：下行码速率：16384bps（测控下行），8192bps（中继））
- perf_req_id: 测试需求标识（示例：TR-XN-001，TR和XN固定不变，序号从1开始递增）
- other_desc: 相关说明（根据性能需求描述提取，示例：速率要求）""",
    "doc_content": """
     性能需求, 时间特性需求, 看门狗
    """,
    "output_model": "PerfReqItemsOutput"
}

# 软件名称等信息 prompt
software_name = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取软件基本信息。",
    "user_prompt": """#角色
你是一个嵌入式软件测试工程师，请按要求生成软件名称、软件标识、型号名称、文档标识，文档版本号、委托单位地址、型号分系统名称和代码版本号。

#技能
1、确保每个测试需求项的描述清晰、完整。

文档：{doc_content}

请根据上述要求，提取并生成以下信息：
- software_name: 软件名称
- software_id: 软件标识号
- model_name: 型号名称
- doc_version: 文档版本号
- entrusting_addr: 委托单位地址
- model_subsystem_name: 型号分系统名称
- code_version: 代码版本号，默认值2.04
- req_spec_file_id: 需求规格文件标识号，默认值为WY-R/HYS-4A/RX016""",
    "doc_content": """
    配置项基本信息
    """,
    "output_model": "SoftwareNameOutput"
}

# 中断使用情况 prompt
use_of_interrupt = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取中断使用情况。",
    "user_prompt": """请从中断设置中提取中断信息，从中断名称和中断向量中提取中断名称，从优先级中提取中断优先级，从触发周期中提取周期(触发频率) / 随机 (频发/偶发)，中断触发方式默认低电平，从中断对应功能名称和中断中使用的竞争资源中提取执行功能。

文档：{doc_content}

请根据上述要求，提取中断使用情况。每个中断项应包含：
- name: 中断名称，示例：外部中断1（0x15）
- priority: 中断优先级，示例：25ms
- cycles: 周期(触发频率) / 随机 (频发/偶发)
- trigger: 触发方式，示例：低电平
- function: 执行功能，示例：BIU中断，存储复接的数据接收缓存""",
    "doc_content": """
    中断设置
    """,
    "output_model": "UseOfInterruptOutput"
}

# 系统概述 prompt
subsystem_relation = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取系统概述。",
    "user_prompt": """你是一位嵌入式软件测试工程师，请从系统组成章节内容中概括出系统概述，这一节很重要不要缺失内容。

文档：{doc_content}

请将输入内容总结成系统概述。系统概述应取自系统组成的内容，示例：数管分系统的配置：由2台计算机构成，分别为系统管理单元（SMU）和数据接口单元（DIU）。数管分系统以SMU为核心，以分级分布式网络体系结构为系统架构，完成在轨运行调度和综合信息处理，对星上各个任务运行进行高效可靠的管理和控制，监视整星状态，协调整星的工作，对有效载荷进行管理和数据处理，实现整星内信息统一处理和共享的一体化电子系统，系统间采用标准接口，可兼顾现有需求和未来功能的扩展能力。DIU面向载荷设备进行管理，完成载荷数据采集和存储、载荷遥测和温度采集、载荷指令管理、载荷信息实时处理等功能。分系统结构及接口如下图所示：""",
    "doc_content": """
    系统组成, 首页, 目录
    """,
    "output_model": "SubsystemRelationOutput"
}

# CPU和存储器信息 prompt
cpu_storage = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取CPU和存储器信息。",
    "user_prompt": """你是一位嵌入式软件测试工程师，请根据示例提取软件CPU、主频和存储器。

文档：{doc_content}

请根据上述要求，提取以下信息：
- cpu_desc: CPU类型、主频，示例：处理器采用BM3823芯片；主频：200MHz；
- use_of_storage_io: 存储器，按示例格式输出，存储器的描述不要简写。示例：存储器分配如下：
\tFLASH容量：1Mbytes，地址范围是0x0800 0000-0x080F FFFF；
\tSDRAM区容量：256Mbytes，地址范围是0x3000 0000-0x3FFF FFFF。""",
    "doc_content": """
    宿主计算机
    """,
    "output_model": "CpuStorageOutput"
}

# 软件级别等信息 prompt
software_level = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取软件级别和功能信息。",
    "user_prompt": """你是一位嵌入式软件测试工程师，请根据要求从目录的功能需求下面获取软件的主要功能和软件强度测试功能，对应功能需求下的三级标题，需要增加软件隐含第一个功能项初始化功能。请从配置项基本信息章节获取软件安全等级和研制单位，研制单位即为软件交办方，请从设计约束章节获取软件使用的编程语言。

文档：{doc_content}

请根据上述要求，提取以下信息：
- software_level: 软件级别 (A, B, C)
- programming_language: 软件编写语言，示例：C语言
- dev_unit: 研制单位
- subsystem_function: 软件的主要功能，每项功能前需要按照示例说明生成一个顺序的编号。示例：
\t a.初始化功能；
\t b.遥测功能；
\t c.遥控接收功能。
- subsystem_function_qd: 软件强度测试功能，示例：遥测功能、遥控接收功能、指令处理功能、空间包路由功能、链路协议管理功能、时间管理功能、内务管理功能""",
    "doc_content": """
    目录, 配置项基本信息, 设计约束, 首页
    """,
    "output_model": "SoftwareLevelOutput"
}

# 开发平台信息 prompt
dev_platform = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取开发平台信息。",
    "user_prompt": """你是一位嵌入式软件测试工程师，请根据要求从编译要求中获取开发环境概述，仿照示例概括性输出主要内容。

文档：{doc_content}

请根据上述要求，提取开发平台信息：
- dev_platform: 软件开发平台、编译环境描述，按照示例输出主要的编译信息，不用逐句全部输出。示例：SMU应用软件开发环境是建立在Windows7操作系统下的GNU工具包，建立了一个交叉编译环境，工具包版本为SPE-C2.5，包含下列部分：
\t a.sparc-elf-gcc.exe  sparc  交叉编译器；
\t b.grmon.exe  3823调试器""",
    "doc_content": """
    编译要求
    """,
    "output_model": "DevPlatformOutput"
}

# GPIO接口信息 prompt
gpio_interface = {
    "system_prompt": "你是一位软件测试工程师，负责从文档中提取GPIO接口信息。",
    "user_prompt": """你是一位软件测试工程师，请根据要求提取GPIO接口信息。

文档：{doc_content}

请根据上述要求，提取GPIO接口信息。每个接口项应包含：
- interface_desc: 接口描述，对接口功能的描述
- table_data: 接口表格""",
    "doc_content": """
    GPIO, 接口
    """,
    "output_model": "GpioInterfaceOutput"
}

# 看门狗接口信息 prompt
dog_interface = {
    "system_prompt": "你是一位软件测试工程师，负责从文档中提取看门狗接口信息。",
    "user_prompt": """你是一位软件测试工程师，请根据要求提取看门狗接口信息。

文档：{doc_content}

请根据上述要求，提取看门狗接口信息。每个接口项应包含：
- interface_desc: 接口描述，对接口功能的描述
- table_data: 接口表格""",
    "doc_content": """
    看门狗， 接口
    """,
    "output_model": "DogInterfaceOutput"
}

# 其他硬件接口信息 prompt
other_hardware_interface = {
    "system_prompt": "你是一位软件测试工程师，负责从文档中提取硬件接口信息。",
    "user_prompt": """你是一位软件测试工程师，请根据要求提取硬件接口信息，不包括gpio接口和看门狗接口。

文档：{doc_content}

请根据上述要求，提取硬件接口信息。每个接口项应包含：
- interface_name: 接口名称，提取硬件接口子章节名即可，示例：启动状态数据读写接口
- interface_type: 接口类型，如FPGA、FLASH等
- interface_desc: 接口描述，对接口功能的描述
- table_data: 接口表格""",
    "doc_content": """
    硬件接口
    """,
    "output_model": "OtherHardwareInterfaceOutput"
}

# 接口需求项 prompt
interface_req_items = {
    "system_prompt": "你是一位嵌入式软件测试工程师，负责从文档中提取接口需求项。",
    "user_prompt": """你是一位嵌入式软件测试工程师，请根据要求仿照示例从硬件接口和软件接口中获取接口需求描述，不要随意扩写，每一小节生成一条接口需求描述，需要根据接口内容按照示例归纳出相关说明和对应接口名称。

文档：{doc_content}

请根据上述要求，提取并生成接口需求项。每个接口需求项应包含：
- interface_req_desc: 接口需求描述，示例：软件通过1553B接口与各RT进行重要数据、遥控、遥测数据的采集与分发
- interface_req_id: 测试需求标识，填写-
- other_desc: 相关说明，根据接口描述从"接口信息格式及内容的测试"和"接口正常/异常测试"中选取一个填写
- interface_name: 对应接口名称，示例：1553B接口
- interface_id: 接口标识，示例：IF-001，从001开始递增
- interface_req_count: 接口需求项数，接口需求中所有条目的总和""",
    "doc_content": """
    接口
    """,
    "output_model": "InterfaceReqItemsOutput"
}

# 可靠性安全性需求项 prompt
reliable_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取可靠性安全性需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户输入的可靠性安全性和可维护性需求，生成可靠性安全性需求，其中可靠性安全性描述取自可靠性要求、安全性要求和可维护性要求三个章节下面的子项，每一项生成一个可靠性安全性需求，不要缺失，按照示例顺序生成对应测试需求标识，按照示例提取简要的相关说明，按实际需求条目生成。

文档：{doc_content}

请根据上述要求，提取并生成可靠性安全性需求项。每个需求项应包含：
- reliable_req_desc: 可靠性安全性需求描述，示例：1553B总线芯片周期性自检，每次发消息前判断总线芯片是否处于增强模式，如果发现异常则对总线芯片进行重新初始化配置
- reliable_req_id: 对应测试需求标识，示例：TR-AQ-001，TR和AQ固定不变，序号从1开始递增
- other_desc: 相关说明，根据描述内容从"提高安全性的容错方案"和"异常条件下软件的处理和保护能力"中选择一个填写
- reliable_req_count: 可靠性安全性需求项数，可靠性安全性需求中所有条目的总和""",
    "doc_content": """
    可靠性安全性和可维护性需求
    """,
    "output_model": "ReliableReqItemsOutput"
}

# 余量需求项 prompt
margin_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取余量需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户的余量需求，从设计约束中提取与软件存储有关系的项概括出余量需求描述，按照示例顺序生成对应测试需求标识，按照示例提取简要的相关说明，不能随意扩写。

文档：{doc_content}

请根据上述要求，提取并生成余量需求项。每个需求项应包含：
- margin_req_desc: 余量需求描述，每一项余量需求描述中隐含余量大于20%的要求，示例：SRAM存储空间使用不超过4M，余量大于20%
- margin_req_id: 对应测试需求标识，示例：TR-YL-001，TR和YL固定不变，序号从1开始递增
- other_desc: 相关说明，从余量需求描述中概括，示例1：存储余量；示例2：时间余量
- margin_req_count: 余量需求项数，余量需求中所有条目的总和""",
    "doc_content": """
    设计约束, 余量
    """,
    "output_model": "MarginReqItemsOutput"
}

# 边界需求项 prompt
boundary_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取边界测试需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户的边界测试需求，按照示例顺序生成对应测试需求标识，按照示例提取简要的相关说明，不能随意扩写。

文档：{doc_content}

请根据上述要求，提取并生成边界需求项。每个需求项应包含：
- boundary_req_desc: 边界需求描述，示例：若收到指令ID为0x06的内存下卸指令包(TC_MEM_DUMP)，下卸长度限幅96字节
- boundary_req_id: 对应测试需求标识，示例：TR-BJ-001，TR和BJ固定不变，序号从1开始递增
- other_desc: 相关说明，从边界需求描述中概括，示例1：数据输出域边界测试；示例2：数据输入域边界测试""",
    "doc_content": """
    设计约束
    """,
    "output_model": "BoundaryReqItemsOutput"
}

# 安全关键功能需求项 prompt
safety_critical_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取安全关键功能需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户的安全关键功能需求，从安全关键功能专项要求总结提取安全关键功能描述，尽量简洁，按照示例提取简要的相关说明，相关说明仿照示例总结归纳出来，不能随意扩写。

文档：{doc_content}

请根据上述要求，提取并生成安全关键功能需求项。每个功能项应包含：
- func_desc: 安全关键功能描述，示例：入轨段程控功能
- func_id: 需求标识，填写-
- other_desc: 相关说明，示例：直接导致危险发生的功能
- safety_critical_req_count: 安全关键功能需求项数，安全关键功能需求中所有条目的总和""",
    "doc_content": """
    安全关键功能专项要求
    """,
    "output_model": "SafetyCriticalReqItemsOutput"
}

# 恢复性需求项 prompt
recover_req_items = {
    "system_prompt": "你是一位专业的航天嵌入式软件测试工程师，负责从文档中提取恢复性需求项。",
    "user_prompt": """#角色
你是一位专业的航天嵌入式软件测试工程师，现在的任务是分析用户的恢复性需求，参考示例书写。

文档：{doc_content}

请根据上述要求，提取并生成恢复性需求项。每个需求项应包含：
- recover_req_desc: 恢复性需求描述，示例：系统异常复位后，启动软件重新运行，进行程序映像的加载，为正常功能，对业务执行无影响
- recover_req_id: 对应测试需求标识，示例：TR-HF-001，TR和HF固定不变，序号从1开始递增
- other_desc: 相关说明，固定填写：故障发生时保护系统状态""",
    "doc_content": """
    恢复性
    """,
    "output_model": "RecoverReqItemsOutput"
}


# ==================== Prompt 注册表（统一访问所有独立 Prompt 变量）====================
# 将所有独立定义的 prompt 变量收集到注册表中，便于统一访问和管理
# 每个 prompt 都是独立变量（使用 Pydantic 模型进行 structured output）

prompt_registry = {
    "perf_req_items": perf_req_items,
    "software_name": software_name,
    "use_of_interrupt": use_of_interrupt,
    "subsystem_relation": subsystem_relation,
    "cpu_storage": cpu_storage,
    "software_level": software_level,
    "dev_platform": dev_platform,
    "gpio_interface": gpio_interface,
    "dog_interface": dog_interface,
    "other_hardware_interface": other_hardware_interface,
    "interface_req_items": interface_req_items,
    "reliable_req_items": reliable_req_items,
    "margin_req_items": margin_req_items,
    "boundary_req_items": boundary_req_items,
    "safety_critical_req_items": safety_critical_req_items,
    "recover_req_items": recover_req_items,
}

