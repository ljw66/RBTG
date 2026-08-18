# 精简版多文档测试数据 - 用于快速测试

# ==================== 需求规格说明书（精简版） ====================
req_spec_dict = {
    "2 功能需求": {
        "content": "本章节描述系统的功能需求，包括用户管理功能。",
        "title_no": "2",
        "level": 1,
        "source": "需求规格说明书"
    },
    "2.1 用户管理功能": {
        "content": """用户管理功能包括用户注册和登录。

输入流：用户名（3-20字符）、邮箱、密码（8-32字符）。

处理逻辑：验证输入有效性，检查用户名是否存在，生成JWT令牌（有效期24小时）。登录失败5次锁定30分钟。

输出流：user_id、username、token（JWT格式）、expires_in（86400秒）。""",
        "title_no": "2.1",
        "level": 2,
        "source": "需求规格说明书"
    },
    "2.2 数据查询功能": {
        "content": """数据查询功能支持按条件查询和分页显示。

输入流：查询条件（关键字、时间范围）、分页参数（page、page_size）。

处理逻辑：解析查询条件，构建SQL语句，执行分页查询，计算总记录数。

输出流：records（数据列表）、total（总记录数）、page（当前页码）、page_size（每页条数）。""",
        "title_no": "2.2",
        "level": 2,
        "source": "需求规格说明书"
    },
    "3 性能需求": {
        "content": "系统响应时间：登录≤1秒，查询≤2秒。支持1000并发用户。看门狗前50秒不喂狗。",
        "title_no": "3",
        "level": 1,
        "source": "需求规格说明书"
    },
    "4 接口需求": {
        "content": "硬件接口：GPIO、1553B总线、看门狗。软件接口：RESTful API（JSON格式）。",
        "title_no": "4",
        "level": 1,
        "source": "需求规格说明书"
    },
    "5 可靠性需求": {
        "content": "1553B总线芯片周期性自检。系统异常复位后自动恢复，重新加载程序映像。",
        "title_no": "5",
        "level": 1,
        "source": "需求规格说明书"
    },
    "6 设计约束": {
        "content": "存储约束：SRAM≤4M（余量>20%）。边界：内存下卸限幅96字节，用户名3-20字符。",
        "title_no": "6",
        "level": 1,
        "source": "需求规格说明书"
    },
    "12 配置项基本信息": {
        "content": "软件名称：SMU应用软件。软件标识号：SMU-SW-001。代码版本号：2.04。",
        "title_no": "12",
        "level": 1,
        "source": "需求规格说明书"
    }
}

# ==================== 任务书（精简版） ====================
task_doc_dict = {
    "1 任务目标": {
        "content": "完成用户管理模块开发，功能测试通过率100%。",
        "title_no": "1",
        "level": 1,
        "source": "任务书"
    }
}

# ==================== 接口协议（精简版） ====================
interface_protocol_dict = {
    "1 用户接口": {
        "content": "注册：POST /api/v1/register，登录：POST /api/v1/login。Token有效期24小时。",
        "title_no": "1",
        "level": 1,
        "source": "接口协议"
    }
}

# ==================== 合并后的多文档字典 ====================
multi_doc_dict = {}
multi_doc_dict.update(req_spec_dict)
multi_doc_dict.update(task_doc_dict)
multi_doc_dict.update(interface_protocol_dict)

# ==================== 按来源分组的字典 ====================
doc_dict_by_source = {
    "需求规格说明书": req_spec_dict,
    "任务书": task_doc_dict,
    "接口协议": interface_protocol_dict
}

if __name__ == "__main__":
    print(f"精简版测试数据：共 {len(multi_doc_dict)} 个章节")
    for source, docs in doc_dict_by_source.items():
        print(f"  - {source}: {len(docs)} 个")
