import ctypes
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file

from service.doc_service.doc_usecase_service_test import GenUserCase
from flask_cors import CORS, cross_origin
from service.doc_service.tr_gen import TRGen
import ctypes

application = Flask(__name__)
CORS(application) # 启用跨域支持

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 存储地址
save_dir = '/root/code/req-spec-to-spec/webroot/storage/'
log_date_dir = "/root/code/req-spec-to-spec/statistic_log"

# 全局变量控制任务状态
tasks = {}
task_lock = threading.Lock()
stop_events = {}

def terminate_thread(thread):
    """强制终止线程"""
    if not thread.is_alive():
        return False

    # 获取线程
    thread_id = thread.ident

    # 使用ctypes强制终止线程
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id),
        ctypes.py_object(SystemExit)
    )
    if res == 0:
        logger.warning(f"线程{thread_id}不存在")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        logger.error(f"强制终止线程{thread_id}失败")
        return False
    else:
        logger.info(f"线程{thread_id}已强制终止")
        return True

@application.route('/req-spec-to-spec/stop-task', methods=['POST'])
def stop_task():
    """停止指定的文档生成任务"""
    global tasks
    request_id = str(uuid.uuid4())

    def error_resp(code, msg):
        response_data = {
            "code": code,
            "msg": msg,
            "request_id": request_id
        }
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return jsonify({"code": code, "msg": msg, "request_id": request_id}),200
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        # 检查是否有停止确认参数
        project_id = data.get('projectId')
        task_id = data.get('task_id')
        if project_id:
            task_id = project_id
        elif task_id:
            pass
        else:
            return error_resp(400, "缺少project_id或task_id参数")
        with task_lock:
            if task_id not in tasks:
                return error_resp(404, f"任务{task_id}不存在")
            if tasks[task_id]["status"] != "running":
                return error_resp(400, f"任务{task_id}状态为{tasks[task_id]['status']}, 无法停止")
            logger.info(f"收到停止任务请求，task_id: {task_id}")
            # 直接杀死线程
            thread = tasks[task_id].get("thread")

            if thread and thread.is_alive():
                success = terminate_thread(thread)
                if success:
                    logger.info(f"线程已成功终止，task_id: {task_id}")
                else:
                    logger.warning(f"线程终止失败，task_id: {task_id}")

            # 更新任务状态
            tasks[task_id]["status"] = "stopped"
            tasks[task_id]["result"] = "任务被强制终止"

            return jsonify({
                "code": "0",
                "msg": "任务停止请求已接收",
                "request_id": request_id,
                "data": {
                    "task_id": task_id,
                    "project_id": project_id,
                    "message": "文档生成任务将在当前步骤完成后停止"
                }
            }), 200
    except Exception as e:
        logger.info(f"停止任务失败：{e}")
        return error_resp("5000", "停止任务时发生内部错误")

@application.route('/req-spec-to-spec/task-status', methods=['GET'])
def task_status():
    """获取任务装填端点"""
    global tasks
    request_id = str(uuid.uuid4())
    # 清理已完成任务（超过1小时）
    current_time = time.time()
    with task_lock:
        tasks_to_remove = []
        for task_id, task_info in tasks.items():
            if task_info["status"] in ["completed", "failed", "stopped"] and current_time - task_info["start_time"] > 3600:
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del tasks[task_id]

    # 获取任务列表
    task_list = []
    with task_lock:
        for task_id, task_info in tasks.items():
            task_list.append({
                "task_id": task_id,
                "project_id": task_id,
                "status": task_info["status"],
                "start_time": task_info["start_time"],
                "duration": current_time - task_info["start_time"]
            })

    return jsonify({
        "code": "0",
        "msg": f"当前有{len(task_list)} 个任务",
        "request_id": request_id,
        "data": {
            "tasks": task_list,
            "total_tasks": len(task_list),
            "running_tasks": len([t for t in task_list if t['status'] == "running"]),
            "timestamp": str(uuid.uuid4())
        }
    }), 200

@application.route('/req-spec-to-spec/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取特定任务状态"""
    global tasks
    request_id = str(uuid.uuid4())

    with task_lock:
        if task_id in tasks:
            task_info = tasks[task_id]
            current_time = time.time()

            return jsonify({
                "code": "0",
                "msg": "任务状态查询成功",
                "request_id": request_id,
                "data":{
                    "task_id": task_id,
                    "project_id": task_id,
                    "status": task_info["status"],
                    "start_time": task_info["start_time"],
                    "duration": current_time - task_info["start_time"]
                       }
                }), 200
        else:
             return jsonify({
                 "code": "404",
                 "msg": f"任务 {task_id} 不存在",
                 "request_id": request_id
             }), 404


# 测试需求生成地址
# 接收文件
@application.route('/req-spec-to-spec/upload', methods=['POST'])
@cross_origin(application, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ['Content-Type', 'Authorization']
    }
})
def upload_file():
    if 'ruleForm' in request.form:
        # 打印原始的ruleForm内容
        print(f"ruleForm原始内容：{request.form['ruleForm']}")
        json_data = request.form['ruleForm']
        json_data = json.loads(json_data)
        # 取前端参数
        user_id = json_data.get('userId', '')
        user_input_dict = {
            'user_id': user_id,
            'model_name': json_data.get('model_name', ''),
            'software_name': json_data.get('software_name', ''),
            'software_id': json_data.get('software_id', ''),
            'code_version': json_data.get('code_version', ''),
            'req_spec_file_id': json_data.get('req_spec_file_id', ''),
            'model_subsystem_name': json_data.get('model_subsystem_name', '')
        }

    for name, file_storage in request.files.items():
        logger.info(f"received file: {file_storage.filename}")
        if file_storage.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        # 添加时间戳到文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # 用户名从输入参数拼接出来
        #型号名称配置项名称（软件标识 - 软件版本号）第三方测试需求_1.00
        #示例：XHDZ - 0000030数传分系统数传控制单元下位机软件（R_HJA03 - 2E_00 - 2.01）第三方测试需求_1.00.docx
        filename = f"{user_input_dict.get('model_name','')}{user_input_dict.get('software_name','')}（{user_input_dict.get('software_id', '')}-{user_input_dict.get('code_version', '')}）第三方测试需求"
        filename = filename.replace('/', '_').replace('\\', '_')
        ext = ".docx"
        new_filename = f"{filename}_{timestamp}{ext}"
        new_dirname = f"{filename}_{timestamp}"

        # 存放测试文件路径
        save_proj_dir = os.path.join(save_dir, new_dirname)
        os.makedirs(save_proj_dir, exist_ok=True)

        # 接收文件存放路径
        save_path = os.path.join(save_proj_dir, new_filename)
        file_storage.save(save_path)
        logger.info(f"File successfully uploaded to {save_path}")
        # 生成测试需求
        send_pre_filename = f"{filename}_{timestamp}{ext}"
        # 生成文件路径
        output_filepath = os.path.join(save_dir, send_pre_filename)
        output_tar_dir = os.path.join(save_proj_dir, "tar")
        os.makedirs(output_tar_dir, exist_ok=True)
        gen_user_case = GenUserCase(save_path, output_filepath, send_pre_filename, user_id, user_input_dict, output_tar_dir)
        gen_user_case.run()

    return jsonify({
        "success": True,
        "message": send_pre_filename,
        "code": 200,
        "userid": user_id
    }), 200


# 下载文件
@application.route('/req-spec-to-spec/download/<filename>', methods=['GET'])
def download_file(filename):
    # 获取文件路径
    file_path = os.path.join(save_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, mimetype='application/octet-stream')
    else:
        logger.error(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404


# 处理耗时统计API
@application.route('/req-spec-to-spec/processing-time', methods=['GET'])
@cross_origin()
def get_processing_time_stats():
    # 获取请求参数
    start_date_str = request.args.get('startDate', '')
    end_date_str = request.args.get('endDate', '')

    try:
        # 解析日期，如果没有提供日期，则使用默认值（过去7天）
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        else:
            # 默认为过去7天
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)

        # 获取日期范围内的所有天
        current_date = start_date
        days = []

        while current_date <= end_date:
            days.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        # 初始化按天统计的结果
        daily_stats = {day: 0 for day in days}

        # 读取日志目录中的处理时间记录文件
        log_dir = log_date_dir
        if os.path.exists(log_dir):
            # 创建字典记录每天的总耗时和文件数量
            daily_total_time = {day: 0 for day in days}
            daily_file_count = {day: 0 for day in days}
            for day in days:
                log_file = os.path.join(log_dir, f"process_time_{day}.log")

                # 如果存在该日期的日志文件，读取并计算总耗时
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) >= 3:
                                try:
                                    # 第三列是处理时间（单位：分钟）
                                    exec_time = float(parts[2])
                                    daily_total_time[day] += exec_time
                                    daily_file_count[day] += 1
                                except (ValueError, IndexError):
                                    continue
            for day in days:
                if daily_file_count[day] > 0: # 避免除零
                    daily_stats[day] = daily_total_time[day] / daily_file_count[day]
                else:
                    daily_stats[day] = 0
        # 格式化结果为所需的格式
        result = [{"x": day, "y": round(daily_stats[day], 2)} for day in days]
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"获取处理耗时统计时出错：{str(e)}")
        return jsonify({"error": str(e)}), 500

@application.route('/req-spec-to-spec/gen', methods=['POST'])
@cross_origin()
def gen_req():
    global tasks
    request_id = str(uuid.uuid4())
    def error_resp(code, msg):
        response_data = {
            "code": code,
            "msg": msg,
            "requestId": request_id
        }
        response = jsonify(response_data)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return jsonify({"code": code, "msg": msg, "requestId": request_id}), 200

    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        if not data:
            return error_resp("4000", "请求数据为空")

        # 使用project_id作为任务ID，确保一个项目只有一个任务
        task_id = data['projectId']

        # 检查是否已有相同项目的任务在运行
        with task_lock:
            if task_id in tasks and tasks[task_id]["status"] == "running":
                return error_resp(400, f"项目{task_id}已有任务正在运行，请等待完成或停止现有任务")

        required = ["projectId", "createBy", "filePath", "projectType", "softwareName", "userId"]
        missing = [k for k in required if not data.get(k)]
        if missing:
            return error_resp("4000", f"缺少参数: {', '.join(missing)}")
        # projectType数据类型检验
        try:
            project_type = int(data["projectType"])
        except ValueError:
            return error_resp("4000", "projectType 必须为整数")

        folder_path = data["filePath"]
        folder_path = folder_path.replace('/home/data/tmate-data/share-data/', '/root/var/data/')
        print(folder_path)
        if not os.path.exists(folder_path):
            return error_resp("4302", "文件夹不存在")

        if not os.path.isdir(folder_path):
            return error_resp("4302", "指定路径不是文件夹")
        # 查找包含"需求规格说明"的文档
        def find_requirement_docs(folder_path):
            requirement_docs = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if "需求规格说明" in file:
                        file_path = os.path.join(root, file)
                        requirement_docs.append({
                            "name": file,
                            "path": file_path,
                            "relative_path": os.path.relpath(file_path, folder_path)
                        })
            return requirement_docs

        # 查找需求文档
        requirement_docs = find_requirement_docs(folder_path)
        if not requirement_docs:
            return error_resp("4303", f"在文件夹 {folder_path} 中未找到包含 需求规格说明 的文档")
        target_doc = requirement_docs[0]
        file_path = target_doc["path"]

        # 检查文档格式，只支持.docx格式
        if not file_path.lower().endswith('.docx'):
            return error_resp("4303", f"不支持的文档格式，只支持.docx格式。当前文档: {os.path.basename(file_path)}")
        file_dir = os.path.dirname(file_path)
        file_name = data["softwareName"]
        # timestamp = int(time.time())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"第三方测试需求-{file_name}-{timestamp}.docx"
        output_filepath = os.path.join(file_dir, output_filename)
        # 共享路径下
        share_output_filepath = output_filepath.replace( '/root/var/data/', '/home/data/tmate-data/share-data/')
        tr_gen = TRGen(file_path, output_filepath, project_id=data["projectId"], create_by=data["createBy"], project_type=data["projectType"], user_id_2=data["userId"])
        # 在后台线程中运行任务
        def run_task():
            global tasks, stop_events
            try:
                stop_events[task_id] = threading.Event()
                with task_lock:
                    tasks[task_id] = {
                        "status": "running",
                        "task": tr_gen,
                        "thread": task_thread,
                        "start_time": time.time()
                    }
                logger.info(f"开始执行任务，task_id:{task_id}, request_id: {request_id}")

                # 检查是否被要求停止

                result = tr_gen.run()

                logger.info(f"文档生成任务完成， task_id: {task_id}, request_id: {request_id}, 结果：{result}")

                with task_lock:
                    tasks[task_id]["status"] = "completed"
                    tasks[task_id]["result"] = result

            except Exception as e:
                logger.error(f"文档生成任务失败，错误{e}")
                with task_lock:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["error"] = str(e)

            finally:
                if task_id in stop_events:
                    del stop_events[task_id]
        # 启动后台线程
        task_thread = threading.Thread(target=run_task, daemon=True)
        task_thread.start()

        return jsonify({
            "code": "0",
            "msg": "成功",
            "requestId": request_id,
            "data": share_output_filepath
        }), 200

    except Exception as e:
        logger.error(f"处理错误： {e}")
        return error_resp("5000", "服务内部错误")


if __name__ == '__main__':
    # 测试用8003
    application.run(host='0.0.0.0', port=8002, debug=True, use_reloader=False)
