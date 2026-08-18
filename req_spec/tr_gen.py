"""
测试需求生成器
负责协调各个模块完成测试需求文档的生成
"""
import sys
import os
import logging

# 项目路径设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# LangChain 相关导入
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings

# 项目模块导入
from req_spec.doc_metadata_db import DocMetadataDB
from req_spec.doc_parser import DocParser
from req_spec.vector_store_manager import VectorStoreManager
from req_spec.requirement_extractor import RequirementExtractor
from req_spec.data_converter import DataConverter
from req_spec.fill_data_builder import FillDataBuilder
from req_spec.document_renderer import DocumentRenderer

# 配置导入
from config.settings import *


class TRGen:
    """测试需求生成器"""
    
    def __init__(self, docx_docs, output_filepath, project_id, create_by, project_type, user_id_2, doc_dict=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 基本参数
        self.docx_docs = docx_docs
        self.project_id = project_id
        self.output_filepath = output_filepath
        self.create_by = create_by
        self.project_type = project_type
        self.user_id_2 = user_id_2
        self.doc_dict = doc_dict
        
        # 初始化模型
        self.model = init_chat_model(
            model_provider=LLM_MODEL_PROVIDER,
                model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        self.embedding_model = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=EMBEDDING_API_KEY
        )
        
        # 数据库
        self.sql_collection_name = f"proj_{self.project_id}"
        self.metadata_db = DocMetadataDB(os.path.join(DB_DIR, self.sql_collection_name))
        
        # Word 模板路径
        self.template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.doc_template_path = os.path.join(self.template_dir, "第三方测试需求编制说明V1.05_template-0324.docx")
        
        # 初始化各模块
        self.doc_parser = DocParser(default_source="其他来源", chunk_size=6000, chunk_overlap=200)
        self.vector_store_manager = VectorStoreManager(
            project_id=self.project_id,
            embedding_model=self.embedding_model,
            metadata_db=self.metadata_db
        )
        self.requirement_extractor = RequirementExtractor(
            model=self.model,
            embedding_model=self.embedding_model,
            max_iterations=1,
            max_workers=30
        )
        self.data_converter = DataConverter()
        self.fill_data_builder = FillDataBuilder()
        self.document_renderer = DocumentRenderer(self.doc_template_path)
        
    def run(self):
        """执行测试需求生成流程"""
        try:
            self.logger.info("开始向量嵌入")
            
            # 0. 清理旧数据（每次运行前清理，避免数据累积）
            self.logger.info("清理旧数据...")
            self.metadata_db.delete_by_collection(self.sql_collection_name)
            self.vector_store_manager.clear_collection()
            
            # 1. 文档解析
            if self.doc_dict:
                self.logger.info("正在解析文档字典...")
                chunked_documents = self.doc_parser.parse(self.doc_dict, enable_chunking=True)
            else:
                self.logger.warning("未提供文档字典")
                chunked_documents = []
            
            # 2. 向量存储
            self.vector_store_manager.add_documents(chunked_documents)
            vector_store = self.vector_store_manager.vector_store
            
            # 3. 查询需求规格说明书
            req_dict = self.vector_store_manager.query_by_source(
                source_value="需求规格说明书",
                limit=100,
                return_format="dict"
            )
            self.logger.info(f"从数据库查询到的章节: {list(req_dict.keys())}")
            
            # 4. 设置输出目录
            output_dir = os.path.dirname(self.output_filepath) if self.output_filepath else "./test_output"
            
            # 5. 需求提取
            self.logger.info("开始需求提取...")
            extraction_results = self.requirement_extractor.extract_all(
                req_dict=req_dict,
                vector_store=vector_store,
                output_dir=output_dir,
                project_id=str(self.project_id)
            )
            
            workflow_results = extraction_results.get("workflow_results", [])
            other_req_results = extraction_results.get("other_req_results", {})
            self.other_req_results = other_req_results
            
            # 6. 数据转换
            self.logger.info("开始数据转换...")
            conversion_results = self.data_converter.convert_all(
                workflow_results=workflow_results,
                other_req_results=other_req_results,
                output_dir=output_dir,
                project_id=str(self.project_id)
            )
            
            req_infos = conversion_results.get("req_infos", [])
            req_item_dict = conversion_results.get("req_item_dict", {})
            output_dict = conversion_results.get("output_dict", {})
            
            # 保存到实例变量
            self.output_dict = output_dict
            self.req_infos = req_infos
            self.req_item_dict = req_item_dict
            
            # 7. 构建 fill_data
            self.logger.info("开始构建 fill_data...")
            fill_data = self.fill_data_builder.build(
                req_infos=req_infos,
                req_item_dict=req_item_dict,
                output_dict=output_dict,
                project_id=str(self.project_id),
                create_by=self.create_by,
                user_id=self.user_id_2,
                doc_type=getattr(self, 'doc_type', 1),
                output_dir=output_dir
            )
            self.fill_data = fill_data
            
            # 8. 渲染文档
            self.logger.info("开始渲染文档...")
            output_docx_path = self.document_renderer.render(
                output_dict=output_dict,
                fill_data=fill_data,
                project_id=str(self.project_id),
                create_by=self.create_by,
                output_dir=output_dir
            )
            
            self.logger.info("=" * 60)
            self.logger.info("TRGen 执行完成!")
            self.logger.info("=" * 60)
            
            if output_docx_path:
                self.logger.info(f"Word 文档已生成: {output_docx_path}")
            else:
                self.logger.warning("Word 文档生成失败，请检查模板文件")
                
        except Exception as e:
            self.logger.error(f"TRGen 执行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


if __name__ == "__main__":
    from req_spec.example.multi_doc_dict import multi_doc_dict
    
    tr_gen = TRGen(
        docx_docs=None,
        output_filepath="./test_output/output.docx",
        project_id=123,
        create_by="test_user",
        project_type=1,
        user_id_2="test_user_id",
        doc_dict=multi_doc_dict
    )
    
    print("=" * 60)
    print("Starting TRGen...")
    print("=" * 60)
    tr_gen.run()
