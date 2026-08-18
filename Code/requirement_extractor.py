"""
Requirement Extraction Module
Responsible for extracting functional requirements and other non-functional requirements from documents
"""

import json
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from req_spec.functional_requirements_agent_1 import LabelSearchAgent
from req_spec.keyword_search_workflow_1 import KeywordSearchWorkflow


class RequirementExtractor:
    """
    Requirement Extractor
    Handles extraction of both functional and other non-functional requirements
    """

    # List of other requirement types
    OTHER_REQ_TYPES = [
        # Basic information
        "software_name",  # Software name and related info
        "subsystem_relation",  # System overview
        "cpu_storage",  # CPU and memory information
        "software_level",  # Software level info
        "dev_platform",  # Development platform info

        # Interface information
        "gpio_interface",  # GPIO interface info
        "dog_interface",  # Watchdog interface info
        "other_hardware_interface",  # Other hardware interface info
        "use_of_interrupt",  # Interrupt usage

        # Requirement items
        "perf_req_items",  # Performance requirements
        "interface_req_items",  # Interface requirements
        "reliable_req_items",  # Reliability and safety requirements
        "margin_req_items",  # Margin requirements
        "boundary_req_items",  # Boundary requirements
        "safety_critical_req_items",  # Safety-critical functional requirements
        "recover_req_items",  # Recovery requirements
    ]

    def __init__(
            self,
            model,
            embedding_model,
            max_iterations: int = 1,
            max_workers: int = 30
    ):
        """
        Initialize the Requirement Extractor

        :param model: LLM model instance
        :param embedding_model: Embedding model instance
        :param max_iterations: Maximum number of workflow iterations
        :param max_workers: Maximum number of parallel threads
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.model = model
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.max_workers = max_workers

    def extract_functional_requirements(
            self,
            req_dict: Dict[str, Dict],
            vector_store,
            target_title: str = "Functional Requirements",
            output_dir: Optional[str] = None,
            project_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract functional requirements

        :param req_dict: Dictionary of requirement documents {title: {content: "...", title_no: "..."}}
        :param vector_store: Vector store instance
        :param target_title: Target title (default "Functional Requirements")
        :param output_dir: Output directory (optional)
        :param project_id: Project ID (used for saving files)
        :return: List of workflow results
        """
        self.logger.info(f"Starting functional requirement extraction, target title: {target_title}")

        # 1. Use LabelSearchAgent to search for chapters related to functional requirements
        table_of_contents = list(req_dict.keys())
        chapters_dict = {title: info.get('content', '') for title, info in req_dict.items()}

        agent = LabelSearchAgent(chapters_dict=chapters_dict, model=self.model)
        result = agent.analyze(table_of_contents=table_of_contents, label=target_title)
        matched_chapters = result.matched_chapters
        self.logger.info(f"Chapters returned by LabelSearchAgent: {matched_chapters}")

        # Filter out chapters not present in req_dict (to avoid LLM hallucinations)
        valid_chapters = [ch for ch in matched_chapters if ch in req_dict]
        invalid_chapters = [ch for ch in matched_chapters if ch not in req_dict]
        if invalid_chapters:
            self.logger.warning(
                f"The following chapters are not in the document and will be ignored: {invalid_chapters}")
        matched_chapters = valid_chapters
        self.logger.info(f"Valid matched chapters: {matched_chapters}")

        if not matched_chapters:
            self.logger.warning(f"No valid chapters found for '{target_title}'")
            return []

        # 2. Process each matched chapter in parallel
        generation_requirement = "Generate detailed functional requirement specifications, including function descriptions, inputs/outputs, processing flow, exception handling, and other complete information"

        def process_chapter(chapter_title):
            """Process a single chapter"""
            try:
                self.logger.info(f"Processing chapter: {chapter_title}")

                chapter_workflow = KeywordSearchWorkflow(
                    model=self.model,
                    embeddings=self.embedding_model,
                    main_doc_dict=req_dict,
                    global_vector_store=vector_store,
                    generation_requirement=generation_requirement,
                    max_iterations=self.max_iterations
                )
                result = chapter_workflow.run(
                    keyword=chapter_title,
                    generation_requirement=generation_requirement
                )

                self.logger.info(f"Chapter '{chapter_title}' processed")
                self.logger.info(f"  - Matched chapter found: {result.get('matched_chapter') is not None}")
                self.logger.info(f"  - Content sufficient: {result.get('evaluation', {}).get('sufficient', False)}")

                final_result = result.get('final_result', '')
                if final_result:
                    self.logger.info(f"    ✓ Generated result length: {len(final_result)} characters")

                return {
                    "chapter_title": chapter_title,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"Error processing chapter '{chapter_title}': {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "chapter_title": chapter_title,
                    "result": {"error": str(e)}
                }

        # 3. Use thread pool to process in parallel
        workflow_results = []
        actual_workers = min(self.max_workers, len(matched_chapters))
        self.logger.info(f"Using {actual_workers} threads to process {len(matched_chapters)} chapters in parallel")

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(process_chapter, chapter_title): chapter_title
                       for chapter_title in matched_chapters}

            results_dict = {}
            for future in concurrent.futures.as_completed(futures):
                chapter_title = futures[future]
                try:
                    result = future.result()
                    results_dict[chapter_title] = result
                except Exception as e:
                    self.logger.error(f"Chapter '{chapter_title}' processing failed: {e}")
                    results_dict[chapter_title] = {
                        "chapter_title": chapter_title,
                        "result": {"error": str(e)}
                    }

            # Arrange results according to original order
            for chapter_title in matched_chapters:
                if chapter_title in results_dict:
                    workflow_results.append(results_dict[chapter_title])

        # 4. Save results if output directory is provided
        if workflow_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"workflow_results_{project_id or 'unknown'}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✓ Functional requirement results saved to: {output_file}")

        # 5. Statistics
        total_chapters = len(workflow_results)
        matched_count = sum(1 for r in workflow_results if r['result'].get('matched_chapter') is not None)
        sufficient_count = sum(
            1 for r in workflow_results if r['result'].get('evaluation', {}).get('sufficient', False))
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Functional requirement extraction completed:")
        self.logger.info(f"  - Total chapters: {total_chapters}")
        self.logger.info(f"  - Chapters matched: {matched_count}/{total_chapters}")
        self.logger.info(f"  - Content sufficient: {sufficient_count}/{total_chapters}")
        self.logger.info(f"{'=' * 60}")

        return workflow_results

    def extract_other_requirements(
            self,
            req_dict: Dict[str, Dict],
            vector_store,
            req_types: Optional[List[str]] = None,
            output_dir: Optional[str] = None,
            project_id: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Extract other non-functional requirements (performance, interface, etc.)

        :param req_dict: Dictionary of requirement documents
        :param vector_store: Vector store instance
        :param req_types: List of requirement types to process (default uses OTHER_REQ_TYPES)
        :param output_dir: Output directory (optional)
        :param project_id: Project ID
        :return: Dictionary of results for each requirement type {req_type: [results]}
        """
        self.logger.info("Starting extraction of other non-functional requirements...")

        # Import prompt configuration
        try:
            from req_spec.prompts.other_prompt import prompt_registry, get_output_model
        except ImportError as e:
            self.logger.warning(f"Cannot import prompt registry: {e}, skipping other requirements extraction")
            return {}

        prompts = prompt_registry
        req_types = req_types or self.OTHER_REQ_TYPES
        other_req_results = {}

        for req_type in req_types:
            if req_type not in prompts:
                self.logger.warning(f"  Skipping {req_type}: not found in prompts")
                continue

            self.logger.info(f"Processing requirement type: {req_type}")
            prompt_config = prompts[req_type]

            # Get output model class
            output_model_class = get_output_model(req_type)
            if output_model_class is None:
                self.logger.warning(f"  {req_type} has no corresponding output model, skipping")
                continue

            # Extract keywords
            doc_content = prompt_config.get("doc_content", "").strip()
            if not doc_content:
                self.logger.warning(f"  {req_type} doc_content is empty, skipping")
                continue

            keywords = [k.strip() for k in doc_content.split(',') if k.strip()]
            if not keywords:
                self.logger.warning(f"  {req_type} has no valid keywords, skipping")
                continue

            # Build prompt_template
            prompt_template = {
                "system_prompt": prompt_config.get("system_prompt", ""),
                "user_prompt": prompt_config.get("user_prompt", ""),
                "extract_content": "",
            }

            # Function to process a single keyword
            def process_keyword(keyword, req_type=req_type, prompt_template=prompt_template,
                                output_model_class=output_model_class):
                try:
                    self.logger.info(f"  Processing keyword: {keyword}")

                    keyword_workflow = KeywordSearchWorkflow(
                        model=self.model,
                        embeddings=self.embedding_model,
                        main_doc_dict=req_dict,
                        global_vector_store=vector_store,
                        generation_requirement=f"Extract information related to {req_type} from the document",
                        max_iterations=self.max_iterations,
                        prompt_template=prompt_template,
                        output_model=output_model_class,
                        skip_chapter_match=True
                    )

                    result = keyword_workflow.run(
                        keyword=keyword,
                        generation_requirement=f"Extract information related to {req_type} from the document"
                    )

                    final_result = result.get('final_result', '')
                    if final_result:
                        self.logger.info(f"    ✓ Generated result length: {len(final_result)} characters")

                    return {
                        "keyword": keyword,
                        "result": result
                    }
                except Exception as e:
                    self.logger.error(f"Error processing keyword '{keyword}': {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return {
                        "keyword": keyword,
                        "result": {"error": str(e)}
                    }

            # Process keywords in parallel
            req_type_results = []
            actual_workers = min(self.max_workers, len(keywords))
            self.logger.info(f"  Using {actual_workers} threads to process {len(keywords)} keywords in parallel")

            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {executor.submit(process_keyword, keyword): keyword
                           for keyword in keywords}

                results_dict = {}
                for future in concurrent.futures.as_completed(futures):
                    keyword = futures[future]
                    try:
                        result = future.result()
                        results_dict[keyword] = result
                    except Exception as e:
                        self.logger.error(f"Keyword '{keyword}' processing failed: {e}")
                        results_dict[keyword] = {
                            "keyword": keyword,
                            "result": {"error": str(e)}
                        }

                # Arrange results according to original order
                for keyword in keywords:
                    if keyword in results_dict:
                        req_type_results.append(results_dict[keyword])

            other_req_results[req_type] = req_type_results
            self.logger.info(f"  ✓ {req_type} processed, total {len(keywords)} keywords")

        # Save results
        if other_req_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"other_req_results_{project_id or 'unknown'}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(other_req_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✓ Other requirement type results saved to: {output_file}")

        return other_req_results

    def extract_all(
            self,
            req_dict: Dict[str, Dict],
            vector_store,
            output_dir: Optional[str] = None,
            project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract all requirements (functional + other)

        :param req_dict: Dictionary of requirement documents
        :param vector_store: Vector store instance
        :param output_dir: Output directory
        :param project_id: Project ID
        :return: Dictionary containing all results
        """
        # Extract functional requirements
        workflow_results = self.extract_functional_requirements(
            req_dict=req_dict,
            vector_store=vector_store,
            output_dir=output_dir,
            project_id=project_id
        )

        # Extract other requirements
        other_req_results = self.extract_other_requirements(
            req_dict=req_dict,
            vector_store=vector_store,
            output_dir=output_dir,
            project_id=project_id
        )

        return {
            "workflow_results": workflow_results,
            "other_req_results": other_req_results
        }
