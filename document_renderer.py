"""
Document Rendering Module
Responsible for rendering data into a Word document template
"""

import datetime
import logging
import os
from typing import Dict, Any, Optional

from docxtpl import DocxTemplate


class DocumentRenderer:
    """
    Document Renderer
    Renders data into a Word document template
    """

    def __init__(self, template_path: str):
        """
        Initialize the document renderer

        :param template_path: Path to the template file
        """
        self.template_path = template_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def render(
        self,
        output_dict: Dict[str, Any],
        fill_data: Dict[str, Any],
        project_id: str,
        create_by: str,
        output_dir: str
    ) -> Optional[str]:
        """
        Render a Word document

        :param output_dict: Dictionary containing various requirement data
        :param fill_data: Dictionary containing chapter structure data
        :param project_id: Project ID
        :param create_by: Creator name
        :param output_dir: Output directory
        :return: Generated document path, or None if failed
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Starting Word document rendering...")
        self.logger.info(f"{'='*60}")

        try:
            # Check whether the template file exists
            if not os.path.exists(self.template_path):
                self.logger.error(f"Template file does not exist: {self.template_path}")
                return None

            # Load the template
            self.logger.info(f"Loading template: {self.template_path}")
            doc = DocxTemplate(self.template_path)

            # Prepare rendering data
            render_data = self._prepare_render_data(
                output_dict, fill_data, project_id, create_by
            )

            # Render the template
            self.logger.info("Rendering template...")
            doc.render(render_data, autoescape=True)

            # Save the rendered document
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename
            output_filename = (
                f"Test_Requirements_{project_id}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            )
            output_path = os.path.join(output_dir, output_filename)

            doc.save(output_path)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Word document generated: {output_path}")
            self.logger.info(f"{'='*60}\n")

            return output_path

        except Exception as e:
            self.logger.error(f"Error occurred while rendering document: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _prepare_render_data(
        self,
        output_dict: Dict[str, Any],
        fill_data: Dict[str, Any],
        project_id: str,
        create_by: str
    ) -> Dict[str, Any]:
        """
        Prepare rendering data by merging output_dict and fill_data
        into the format required by the template.

        Common placeholders used in the template include:
        - {{model_name}} - Model name
        - {{software_name}} - Software name
        - {{software_id}} - Software identifier
        - {{code_version}} - Code version
        - {{doc_version}} - Document version
        - {{chapters}} - Chapter contents
        """
        render_data = {}

        # Basic information fields (now flattened at the top level of output_dict)
        render_data["software_name"] = output_dict.get("software_name", "")
        render_data["software_id"] = output_dict.get("software_id", "")
        render_data["code_version"] = output_dict.get("code_version", "")
        render_data["model_name"] = output_dict.get("model_name", "")
        render_data["doc_version"] = output_dict.get("doc_version", "V1.0")
        render_data["model_subsystem_name"] = output_dict.get("model_subsystem_name", "")
        render_data["req_spec_file_id"] = output_dict.get("req_spec_file_id", "")
        render_data["entrusting_addr"] = output_dict.get("entrusting_addr", "")

        # software_level related fields
        render_data["software_level"] = output_dict.get("software_level", "")
        render_data["programming_language"] = output_dict.get("programming_language", "")
        render_data["dev_unit"] = output_dict.get("dev_unit", "")
        render_data["subsystem_function"] = output_dict.get("subsystem_function", "")
        render_data["subsystem_function_qd"] = output_dict.get("subsystem_function_qd", "")

        # cpu_storage related fields
        render_data["cpu_desc"] = output_dict.get("cpu_desc", "")
        render_data["use_of_storage_io"] = output_dict.get("use_of_storage_io", "")

        # Other flattened fields
        render_data["subsystem_relation"] = output_dict.get("subsystem_relation", "")
        render_data["dev_platform"] = output_dict.get("dev_platform", "")

        # Project information
        render_data["project_id"] = project_id
        render_data["create_by"] = create_by
        render_data["create_date"] = datetime.datetime.now().strftime("%Y-%m-%d")

        # Chapter structure data (used for generating requirement tables)
        render_data["chapterTrList"] = fill_data.get("chapterTrList", {})
        render_data["chapters"] = fill_data.get("chapterTrList", {}).get("children", [])

        # Requirement data categories (now stored as arrays, counts computed via len())
        perf_items = output_dict.get("perf_req_items", [])
        interface_items = output_dict.get("interface_req_items", [])
        reliable_items = output_dict.get("reliable_req_items", [])
        margin_items = output_dict.get("margin_req_items", [])
        boundary_items = output_dict.get("boundary_req_items", [])
        recover_items = output_dict.get("recover_req_items", [])
        safety_items = output_dict.get("safety_critical_req_items", [])

        render_data["perf_req_items"] = perf_items
        render_data["interface_req_items"] = interface_items
        render_data["reliable_req_items"] = reliable_items
        render_data["margin_req_items"] = margin_items
        render_data["boundary_req_items"] = boundary_items
        render_data["recover_req_items"] = recover_items
        render_data["safety_critical_req_items"] = safety_items

        # Count fields (computed in code, not dependent on model output)
        render_data["perf_req_count"] = str(len(perf_items))
        render_data["interface_req_count"] = str(len(interface_items))
        render_data["reliable_req_count"] = str(len(reliable_items))
        render_data["margin_req_count"] = str(len(margin_items))
        render_data["boundary_req_count"] = str(len(boundary_items))
        render_data["recover_req_count"] = str(len(recover_items))
        render_data["safety_critical_req_count"] = str(len(safety_items))

        # Interface-related data (now stored as arrays)
        render_data["gpio_interface"] = output_dict.get("gpio_interface", [])
        render_data["dog_interface"] = output_dict.get("dog_interface", [])
        render_data["other_hardware_interface"] = output_dict.get("other_hardware_interface", [])
        render_data["use_of_interrupt"] = output_dict.get("use_of_interrupt", [])

        # Functional requirement data
        render_data["req_infos"] = output_dict.get("req_infos", [])
        render_data["func_req_items"] = output_dict.get("func_req_items", "")

        # Log rendering field summary (simplified logging)
        self.logger.info(f"Prepared rendering data with {len(render_data)} fields")

        return render_data
