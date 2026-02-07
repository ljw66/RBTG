"""
fill_data Construction Module
Responsible for generating the fill_data structure required by document templates.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional


class FillDataBuilder:
    """
    fill_data Builder
    Generates the data structure used for document template rendering.
    """

    def __init__(self):
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

    def _add_dict(
            self,
            titleNo: str,
            title: str,
            level: int,
            children_titleNo: str,
            children_title: str,
            children_level: int,
            input_dict: List[Dict],
            func_id_keyword: str,
            item_handling_keyword: str,
            item_name_keyword: str
    ) -> Dict:
        """
        Build a standard chapter structure.

        :param titleNo: Chapter number (e.g., "4.1")
        :param title: Chapter title (e.g., "Functional Testing")
        :param level: Chapter level
        :param children_titleNo: Subchapter number
        :param children_title: Subchapter title
        :param children_level: Subchapter level
        :param input_dict: List of requirement items
        :param func_id_keyword: Field name of requirement ID
        :param item_handling_keyword: Field name of requirement description
        :param item_name_keyword: Field name of requirement name
        :return: Chapter structure dictionary
        """
        _dict = {
            "titleNo": titleNo,
            "title": title,
            "level": level,
            "funItem": [],
            "children": []
        }
        _children_dict = {
            "titleNo": children_titleNo,
            "title": children_title,
            "level": children_level,
            "children": None,
            "funItem": []
        }
        for item in input_dict:
            item_dict = {
                "funId": item.get(func_id_keyword, ""),
                "handling": item.get(item_handling_keyword, ""),
                "name": item.get(item_name_keyword, ""),
                "priority": None,
                "sufficiency": None,
                "testMethod": None,
                "passCriteria": None
            }
            _children_dict["funItem"].append(item_dict)

        _dict["children"].append(_children_dict)
        return _dict

    def _add_empty_dict(
            self,
            titleNo: str,
            title: str,
            level: int,
            children_titleNo: str,
            children_title: str,
            children_level: int
    ) -> Dict:
        """
        Build an empty chapter structure (used for requirement types not yet generated).
        """
        _dict = {
            "titleNo": titleNo,
            "title": title,
            "level": level,
            "funItem": [],
            "children": []
        }
        _children_dict = {
            "titleNo": children_titleNo,
            "title": children_title,
            "level": children_level,
            "children": None,
            "funItem": []
        }
        _dict["children"].append(_children_dict)
        return _dict

    def build(
            self,
            req_infos: List[Dict],
            req_item_dict: Dict[str, List[Dict]],
            output_dict: Dict[str, Any],
            project_id: str,
            create_by: str,
            user_id: str,
            doc_type: int = 1,
            output_dir: Optional[str] = None
    ) -> Dict:
        """
        Build the complete fill_data structure.

        :param req_infos: List of functional requirement information
        :param req_item_dict: Dictionary of functional requirement items
        :param output_dict: Dictionary of other requirement data
        :param project_id: Project ID
        :param create_by: Creator name
        :param user_id: User ID
        :param doc_type: Document type
        :param output_dir: Output directory (optional)
        :return: fill_data structure
        """
        # 1. Build the base structure
        fill_data = {
            "chapterTrList": {
                "titleNo": "4",
                "title": "Test Type Description",
                "level": 1,
                "funItem": [],
                "children": []
            }
        }

        # 2. Add Functional Testing section (4.1)
        self._add_functional_test(fill_data, req_infos, req_item_dict)

        # 3. Add Performance Testing section (4.2)
        self._add_performance_test(fill_data, output_dict)

        # 4. Add Interface Testing section (4.3)
        self._add_interface_test(fill_data, output_dict)

        # 5. Add Reliability & Safety Testing section (4.4)
        self._add_reliability_test(fill_data, output_dict)

        # 6. Add Margin Testing section (4.5)
        self._add_margin_test(fill_data, output_dict)

        # 7. Add Boundary Testing section (4.6)
        self._add_boundary_test(fill_data, output_dict)

        # 8. Add Data Processing Testing section (4.7) - empty for now
        fill_data["chapterTrList"]["children"].append(
            self._add_empty_dict("4.7", "Data Processing Testing", 2,
                                 "4.7.1", "Data Processing Testing Requirement Items", 3)
        )
        self.logger.info("  Data Processing Testing section added (empty).")

        # 9. Add Recovery Testing section (4.8)
        self._add_recovery_test(fill_data, output_dict)

        # 10. Add Stress Testing section (4.9) - empty for now
        fill_data["chapterTrList"]["children"].append(
            self._add_empty_dict("4.9", "Stress Testing", 2,
                                 "4.9.1", "Stress Testing Requirement Items", 3)
        )
        self.logger.info("  Stress Testing section added (empty).")

        # 11. Add Code Review section (4.10)
        self._add_code_review(fill_data, output_dict)

        # 12. Add Static Analysis section (4.11)
        self._add_static_analysis(fill_data)

        # 13. Add Logical Testing section (4.12)
        self._add_logical_test(fill_data)

        # 14. Add metadata
        fill_data["docType"] = doc_type
        fill_data["projectId"] = project_id
        fill_data["createBy"] = create_by
        fill_data["userId"] = user_id
        fill_data["softwareId"] = output_dict.get("software_id", "")
        fill_data["codeVersion"] = output_dict.get("code_version", "")

        # 15. Save to file (if output_dir is provided)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filldata_path = os.path.join(output_dir, f"fill_data_{project_id}.json")
            self._save_json(fill_data, filldata_path)
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"fill_data has been generated and saved to: {filldata_path}")
            self.logger.info(
                f"  - Contains {len(fill_data['chapterTrList']['children'])} test type sections in total"
            )
            self.logger.info(f"{'=' * 60}\n")

        return fill_data

    def _add_functional_test(self, fill_data: Dict, req_infos: List[Dict], req_item_dict: Dict):
        """Add Functional Testing section (4.1)."""
        func_test_node = {
            "titleNo": "4.1",
            "title": "Functional Testing",
            "level": 2,
            "funItem": [],
            "children": []
        }

        if req_infos:
            for i, req_info in enumerate(req_infos):
                chapter_title = req_info.get("title", "")
                child_node = {
                    "titleNo": f"4.1.{i + 1}",
                    "title": chapter_title,
                    "level": 3,
                    "funItem": req_item_dict.get(chapter_title, []),
                    "children": None
                }
                func_test_node["children"].append(child_node)

        if not func_test_node["children"]:
            func_test_node["children"] = None

        fill_data["chapterTrList"]["children"].append(func_test_node)
        self.logger.info(
            f"Functional Testing section added, containing {len(req_infos) if req_infos else 0} subsections."
        )

    def _add_performance_test(self, fill_data: Dict, output_dict: Dict):
        """Add Performance Testing section (4.2)."""
        perf_items = output_dict.get("perf_req_items", [])
        if perf_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.2", "Performance Testing", 2,
                               "4.2.1", "Performance Requirement Items", 3,
                               perf_items, "perf_req_id", "perf_req_desc", "other_desc")
            )
            self.logger.info(f"Performance Testing section added, containing {len(perf_items)} items.")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.2", "Performance Testing", 2,
                                     "4.2.1", "Performance Requirement Items", 3)
            )
            self.logger.info("  Performance Testing section added (empty).")

    def _save_json(self, data: Any, filepath: str):
        """Save JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
