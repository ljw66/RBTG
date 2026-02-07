"""
Data Conversion Module

Responsible for converting requirement extraction results into the format
expected by the legacy codebase.
"""

import json
import logging
import os
from typing import List, Dict, Any, Tuple, Optional


class DataConverter:
    """
    Data Converter

    Converts workflow_results and other_req_results into:
    - req_infos
    - req_item_dict
    - output_dict
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

    def convert_functional_requirements(
            self,
            workflow_results: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Convert functional requirement results

        :param workflow_results: List of workflow results for functional requirements
        :return: (req_infos, req_item_dict)
        """
        req_infos = []
        req_item_dict = {}

        for workflow_item in workflow_results:
            chapter_title = workflow_item.get("chapter_title", "")
            result = workflow_item.get("result", {})
            final_result_str = result.get("final_result", "")

            if not final_result_str:
                self.logger.warning(
                    f"No generated result for chapter '{chapter_title}', skipping"
                )
                continue

            try:
                # Parse final_result JSON string
                final_result_dict = json.loads(final_result_str)

                # Extract fun_items
                fun_items = final_result_dict.get("fun_items", [])
                if not fun_items:
                    self.logger.warning(
                        f"No fun_items found in chapter '{chapter_title}', skipping"
                    )
                    continue

                # Build req_infos format
                req_infos.append({
                    "title": chapter_title,
                    "response": {
                        "fun_items": fun_items,
                        "fun_req_summary": final_result_dict.get("fun_req_summary", ""),
                        "input_stream_desc": final_result_dict.get("input_stream_desc", []),
                        "output_stream_desc": final_result_dict.get("output_stream_desc", [])
                    }
                })

                # Build req_item_dict format (with additional fields)
                processed_fun_items = []
                for fun_item in fun_items:
                    processed_item = {
                        "funId": fun_item.get("fun_id", ""),
                        "handling": fun_item.get("handling", ""),
                        "name": chapter_title,
                        "priority": None,
                        "sufficiency": None,
                        "testMethod": None,
                        "passCriteria": None
                    }
                    processed_fun_items.append(processed_item)

                req_item_dict[chapter_title] = processed_fun_items

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse final_result for chapter '{chapter_title}': {e}"
                )
                continue
            except Exception as e:
                self.logger.error(
                    f"Error while processing chapter '{chapter_title}': {e}"
                )
                import traceback
                self.logger.error(traceback.format_exc())
                continue

        self.logger.info(
            f"Functional requirement conversion completed: "
            f"{len(req_infos)} chapters, "
            f"{sum(len(items) for items in req_item_dict.values())} functional items"
        )

        return req_infos, req_item_dict

    def convert_other_requirements(
            self,
            other_req_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Convert other requirement extraction results

        :param other_req_results: Other requirement results in the form {req_type: [keyword_results]}
        :return: output_dict
        """
        output_dict = {}

        if not other_req_results:
            return output_dict

        for req_type, keyword_results in other_req_results.items():
            if not keyword_results:
                continue

            self.logger.info(f"Processing requirement type: {req_type}")

            # Merge results from all keywords
            merged_items = []
            merged_count = ""
            is_processed = False

            for keyword_item in keyword_results:
                keyword = keyword_item.get("keyword", "")
                result = keyword_item.get("result", {})
                final_result_str = result.get("final_result", "")

                if not final_result_str:
                    continue

                try:
                    final_result_dict = json.loads(final_result_str)

                    # Extract data depending on requirement type
                    if req_type == "perf_req_items":
                        items = final_result_dict.get("perf_req_items", [])
                        merged_items.extend(items)

                    elif req_type == "interface_req_items":
                        items = final_result_dict.get("interface_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("interface_req_count", "")
                        if count:
                            merged_count = count

                    elif req_type == "reliable_req_items":
                        items = final_result_dict.get("reliable_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("reliable_req_count", "")
                        if count:
                            merged_count = count

                    elif req_type == "margin_req_items":
                        items = final_result_dict.get("margin_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("margin_req_count", "")
                        if count:
                            merged_count = count

                    elif req_type == "boundary_req_items":
                        items = final_result_dict.get("boundary_req_items", [])
                        merged_items.extend(items)

                    elif req_type == "safety_critical_req_items":
                        items = final_result_dict.get("all_funcs", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("safety_critical_req_count", "")
                        if count:
                            merged_count = count

                    elif req_type == "recover_req_items":
                        items = final_result_dict.get("recover_req_summary", [])
                        merged_items.extend(items)

                    elif req_type in [
                        "software_name", "use_of_interrupt", "subsystem_relation",
                        "cpu_storage", "software_level", "dev_platform"
                    ]:
                        # Basic information types - flatten into the top-level output_dict
                        if req_type == "use_of_interrupt":
                            # use_of_interrupt contains a list
                            new_list = final_result_dict.get("interrupt_list", [])
                            if "use_of_interrupt" not in output_dict:
                                output_dict["use_of_interrupt"] = new_list
                            else:
                                existing = (
                                    output_dict["use_of_interrupt"]
                                    if isinstance(output_dict["use_of_interrupt"], list)
                                    else []
                                )
                                output_dict["use_of_interrupt"] = existing + new_list
                        else:
                            # Other types: flatten all fields into output_dict
                            for key, value in final_result_dict.items():
                                if key not in output_dict:
                                    output_dict[key] = value
                                else:
                                    self.logger.info(
                                        f"Field {key} already exists, keeping the first result"
                                    )
                        is_processed = True

                    elif req_type in [
                        "gpio_interface", "dog_interface", "other_hardware_interface"
                    ]:
                        # Interface types - store arrays directly
                        list_key = {
                            "gpio_interface": "gpio_interfaces",
                            "dog_interface": "dog_interfaces",
                            "other_hardware_interface": "other_hardware_interfaces"
                        }[req_type]

                        new_list = final_result_dict.get(list_key, [])

                        if req_type not in output_dict:
                            output_dict[req_type] = new_list
                        else:
                            # Merge interface lists
                            existing = (
                                output_dict[req_type]
                                if isinstance(output_dict[req_type], list)
                                else []
                            )
                            output_dict[req_type] = existing + new_list
                        is_processed = True

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"Failed to parse final_result for {req_type} keyword '{keyword}': {e}"
                    )
                    continue
                except Exception as e:
                    self.logger.error(
                        f"Error while processing {req_type} keyword '{keyword}': {e}"
                    )
                    continue

            # For merged types, store arrays directly (no nesting; count is computed in code)
            if not is_processed and merged_items:
                output_dict[req_type] = merged_items

        self.logger.info(
            f"Other requirement conversion completed: {len(output_dict)} requirement types"
        )
        self._log_output_dict_stats(output_dict)

        return output_dict

    def _log_output_dict_stats(self, output_dict: Dict[str, Any]):
        """Log statistics of output_dict"""
        for req_type, data in output_dict.items():
            if isinstance(data, list):
                self.logger.info(f"  - {req_type}: {len(data)} items")
            elif isinstance(data, str):
                self.logger.info(f"  - {req_type}: saved (string)")
            elif isinstance(data, dict):
                self.logger.info(f"  - {req_type}: saved (object)")
            else:
                self.logger.info(f"  - {req_type}: saved")

    def convert_all(
            self,
            workflow_results: List[Dict],
            other_req_results: Dict[str, List[Dict]],
            output_dir: Optional[str] = None,
            project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert all requirement extraction results

        :param workflow_results: Functional requirement workflow results
        :param other_req_results: Other requirement extraction results
        :param output_dir: Output directory (optional, for saving intermediate files)
        :param project_id: Project ID
        :return: Dictionary containing all converted results
        """
        # 1. Convert functional requirements
        req_infos, req_item_dict = self.convert_functional_requirements(workflow_results)

        # 2. Convert other requirements
        output_dict = self.convert_other_requirements(other_req_results)

        # 3. Merge into output_dict
        output_dict['req_infos'] = req_infos
        output_dict['func_req_items'] = "{{p chapters}}"  # Placeholder

        # 4. Save intermediate files (if output_dir is provided)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save req_item_dict
            if req_item_dict:
                self._save_json(
                    req_item_dict,
                    os.path.join(output_dir, "func_req_item.json")
                )
                self.logger.info(
                    f"Functional requirement item dictionary saved: "
                    f"{len(req_item_dict)} chapters"
                )

            # Save complete output_dict
            complete_output_file = os.path.join(
                output_dir,
                f"complete_output_dict_{project_id or 'unknown'}.json"
            )
            self._save_json(output_dict, complete_output_file)
            self.logger.info(
                f"Complete output dictionary saved to: {complete_output_file}"
            )

        # 5. Print conversion summary
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Conversion Summary:")
        self.logger.info(f"  - req_infos: {len(req_infos)} chapters")
        self.logger.info(f"  - req_item_dict: {len(req_item_dict)} chapters")
        self.logger.info(
            f"  - output_dict: {len(output_dict)} requirement types (including req_infos)"
        )
        self.logger.info(f"{'=' * 60}\n")

        return {
            "req_infos": req_infos,
            "req_item_dict": req_item_dict,
            "output_dict": output_dict
        }

    def _save_json(self, data: Any, filepath: str):
        """Save JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
