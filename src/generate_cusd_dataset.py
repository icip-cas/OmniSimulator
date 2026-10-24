import argparse
import difflib
import glob
import json
import os
import queue
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from dynamic_persona import build_prediction_questions, _parse_timestamp
    from prompt_utils import format_action_context, format_action_result, render_prompt_template
except ImportError as exc:
    print(f"Failed to import project modules from {SRC_DIR}: {exc}")
    sys.exit(1)

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "sk-dummy")
DEFAULT_MODEL_NAME = os.getenv("CUSD_MODEL", "gpt-oss-120b")
DEFAULT_API_BASES = os.getenv("CUSD_API_BASES", "http://127.0.0.1:30000/v1") # replace the url of the teacher model, such as GPT-OSS-120B
DEFAULT_MAX_WORKERS = int(os.getenv("CUSD_MAX_WORKERS", "64"))

HIGH_VALUE_ACTIONS = {
    "视频浏览": {"like", "collect", "share", "comment"},
    "直播间": {"send_gift", "add_to_cart", "follow"},
    "商城购物": {"purchase", "order_success", "add_to_cart"},
    "广告推荐": {"conversion", "activation", "purchase", "submit", "click"},
    "电商客服对话": {"purchase", "positive_feedback"},
}

def as_action_list(action):
    actions = action.get("action", [])
    if isinstance(actions, dict):
        return [actions]
    return actions if isinstance(actions, list) else []

def build_api_endpoints(api_bases=None, max_workers=DEFAULT_MAX_WORKERS):
    """Parse comma-separated URLs from CLI arguments or environment defaults."""
    values = api_bases if api_bases is not None else DEFAULT_API_BASES
    if isinstance(values, str):
        values = [values]
    bases = [base.strip() for value in values for base in value.split(",") if base.strip()]
    if not bases:
        raise ValueError("Provide at least one API base URL via --api_base or CUSD_API_BASES.")
    return [{"api_base": base, "max_workers": max_workers} for base in bases]

def is_high_value_action(item: dict) -> bool:
    domain = item.get("type", "")
    if domain not in HIGH_VALUE_ACTIONS:
        return False
    target_actions = HIGH_VALUE_ACTIONS[domain]
    actions = as_action_list(item)
    for act in actions:
        if act.get("type") in target_actions:
            return True
    return False

def has_sufficient_context_for_prediction(action: dict) -> bool:
    action_type = action.get("type", "")
    context = action.get("context", {})

    if action_type == "视频浏览":
        caption = context.get("caption", "")
        ocr = context.get("ocr", "")
        asr = context.get("asr", "")
        ocr = ocr[:500] if ocr else ocr
        asr = asr[:500] if asr else asr
        if (not caption or not caption.strip()) and (not ocr or not ocr.strip()) and (not asr or not asr.strip()):
            return False
    elif action_type == "直播间":
        live_title = context.get("live_title", "")
        live_category = context.get("live_category", "")
        if (not live_title or not live_title.strip()) and (not live_category or not live_category.strip()):
            return False

    return True

def should_filter_action(action: dict) -> bool:
    action_type = action.get("type", "")
    context = action.get("context", {})
    action_list = as_action_list(action)

    if action_type == "广告推荐":
        show_cnt = context.get("show_cnt")
        if show_cnt is not None and show_cnt == 0:
            return True

    if action_type == "直播间":
        if (len(context) == 2 and 'live_comment_cnt' in context and 'live_like_cnt' in context and
            context.get('live_comment_cnt') == 0 and context.get('live_like_cnt') == 0):
            return True

    for act in action_list:
        if act.get("type") == "watch":
            play_duration = act.get("play_duration")
            if play_duration is not None:
                if isinstance(play_duration, str):
                    try:
                        if float(play_duration.replace("秒", "").strip()) == 0: return True
                    except (ValueError, AttributeError): pass
                elif isinstance(play_duration, (int, float)):
                    if play_duration == 0: return True
            if act.get("watch_seconds") is not None and act.get("watch_seconds") == 0:
                return True

    return False

def _balanced_sample(candidates_all: list, total_count: int, m_buckets: int = 10, seed: int = None, high_value_ratio: float = 0.15, max_high_ratio: float = 0.4) -> list:
    import math
    if seed is not None: random.seed(seed)
    if not candidates_all or total_count <= 0: return []
    n = len(candidates_all)
    if n <= total_count: return candidates_all.copy()

    for item in candidates_all:
        if 'valuable_score' not in item:
            action = item.get("action", item)
            score = 1.0
            context = action.get("context", {})
            text_len = sum(len(str(v)) for k, v in context.items() if isinstance(v, str))
            score += math.log1p(text_len) * 0.5
            questions = item.get("questions", [])
            if questions:
                pos_cnt = sum(1 for q in questions if (isinstance(q.get("true_value"), (int, float)) and q.get("true_value") > 0) or (isinstance(q.get("true_value"), str) and len(q.get("true_value").strip()) > 0))
                score += (pos_cnt / len(questions)) * 10.0
            item['valuable_score'] = score

    actual_buckets = min(m_buckets, n // 2, total_count)
    if actual_buckets < 1: actual_buckets = 1
    base_per_bucket = total_count // actual_buckets
    remainder = total_count % actual_buckets
    bucket_data_size = n // actual_buckets

    buckets = []
    start = 0
    for i in range(actual_buckets):
        end = n if i == actual_buckets - 1 else start + bucket_data_size
        buckets.append(candidates_all[start:end])
        start = end

    sampled = []
    for i, bucket_items in enumerate(buckets):
        if not bucket_items: continue
        target_k = base_per_bucket + (1 if i < remainder else 0)
        if target_k <= 0: continue
        if len(bucket_items) <= target_k:
            sampled.extend(bucket_items); continue

        domain_pools = defaultdict(lambda: {'high': [], 'low': []})
        for item in bucket_items:
            action = item.get("action", item)
            domain = action.get("type", "unknown")
            key = 'high' if is_high_value_action(action) else 'low'
            domain_pools[domain][key].append(item)

        for d in domain_pools:
            domain_pools[d]['high'].sort(key=lambda x: x.get('valuable_score', 0), reverse=True)
            domain_pools[d]['low'].sort(key=lambda x: x.get('valuable_score', 0), reverse=True)

        active_domains = list(domain_pools.keys())
        random.shuffle(active_domains)
        bucket_sampled = []
        domain_idx = 0

        while len(bucket_sampled) < target_k and active_domains:
            cur = active_domains[domain_idx]
            pool = domain_pools[cur]
            has_h = len(pool['high']) > 0
            has_l = len(pool['low']) > 0

            if not has_h and not has_l:
                active_domains.pop(domain_idx)
                if active_domains: domain_idx %= len(active_domains)
                continue

            target_ratio = min(high_value_ratio * 2.0, max_high_ratio)
            if has_h and has_l: pick_h = random.random() < target_ratio
            elif has_h: pick_h = True
            else: pick_h = False
            src = pool['high'] if pick_h else pool['low']

            top_m = min(len(src), 3)
            idx_to_pop = random.randint(0, top_m - 1)
            bucket_sampled.append(src.pop(idx_to_pop))

            if active_domains: domain_idx = (domain_idx + 1) % len(active_domains)

        sampled.extend(bucket_sampled)

    sampled.sort(key=lambda x: x.get("idx", 0))
    for item in sampled:
        item.pop('valuable_score', None)
    return sampled

def format_history_sequence(sequence, current_ts, max_events=30):
    history = sequence[-max_events:] if len(sequence) > max_events else sequence
    lines = []
    for i, e in enumerate(history, 1):
        ts = e.get("timestamp", "")
        t = e.get("type", "未知场景")
        ctx = format_action_context(e)
        res = format_action_result(e)
        lines.append(f"{i}. [{ts}] {t} | 内容: {ctx} | 动作: {res}")
    return "\n".join(lines) if lines else "无近期记录。"

def _call_llm_api_batch(prompt, system_prompt, api_base, api_key, model, n=5, temperature=0.9, max_retries=3, enable_thinking=False):
    """Generate multiple responses for one prompt."""
    client = OpenAI(api_key=api_key, base_url=api_base)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "max_tokens": 2048,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        } if enable_thinking else {}
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM API batch generation failed after {max_retries} attempts: {e}")

def _extract_prediction(response_text, task_type):
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if task_type == "binary":
        match = re.search(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"
    elif task_type == "continuous":
        match = re.search(r"(\d+)", cleaned)
        return match.group(1) if match else "unknown"
    return cleaned.strip()

def filter_responses_by_gt(responses, gt, task_type):
    """Select the best valid response against ground truth."""
    gt_val = None
    if task_type == "binary":
        gt_val = "yes" if gt == 1 else "no"
    elif task_type == "continuous":
        gt_val = str(int(gt) if isinstance(gt, (int, float)) else gt)
    else:
        gt_val = str(gt)

    discards_info = []

    result = {
        "best_response": None,
        "alternative_cots": [],
        "all_failed": False,
        "gt_val": gt_val,
        "discards_info": discards_info
    }

    valid_format_responses = []
    for resp in responses:
        if "<think>" not in resp or "</think>" not in resp:
            discards_info.append({"reason": "Missing_Think_Tag", "raw": resp, "gt": gt_val})
            continue
        pred = _extract_prediction(resp, task_type)
        valid_format_responses.append({"resp": resp, "pred": pred})

    if not valid_format_responses:
        result["all_failed"] = True
        return result

    if task_type == "binary":
        matched = []
        unmatched = []
        for item in valid_format_responses:
            if item["pred"] == gt_val:
                matched.append(item["resp"])
            else:
                unmatched.append(item["resp"])
                discards_info.append({"reason": "Mismatch_Truth", "raw": item["resp"], "pred": item["pred"], "gt": gt_val})

        if matched:
            result["best_response"] = random.choice(matched)
            all_valid = [r["resp"] for r in valid_format_responses]
            all_valid.remove(result["best_response"])
            result["alternative_cots"] = all_valid
        else:
            result["all_failed"] = True
            result["alternative_cots"] = [r["resp"] for r in valid_format_responses]

    elif task_type == "continuous":
        scored_candidates = []
        for item in valid_format_responses:
            try:
                pred_f = float(item["pred"])
                gt_f = float(gt_val)
                err = abs(pred_f - gt_f)
                scored_candidates.append({"resp": item["resp"], "err": err, "pred": item["pred"]})
            except:
                discards_info.append({"reason": "Parse_Numeric_Failed", "raw": item["resp"], "pred": item["pred"], "gt": gt_val})

        if scored_candidates:
            scored_candidates.sort(key=lambda x: x["err"])
            result["best_response"] = scored_candidates[0]["resp"]

            for cand in scored_candidates[1:]:
                discards_info.append({"reason": "Not_Closest", "raw": cand["resp"], "pred": cand["pred"], "gt": gt_val, "err": cand["err"]})
        else:
            result["all_failed"] = True

    else:
        scored_candidates = []
        for item in valid_format_responses:
            if item["pred"].strip().lower() == gt_val.strip().lower():
                scored_candidates.append({"resp": item["resp"], "sim": 1.0, "pred": item["pred"]})
            else:
                sim = difflib.SequenceMatcher(None, item["pred"].lower(), gt_val.lower()).ratio()
                discards_info.append({"reason": "Must_Match_Exactly", "raw": item["resp"], "pred": item["pred"], "gt": gt_val, "sim": sim})

        if scored_candidates:
            result["best_response"] = scored_candidates[0]["resp"]
            for cand in scored_candidates[1:]:
                discards_info.append({"reason": "Not_Highest_Relevance", "raw": cand["resp"], "pred": cand["pred"], "gt": gt_val, "sim": cand["sim"]})
        else:
            result["all_failed"] = True

    return result

def _compute_continuous_hint_range(gt_val):
    """Compute a soft teacher hint range for continuous targets."""
    try:
        gt_f = float(gt_val)
    except (TypeError, ValueError):
        return None, None
    if gt_f < 10:
        low = max(0, gt_f - 5)
        high = gt_f + 5
    else:
        low = max(0, gt_f * 0.5)
        high = gt_f * 1.5
    return int(low), int(high)

def build_forward_prompt(target_event, q_info, sequence, user_profile=""):
    current_ts = _parse_timestamp(target_event.get("timestamp", ""))
    history_str = format_history_sequence(sequence, current_ts)
    scene = target_event.get("type", "未知")
    context_str = format_action_context(target_event, is_target=True)

    q_type = q_info["type"]
    if q_type == "binary":
        from dynamic_persona import _YES_NO_QUESTIONS
        question_str = _YES_NO_QUESTIONS.get(q_info["field"], "请问你会执行该行为吗？(Yes/No)") + " (请用 Yes 或 No 作为最终答案)"
        constraint_str = "仅输出 Yes 或 No"
        template_ans = "Yes or No"
    elif q_type == "continuous":
        from dynamic_persona import _CONTINUOUS_QUESTIONS
        question_str = _CONTINUOUS_QUESTIONS.get(q_info["field"], "你预计会在该内容停留多少秒？") + " (请仅输出数字结果)"
        constraint_str = "仅输出数字（例如秒数）"
        template_ans = "15"
    else:
        question_str = q_info.get("question", "请输出你要预期的具体文字内容。")
        constraint_str = "仅输出核心文本内容，不要包含任何前缀或解释"
        template_ans = "你预测的具体文本内容"

    if user_profile:
        first_period_idx = user_profile.find('。')
        if first_period_idx != -1:
            user_profile_short = user_profile[:first_period_idx] + '。'
        else:
            user_profile_short = user_profile
    else:
        user_profile_short = ""

    profile_text = f"{user_profile_short}" if user_profile_short else ""

    student_prompt_base = render_prompt_template(
        "forward_student_base.txt",
        profile_text=profile_text,
        history_str=history_str,
        scene=scene,
        context_str=context_str,
        question_str=question_str,
    )
    format_protocol = render_prompt_template(
        "forward_output_protocol.txt",
        constraint_str=constraint_str,
        template_ans=template_ans,
    )

    student_prompt = student_prompt_base + format_protocol

    if q_type == "binary":
        teacher_prompt = student_prompt

    elif q_type == "continuous":
        gt_val = q_info.get("true_value", 0)
        range_low, range_high = _compute_continuous_hint_range(gt_val)
        if range_low is not None:
            hint_section = render_prompt_template(
                "forward_continuous_hint.txt",
                range_low=range_low,
                range_high=range_high,
            )
        else:
            hint_section = ""
        teacher_prompt = student_prompt_base + hint_section + format_protocol

    else:
        gt_text = str(q_info.get("true_value", ""))
        hint_section = render_prompt_template("forward_text_hint.txt", gt_text=gt_text)
        teacher_prompt = student_prompt_base + hint_section + format_protocol

    return student_prompt, teacher_prompt

def generate_cusd_dataset(
    input_path, output_path, target_users=200, test_length=50,
    seed=42, num_n_samples=5, max_retries=3, temperature=0.9,
    max_samples=0, api_endpoints=None, api_key=DEFAULT_API_KEY,
    model_name=DEFAULT_MODEL_NAME
):
    random.seed(seed)
    print("=" * 70)
    print("CUSD RFT dataset generation")
    print("=" * 70)
    print(f"  Input path: {input_path}")
    print(f"  Rejection-sampling attempts per prompt: {num_n_samples}")
    print(f"  Teacher model: {model_name} (temperature={temperature})")

    pending_items = []

    def add_pending_item(target_event, q_info, history_seq, user_profile, user_id):
        student_prompt, teacher_prompt = build_forward_prompt(target_event, q_info, history_seq, user_profile)
        pending_items.append({
            "student_prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
            "task_type": q_info["type"],
            "ground_truth": q_info["true_value"],
            "field": q_info.get("field", ""),
            "user_id": user_id,
            "timestamp": target_event.get("timestamp", ""),
        })

    if os.path.isfile(input_path):
        print("Mode: prepared experiment file")
        with open(input_path, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)

        from dynamic_persona import build_prediction_questions
        user_list = experiment_data.get("users", []) if isinstance(experiment_data, dict) else experiment_data

        for user_entry in user_list:
            uid = user_entry.get("user_id", "unknown")
            user_profile = user_entry.get("user_profile", "")
            if "test_actions" in user_entry:
                base_history = user_entry.get("base_history", [])
                test_time_all_actions = user_entry.get("test_time_all_actions", [])
                test_actions = user_entry.get("test_actions", [])

                for test_item in test_actions:
                    target_event = test_item.get("action", {})
                    if not target_event: continue
                    test_time_index = test_item.get("test_time_index", 0)
                    history_seq = base_history + test_time_all_actions[:test_time_index]

                    questions = build_prediction_questions(target_event)
                    if not questions: continue
                    for q_info in questions:
                        add_pending_item(target_event, q_info, history_seq, user_profile, uid)
            else:
                target_event = user_entry.get("test_action", {})
                if not target_event: continue
                history_seq = user_entry.get("action_history", [])
                questions = build_prediction_questions(target_event)
                if not questions: continue
                for q_info in questions:
                    add_pending_item(target_event, q_info, history_seq, user_profile, uid)
    else:
        print(f"Mode: raw dataset directory ({target_users} users x {test_length} events)")
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        user_data_dict = {}
        for f_path in json_files:
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for uid, u_profile in data.items():
                        act_history = u_profile.get("action_history", [])
                        if len(act_history) >= 20:
                            user_data_dict[uid] = u_profile
                        if len(user_data_dict) >= target_users:
                            break
            except Exception:
                pass
            if len(user_data_dict) >= target_users: break

        for uid, u_data in user_data_dict.items():
            user_profile = u_data.get("user_profile", "")
            all_actions = u_data.get("action_history", [])
            all_actions.sort(key=lambda x: _parse_timestamp(x.get("timestamp", "")))

            test_pool = all_actions
            valid_action_wrappers = []
            for idx, action in enumerate(test_pool):
                if not has_sufficient_context_for_prediction(action): continue
                if should_filter_action(action): continue
                questions = build_prediction_questions(action)
                if not questions: continue
                valid_action_wrappers.append({"action": action, "idx": idx, "questions": questions})

            if not valid_action_wrappers: continue

            sampled_wrappers = _balanced_sample(valid_action_wrappers, test_length, m_buckets=10, seed=seed)
            for wrapper in sampled_wrappers:
                target_event = wrapper["action"]
                history_seq = all_actions[:wrapper["idx"]]
                for q_info in wrapper["questions"]:
                    add_pending_item(target_event, q_info, history_seq, user_profile, uid)

    if max_samples > 0:
        pending_items = pending_items[:max_samples]
        print(f"Debug limit enabled: using the first {max_samples} candidate prompts.")

    print(f"Prepared {len(pending_items)} Forward-RFT candidate prompts.")

    system_prompt = render_prompt_template("system_prompt.txt")

    endpoints = api_endpoints or build_api_endpoints()
    endpoint_queue = queue.Queue()
    total_workers_expected = 0
    for ep in endpoints:
        workers = int(ep.get("max_workers", 4))
        total_workers_expected += workers
        for _ in range(workers):
            endpoint_queue.put({"api_base": ep.get("api_base")})

    print(f"Initialized {len(endpoints)} API endpoint(s) with {total_workers_expected} total worker(s).")

    stats = {"success": 0, "kept": 0, "discarded": 0, "error": 0, "binary_all_failed": 0}
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sft_ready_path = output_path.replace(".json", "_sft_ready.json")
    binary_failed_path = output_path.replace(".json", "_binary_all_failed.json")

    f_out = open(output_path, "w", encoding="utf-8")
    f_out.write("[\n")

    f_sft = open(sft_ready_path, "w", encoding="utf-8")
    f_sft.write("[\n")

    f_fail = open(binary_failed_path, "w", encoding="utf-8")
    f_fail.write("[\n")

    first_out = True
    first_fail = True

    debug_log_path = os.path.join(output_dir or ".", "debug_discard_reasons.log")
    f_debug = open(debug_log_path, "w", encoding="utf-8")

    def _process_one(item):
        ep = endpoint_queue.get()
        try:
            responses = _call_llm_api_batch(
                prompt=item["teacher_prompt"], system_prompt=system_prompt,
                api_base=ep["api_base"], api_key=api_key, model=model_name,
                n=num_n_samples, temperature=temperature, max_retries=max_retries
            )
            eval_result = filter_responses_by_gt(responses, item["ground_truth"], item["task_type"])
            return {"eval_result": eval_result, "item": item, "error": None}
        except Exception as e:
            return {"eval_result": None, "item": item, "error": str(e)}
        finally:
            endpoint_queue.put(ep)

    total_workers = endpoint_queue.qsize()
    with tqdm(total=len(pending_items), desc="RFT distillation", unit="item") as pbar:
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = {executor.submit(_process_one, item): item for item in pending_items}
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result["error"]:
                    stats["error"] += 1
                else:
                    stats["success"] += 1
                    eval_result = result["eval_result"]
                    item = result["item"]

                    if eval_result["all_failed"] and item["task_type"] == "binary":
                        stats["binary_all_failed"] += 1
                        stats["discarded"] += 1
                        failed_data = {
                            "user_id": item["user_id"],
                            "timestamp": item.get("timestamp", ""),
                            "task_type": item["task_type"],
                            "field": item.get("field", ""),
                            "ground_truth": eval_result["gt_val"],
                            "student_prompt": item["student_prompt"],
                            "all_cots": eval_result["alternative_cots"]
                        }
                        if not first_fail:
                            f_fail.write(",\n")
                        f_fail.write(json.dumps(failed_data, ensure_ascii=False, indent=2))
                        f_fail.flush()
                        first_fail = False

                    elif not eval_result["all_failed"]:
                        stats["kept"] += 1
                        best_response = eval_result["best_response"]

                        teacher_sft_data = {
                            "conversations": [
                                {"from": "human", "value": item["teacher_prompt"]},
                                {"from": "gpt", "value": best_response}
                            ],
                            "task_type": item["task_type"],
                            "field": item.get("field", ""),
                            "ground_truth": eval_result["gt_val"],
                            "user_id": item["user_id"],
                            "timestamp": item.get("timestamp", ""),
                            "alternative_cots": eval_result.get("alternative_cots", [])
                        }

                        student_sft_data = {
                            "conversations": [
                                {"from": "human", "value": item["student_prompt"]},
                                {"from": "gpt", "value": best_response}
                            ],
                            "task_type": item["task_type"],
                            "field": item.get("field", ""),
                            "ground_truth": eval_result["gt_val"],
                            "user_id": item["user_id"],
                            "timestamp": item.get("timestamp", ""),
                            "alternative_cots": eval_result.get("alternative_cots", [])
                        }

                        if not first_out:
                            f_out.write(",\n")
                            f_sft.write(",\n")
                        f_out.write(json.dumps(teacher_sft_data, ensure_ascii=False, indent=2))
                        f_sft.write(json.dumps(student_sft_data, ensure_ascii=False, indent=2))

                        f_out.flush()
                        f_sft.flush()

                        first_out = False
                    else:
                        stats["discarded"] += 1

                    if eval_result.get("discards_info"):
                        f_debug.write(f"\n[USER_ID: {item['user_id']} | Type: {item['task_type']}]\n")
                        for d in eval_result["discards_info"]:
                            f_debug.write(f"  -> Reason: {d['reason']} | GT: {d.get('gt')} | Pred: {d.get('pred', 'N/A')}\n")
                            f_debug.write(f"     RAW OUTPUT:\n       {repr(d['raw'])[:2048]}\n")
                        f_debug.flush()

                pbar.set_postfix(kept=stats["kept"], disc=stats["discarded"])

    f_out.write("\n]\n")
    f_sft.write("\n]\n")
    f_fail.write("\n]\n")
    f_out.close()
    f_sft.close()
    f_fail.close()
    f_debug.close()

    print("\n" + "=" * 70)
    print("CUSD generation report")
    print(f"  Candidate prompts: {len(pending_items)}")
    print(f"  Discarded samples: {stats['discarded']} (binary all-failed: {stats['binary_all_failed']})")
    print(f"  Kept samples: {stats['kept']}")
    if stats['success'] > 0:
        print(f"  Keep rate: {stats['kept']/stats['success']:.1%}")
    print("=" * 70)
    print(f"Teacher traces saved to: {output_path}")
    print(f"SFT-ready dataset saved to: {sft_ready_path}")
    print(f"Binary all-failed records saved to: {binary_failed_path}")
    print(f"Discard diagnostics saved to: {debug_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CUSD ShareGPT-style training data.")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSON file or directory containing raw user JSON files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the teacher-trace output JSON file.")
    parser.add_argument("--n_samples", type=int, default=3, help="Best-of-N rejection-sampling attempts per prompt.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit the number of candidate prompts; 0 means no limit.")
    parser.add_argument("--target_users", type=int, default=200, help="Maximum number of users to load from a raw data directory.")
    parser.add_argument("--test_length", type=int, default=50, help="Number of sampled target events per user in raw-directory mode.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for teacher generation.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for each LLM request.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="OpenAI-compatible model name.")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--api_base", action="append", help="OpenAI-compatible API base URL(s), separated by commas.")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Workers allocated to each API endpoint.")

    args = parser.parse_args()
    try:
        endpoints = build_api_endpoints(args.api_base, args.max_workers)
    except ValueError as exc:
        parser.error(str(exc))

    generate_cusd_dataset(
        input_path=args.input_path,
        output_path=args.output_file,
        num_n_samples=args.n_samples,
        max_samples=args.max_samples,
        target_users=args.target_users,
        test_length=args.test_length,
        max_retries=args.max_retries,
        temperature=args.temperature,
        api_endpoints=endpoints,
        api_key=args.api_key,
        model_name=args.model,
    )
