from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

from prompt_utils import format_action_context, format_action_result, render_prompt_template

SCENE_NAMES = {
    "视频浏览": "短视频",
    "直播间": "直播间",
    "商城购物": "电商购物",
    "广告推荐": "广告",
    "电商客服对话": "客服对话",
    "搜索行为": "搜索"
}

TARGET_ACTION_NAMES = {
    "watch": "观看",
    "like": "点赞",
    "comment": "评论",
    "share": "分享",
    "collect": "收藏",
    "download": "下载",
    "follow": "关注",
    "unfollow": "取消关注",
    "dislike": "不感兴趣",
    "report": "举报",
    "cart": "加购",
    "purchase": "购买/下单",
    "conversion": "转化",
    "activation": "激活",
    "submit": "提交表单",
    "send_gift": "送礼",
    "click_cart": "点击购物车",
    "join_group": "加入粉丝团",
    "search": "搜索",
    "show": "展示",
    "click": "点击",
    "dialogue": "客服对话"
}

DRIFT_DESCRIPTION_TEMPLATES = {
    "video_ctr":         {"subject": "视频点击率", "上升": "提升", "下降": "下滑"},
    "emotion_mean":      {"subject": "情绪状态", "上升": "趋于积极", "下降": "趋于消极"},
    "purchase_rate":     {"subject": "购买转化率", "上升": "有所提升", "下降": "明显下降"},
    "avg_price":         {"subject": "消费价格带", "上升": "上移（消费升级信号）", "下降": "下移（降级信号）"},
    "refund_rate":       {"subject": "退款频率", "上升": "显著上升（不满意信号）", "下降": "有所下降"},
    "session_duration":  {"subject": "会话时长", "上升": "变长（粘性增强）", "下降": "缩短（粘性下降）"},
    "novelty_seeking":   {"subject": "内容探索欲", "上升": "明显增强", "下降": "趋于保守"},
    "complaint_rate":    {"subject": "投诉行为", "上升": "频率上升（体验下降信号）", "下降": "有所减少"},
}

_YES_NO_QUESTIONS = {
    "video_completed": "该用户会把这个视频完整看完吗？",
    "video_liked": "该用户会为这个视频点赞吗？",
    "video_commented": "该用户会在这个视频下发表评论吗？",
    "video_shared": "该用户会把这个视频分享给朋友吗？",
    "video_collected": "该用户会收藏这个视频吗？",
    "video_followed": "该用户会关注这个视频的作者吗？",
    "shop_added_to_cart": "该用户会把这件商品加入购物车吗？",
    "shop_order_success": "该用户会购买这件商品吗？",
    "ad_liked": "该用户会为这条广告点赞吗？",
    "ad_commented": "该用户会在这条广告下发表评论吗？",
    "ad_activated": "该用户会激活广告中的应用/服务吗？",
    "ad_form_submitted": "该用户会提交广告中的表单吗？",
    "live_liked": "该用户会为主播点赞吗？",
    "live_commented": "该用户会在直播间发送弹幕吗？",
    "live_sent_gift": "该用户会给主播送礼物吗？",
    "live_followed": "该用户会关注这个主播吗？",
    "live_shared": "该用户会分享这个直播间吗？",
    "live_clicked_cart": "该用户会把直播间商品加入购物车吗？",
    "search_clicked": "该用户会点击搜索结果吗？",
}

FIELD_TO_ACT_TYPE = {
    "video_completed": "watch",
    "video_liked": "like",
    "video_commented": "comment",
    "video_shared": "share",
    "video_collected": "collect",
    "video_followed": "follow",
    "shop_added_to_cart": "cart",
    "shop_order_success": "purchase",
    "ad_liked": "like",
    "ad_commented": "comment",
    "ad_activated": "activation",
    "ad_form_submitted": "submit",
    "live_liked": "like",
    "live_commented": "comment",
    "live_sent_gift": "send_gift",
    "live_followed": "follow",
    "live_shared": "share",
    "live_clicked_cart": "click_cart",
    "search_clicked": "click",
}

_CONTINUOUS_QUESTIONS = {
    "video_watch_seconds": "你预计该用户会在这个视频上观看多少秒？",
    "live_watch_seconds": "你预计该用户会在这个直播间停留多少秒？",
    "ad_watch_seconds": "你预计该用户会在这条广告上停留多少秒？",
}

@dataclass
class L1BaselinePersona:
    """Long-term stable persona baseline."""
    natural_language_summary: str
    emotion: dict
    scene_stats: dict
    anchor_vectors: dict
    anchor_confidence_intervals: dict
    meta: dict

@dataclass
class L3ShortTermState:
    """Short-term session state."""
    session_start_ts: int
    current_scene: str
    events_in_session: int
    fatigue_level: float
    consecutive_skips: int
    last_positive_event_ago_min: float
    current_emotion: str
    emotion_trigger: Optional[str]
    active_intent: Optional[str]
    intent_evidence: List[str]
    intent_strength: float
    pre_scene_context: str
    cross_scene_signal: Optional[str]

def _parse_timestamp(ts_str: str) -> int:
    """Parse a timestamp string into milliseconds."""
    try:
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)
    except:
        return 0

def _get_hour_from_ts(ts: int) -> int:
    try:
        dt = datetime.fromtimestamp(ts / 1000.0)
        return dt.hour
    except:
        return 12

def _is_positive_action(action_item: dict) -> bool:
    """Return whether an event contains a positive action."""
    actions = action_item.get("action", [])
    if isinstance(actions, dict):
        actions = [actions]
    for act in actions:
        act_type = act.get("type", "")
        if act_type in ["like", "comment", "share", "collect", "download", "purchase", "follow",
                        "click", "conversion", "activation", "submit", "send_gift", "join_group", "dialogue"]:
            return True
        if act_type == "cart" and act.get("is_add_to_cart"):
            return True
        if act_type == "click_cart" and act.get("is_click_cart_action"):
            return True
        if act_type == "watch" and (act.get("completed") or act.get("is_complete_play")):
            return True
    return False

def _is_negative_action(action_item: dict) -> bool:
    """Return whether an event contains a negative action."""
    actions = action_item.get("action", [])
    if isinstance(actions, dict):
        actions = [actions]
    for act in actions:
        act_type = act.get("type", "")
        if act_type in ["dislike", "report", "unfollow"]:
            return True
        if act_type == "watch":
            pd_val = act.get("play_duration", None)
            if pd_val is None:
                continue
            if isinstance(pd_val, str):
                try:
                    pd_val = float(pd_val.replace("秒", "")) if "秒" in pd_val else float(pd_val)
                except:
                    continue
            if isinstance(pd_val, (int, float)) and 0 <= pd_val <= 2.0:
                return True
    return False

def compute_l1_baseline(all_l1_events: List[Dict]) -> L1BaselinePersona:
    """Compute the long-term persona baseline from historical events."""
    if not all_l1_events:
        return L1BaselinePersona(
            natural_language_summary="缺乏长期数据，默认基线。",
            emotion={"mean": 0.0, "negative_ratio": 0.0},
            scene_stats={},
            anchor_vectors={},
            anchor_confidence_intervals={},
            meta={"persona_reliable": False}
        )

    scene_counts = Counter()
    scene_positive_counts = Counter()
    action_counts = defaultdict(Counter)
    scene_watch_seconds_sum = Counter()
    scene_watch_count = Counter()

    for e in all_l1_events:
        scene = e.get("type", "unknown")
        scene_counts[scene] += 1

        if _is_positive_action(e):
            scene_positive_counts[scene] += 1

        acts = e.get("action", [])
        if isinstance(acts, dict): acts = [acts]

        performed_acts = set()
        for act in acts:
            act_type = act.get("type", "")

            is_valid_act = True
            if act_type == "cart" and not act.get("is_add_to_cart"):
                is_valid_act = False
            elif act_type == "click_cart" and not act.get("is_click_cart_action"):
                is_valid_act = False
            elif act_type == "watch" and not (act.get("completed") or act.get("is_complete_play")):
                is_valid_act = False
            elif act_type == "purchase" and not (act.get("order_success") or act.get("paid") or act.get("is_pay")):
                is_valid_act = False

            if is_valid_act:
                performed_acts.add(act_type)

            if act_type == "watch":
                ws = 0
                if "watch_seconds" in act:
                    ws = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    pd = act.get("play_duration")
                    if isinstance(pd, str):
                        try: ws = float(pd.replace("秒", "").strip())
                        except: ws = 0
                    elif isinstance(pd, (int, float)):
                        ws = float(pd)
                if ws > 0:
                    scene_watch_seconds_sum[scene] += ws
                    scene_watch_count[scene] += 1

        for act_type in performed_acts:
            action_counts[scene][act_type] += 1

    scene_stats = {}
    for scene, count in scene_counts.items():
        base_rate = scene_positive_counts[scene] / count if count > 0 else 0
        specific_rates = {}
        for act_type, act_count in action_counts[scene].items():
            specific_rates[act_type] = act_count / count
        avg_watch_seconds = scene_watch_seconds_sum[scene] / scene_watch_count[scene] if scene_watch_count[scene] > 0 else 0
        scene_stats[scene] = {"action_rate": base_rate, "count": count, "specific_rates": specific_rates, "avg_watch_seconds": avg_watch_seconds}

    anchor_vectors = {
        "video_ctr": scene_stats.get("视频浏览", {}).get("action_rate", 0.0),
        "emotion_mean": 0.05,
        "purchase_rate": scene_stats.get("商城购物", {}).get("action_rate", 0.0),
    }

    anchor_ci = {
        k: (v * 0.7, v * 1.3) if v > 0 else (0.0, 0.05)
        for k, v in anchor_vectors.items()
    }
    neg_count = sum(1 for e in all_l1_events if _is_negative_action(e))
    negative_ratio = neg_count / len(all_l1_events) if all_l1_events else 0.0

    top_scenes = [s for s, _ in scene_counts.most_common(2)]
    top_scenes_str = "、".join(top_scenes) if top_scenes else "各类"

    return L1BaselinePersona(
        natural_language_summary=f"主要活跃于{top_scenes_str}场景，具备一定的内容消费与转化习惯。",
        emotion={"mean": anchor_vectors["emotion_mean"], "negative_ratio": negative_ratio},
        scene_stats=scene_stats,
        anchor_vectors=anchor_vectors,
        anchor_confidence_intervals=anchor_ci,
        meta={"persona_reliable": True}
    )

def compute_daily_stats(recent_events: List[Dict], window_days: int) -> dict:
    daily_stats = {
        "video_ctr": [],
        "purchase_rate": [],
        "emotion_mean": []
    }
    chunks = np.array_split(recent_events, window_days) if len(recent_events) >= window_days else [recent_events]
    for chunk in chunks:
        if not len(chunk): continue
        v_count = sum(1 for e in chunk if e.get("type") == "视频浏览")
        v_clicks = sum(1 for e in chunk if e.get("type") == "视频浏览" and _is_positive_action(e))
        p_count = sum(1 for e in chunk if e.get("type") == "商城购物")
        p_clicks = sum(1 for e in chunk if e.get("type") == "商城购物" and _is_positive_action(e))

        daily_stats["video_ctr"].append(v_clicks/v_count if v_count else 0)
        daily_stats["purchase_rate"].append(p_clicks/p_count if p_count else 0)
        daily_stats["emotion_mean"].append(0.0)

    return daily_stats

def compute_l2_trend(recent_events: List[Dict], l1_baseline: L1BaselinePersona, window_days: int = 14) -> dict:
    daily_stats = compute_daily_stats(recent_events, window_days)
    drifts = {}

    for feature_name, daily_values in daily_stats.items():
        if len(daily_values) < 3:
            continue

        anchor_val = l1_baseline.anchor_vectors.get(feature_name)
        if anchor_val is None:
            continue

        recent_mean = float(np.mean(daily_values[-7:])) if len(daily_values) >= 7 else float(np.mean(daily_values))
        ci_low, ci_high = l1_baseline.anchor_confidence_intervals.get(feature_name, (anchor_val * 0.8, anchor_val * 1.2))

        is_significant = recent_mean < ci_low or recent_mean > ci_high
        if not is_significant:
            continue

        delta = recent_mean - anchor_val
        delta_pct = delta / (abs(anchor_val) + 1e-8)

        slope = np.polyfit(range(len(daily_values[-7:])), daily_values[-7:], 1)[0] if len(daily_values) >= 7 else 0
        trend_direction = "持续" if np.sign(slope) == np.sign(delta) else "趋于稳定"

        drifts[feature_name] = {
            "anchor": anchor_val,
            "recent": recent_mean,
            "delta_pct": delta_pct,
            "direction": "上升" if delta > 0 else "下降",
            "trend": trend_direction,
            "severity": "显著" if abs(delta_pct) > 0.3 else "轻微"
        }
    return drifts

def generate_l2_summary_with_llm(drifts: dict, l1_summary: str) -> str:
    if not drifts:
        return "近14天行为模式与长期基线基本一致，无显著变化。"

    top_drifts = sorted(drifts.items(), key=lambda x: abs(x[1]["delta_pct"]), reverse=True)[:5]
    sentences = []

    for feat, info in top_drifts:
        desc = DRIFT_DESCRIPTION_TEMPLATES.get(feat, {})
        if not desc:
            continue
        direction_word = desc.get(info["direction"], info["direction"])
        severity_word = "明显" if info["severity"] == "显著" else "略有"
        trend_word = f"，且{info['trend']}" if info["trend"] != "未知" else ""

        sentences.append(
            f"{desc['subject']}{severity_word}{direction_word}"
            f"（{info['recent']:.2f} vs 基线{info['anchor']:.2f}，变化{info['delta_pct']:+.0%}）{trend_word}"
        )
    return "近14天，" + "；".join(sentences) + "。"

def extract_current_session(recent_events: List[Dict], max_gap_ms: int = 10 * 60000) -> List[Dict]:
    """Extract the current session using an inactivity gap threshold."""
    if not recent_events:
        return []
    session_events = []
    last_ts = _parse_timestamp(recent_events[-1].get("timestamp", ""))

    for e in reversed(recent_events):
        ts = _parse_timestamp(e.get("timestamp", ""))
        if (last_ts - ts) > max_gap_ms:
            break
        session_events.insert(0, e)
        last_ts = ts
    return session_events

def compute_l3_state(recent_events: List[Dict], current_event: Dict) -> L3ShortTermState:
    session_events = extract_current_session(recent_events)
    last_3h_events = recent_events

    current_ts = _parse_timestamp(current_event.get("timestamp", ""))
    current_scene = current_event.get("type", "unknown")

    consecutive_skips = 0
    for e in reversed(session_events):
        if _is_negative_action(e):
            consecutive_skips += 1
        else:
            break

    if len(session_events) >= 10:
        early_density = sum(1 for e in session_events[:5] if _is_positive_action(e)) / 5
        recent_density = sum(1 for e in session_events[-5:] if _is_positive_action(e)) / 5
        density_drop = max(0.0, early_density - recent_density)
    else:
        density_drop = 0.0

    last_positive_ts = current_ts
    for e in reversed(session_events):
        if _is_positive_action(e):
            last_positive_ts = _parse_timestamp(e.get("timestamp", ""))
            break
    last_positive_ago = (current_ts - last_positive_ts) / 60000.0 if session_events else 0

    fatigue = min(1.0, consecutive_skips * 0.15 + density_drop * 0.4 + min(last_positive_ago / 30.0, 1.0) * 0.45)

    current_emotion = "neutral"
    emotion_trigger = None

    recent_30min = [e for e in last_3h_events if (current_ts - _parse_timestamp(e.get("timestamp", ""))) < 30 * 60000]
    for e in reversed(recent_30min):
        acts = e.get("action", [])
        if isinstance(acts, dict): acts = [acts]
        act_types = [a.get("type", "") for a in acts]

        if "report" in act_types or "dislike" in act_types or "unfollow" in act_types:
            current_emotion = "negative"
            emotion_trigger = "刚发起过举报或拉黑等负面反馈"
            break
        if "purchase" in act_types:
            current_emotion = "positive"
            emotion_trigger = "刚完成了购买"
            break

    active_intent = None
    intent_evidence = []
    intent_strength = 0.0

    recent_search = [e for e in last_3h_events if e.get("type") == "搜索行为" and (current_ts - _parse_timestamp(e.get("timestamp", ""))) < 2 * 3600000]

    if recent_search and current_scene == "商城购物":
        latest_search = recent_search[-1]
        kw = None
        search_acts = latest_search.get("action", [])
        if isinstance(search_acts, dict): search_acts = [search_acts]
        for sa in search_acts:
            if sa.get("type") == "search" and sa.get("keyword"):
                kw = sa["keyword"]
                break
        if not kw:
            kw = latest_search.get("context", {}).get("keyword", None)
        if kw:
            active_intent = f"正在寻找与\u201c{kw}\u201d相关的商品"
            intent_evidence.append(f"2小时内搜索了\u201c{kw}\u201d")
            intent_strength = 0.8

    pre_scene_events = [e for e in last_3h_events if e.get("type", "") != current_scene]
    if pre_scene_events:
        last_pre = pre_scene_events[-1]
        pre_scene_name = SCENE_NAMES.get(last_pre.get("type", ""), "其他")
        if _is_positive_action(last_pre):
            pre_scene_context = f"进入跨场景前，在[{pre_scene_name}]完成了积极互动"
        else:
            pre_scene_context = f"进入跨场景前，刚在[{pre_scene_name}]短暂停留"
    else:
        pre_scene_context = f"直接进入[{SCENE_NAMES.get(current_scene, current_scene)}]场景"

    session_start_ts = _parse_timestamp(session_events[0].get("timestamp", "")) if session_events else current_ts

    return L3ShortTermState(
        session_start_ts=session_start_ts,
        current_scene=current_scene,
        events_in_session=len(session_events),
        fatigue_level=fatigue,
        consecutive_skips=consecutive_skips,
        last_positive_event_ago_min=last_positive_ago,
        current_emotion=current_emotion,
        emotion_trigger=emotion_trigger,
        active_intent=active_intent,
        intent_evidence=intent_evidence,
        intent_strength=intent_strength,
        pre_scene_context=pre_scene_context,
        cross_scene_signal=intent_evidence[0] if intent_evidence else None
    )

def _emotion_desc(mean: float) -> str:
    if mean > 0.3: return "偏积极"
    elif mean > -0.1: return "较中性"
    elif mean > -0.3: return "略偏消极"
    else: return "明显偏消极"

def fuse_dynamic_persona(l1: L1BaselinePersona, l2_drifts: dict, l2_summary: str, l3: L3ShortTermState) -> str:
    sections = []

    current_scene = l3.current_scene
    scene_rate = l1.scene_stats.get(current_scene, {}).get("action_rate", 0.0)
    l1_section = (
        f"### 长期特征\n"
        f"- {l1.natural_language_summary}\n"
        f"- 长期情绪基线：{_emotion_desc(l1.emotion['mean'])}（均值{l1.emotion['mean']:.2f}），"
        f"负面行为率{l1.emotion['negative_ratio']:.0%}。\n"
        f"- 该用户历史{scene_rate:.1%}的概率在{current_scene}场景发生互动。"
    )
    sections.append(l1_section)

    if l2_summary and "基本一致" not in l2_summary:
        l2_section = f"### 近14天变化\n- {l2_summary}"
        if "emotion_mean" in l2_drifts:
            drift_info = l2_drifts["emotion_mean"]
            l2_section += (
                f"\n- 注意：该用户近期情绪状态相比长期基线"
                f"{'偏积极' if drift_info['delta_pct'] > 0 else '偏消极'}，"
                f"预测时请参考近期状态而非长期均值。"
            )
        sections.append(l2_section)

    l3_parts = []
    if l3.fatigue_level > 0.6:
        l3_parts.append(f"当前疲劳程度较高（疲劳指数{l3.fatigue_level:.2f}）：连续划走/跳过{l3.consecutive_skips}次。")
    elif l3.fatigue_level > 0.3:
        l3_parts.append(f"当前略有疲劳感（疲劳指数{l3.fatigue_level:.2f}），互动意愿有所下降。")
    else:
        l3_parts.append(f"当前状态较为活跃（疲劳指数{l3.fatigue_level:.2f}），互动意愿正常。")

    if l3.current_emotion != "neutral":
        emotion_desc = "积极" if l3.current_emotion == "positive" else "偏负面"
        trigger_str = f"（原因：{l3.emotion_trigger}）" if l3.emotion_trigger else ""
        l3_parts.append(f"当前情绪{emotion_desc}{trigger_str}。")

    if l3.active_intent and l3.intent_strength > 0.5:
        evidence_str = "；".join(l3.intent_evidence[:2])
        l3_parts.append(f"检测到明确即时意图：{l3.active_intent}（强度{l3.intent_strength:.1f}）。证据：{evidence_str}。")

    l3_parts.append(f"场景上下文：{l3.pre_scene_context}。")

    l3_section = "### 当前即时状态\n" + "\n".join([f"- {p}" for p in l3_parts])
    sections.append(l3_section)

    return "\n\n".join(sections)

def _format_behavior_sequence(sequence: List[Dict], current_ts: int, max_events: int = 30) -> str:
    history = sequence[-max_events:] if len(sequence) > max_events else sequence
    lines = []
    for i, e in enumerate(history, 1):
        ts = e.get("timestamp", "")
        t = e.get("type", "未知场景")
        ctx = format_action_context(e)
        res = format_action_result(e)
        lines.append(f"{i}. [{ts}] {t} | 内容: {ctx} | 动作: {res}")
    return "\n".join(lines) if lines else "无近期记录。"

def build_prediction_questions(action: Dict) -> List[Dict]:
    """Extract prediction subtasks from one behavior event."""
    questions = []
    action_type = action.get("type", "")
    action_list = action.get("action", [])
    if isinstance(action_list, dict):
        action_list = [action_list]

    if action_type == "视频浏览":
        context = action.get("context", {})
        duration_raw = context.get("duration", 0)
        try:
            if isinstance(duration_raw, (int, float)):
                video_duration = float(duration_raw)
            elif isinstance(duration_raw, str):
                video_duration = float(duration_raw.replace("秒", "").strip()) if duration_raw else 0
            else:
                video_duration = 0
        except:
            video_duration = 0

        watch_seconds = 0
        for act in action_list:
            if act.get("type") == "watch":
                pd = act.get("play_duration", 0)
                if isinstance(pd, str):
                    try: watch_seconds = float(pd.replace("秒", "").strip()) if pd else 0
                    except: watch_seconds = 0
                elif isinstance(pd, (int, float)):
                    watch_seconds = float(pd)
                break

        completed = False
        for act in action_list:
            if act.get("type") == "watch":
                if "is_complete_play" in act:
                    completed = act.get("is_complete_play", False)
                elif video_duration > 0:
                    completed = watch_seconds >= video_duration
                break

        questions.append({"type": "continuous", "field": "video_watch_seconds", "true_value": watch_seconds, "video_duration": video_duration})
        questions.append({"type": "binary", "field": "video_completed", "true_value": 1 if completed else 0})

        liked = commented = shared = collected = followed = False
        for act in action_list:
            t = act.get("type", "")
            if t == "like": liked = True
            elif t == "comment": commented = True
            elif t == "share": shared = True
            elif t == "collect": collected = True
            elif t == "follow": followed = True
        questions.append({"type": "binary", "field": "video_liked", "true_value": 1 if liked else 0})
        questions.append({"type": "binary", "field": "video_commented", "true_value": 1 if commented else 0})
        questions.append({"type": "binary", "field": "video_shared", "true_value": 1 if shared else 0})
        questions.append({"type": "binary", "field": "video_collected", "true_value": 1 if collected else 0})
        questions.append({"type": "binary", "field": "video_followed", "true_value": 1 if followed else 0})

    elif action_type == "商城购物":
        added = False
        for act in action_list:
            if act.get("type") == "cart" and act.get("is_add_to_cart"):
                added = True; break
        questions.append({"type": "binary", "field": "shop_added_to_cart", "true_value": 1 if added else 0})

        order_ok = False
        for act in action_list:
            if act.get("type") == "purchase":
                if act.get("is_pay") or act.get("paid") or act.get("order_success"):
                    order_ok = True; break
        questions.append({"type": "binary", "field": "shop_order_success", "true_value": 1 if order_ok else 0})

    elif action_type == "广告推荐":
        ws = 0
        for act in action_list:
            if act.get("type") == "watch":
                if "watch_seconds" in act: ws = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    pd = act.get("play_duration")
                    if isinstance(pd, str):
                        try: ws = float(pd.replace("秒", "").strip())
                        except: ws = 0
                    elif isinstance(pd, (int, float)): ws = float(pd)
                break
        if ws > 0 or any(a.get("type") == "watch" for a in action_list):
            questions.append({"type": "continuous", "field": "ad_watch_seconds", "true_value": ws})

        liked = commented = activated = form_submitted = False
        for act in action_list:
            t = act.get("type", "")
            if t == "like": liked = True
            elif t == "comment": commented = True
            elif t == "activation" and act.get("activation_cnt", 0) > 0: activated = True
            elif t == "submit" and act.get("form_submit_total_cnt", 0) > 0: form_submitted = True
        questions.append({"type": "binary", "field": "ad_liked", "true_value": 1 if liked else 0})
        questions.append({"type": "binary", "field": "ad_commented", "true_value": 1 if commented else 0})
        questions.append({"type": "binary", "field": "ad_activated", "true_value": 1 if activated else 0})
        questions.append({"type": "binary", "field": "ad_form_submitted", "true_value": 1 if form_submitted else 0})

    elif action_type == "直播间":
        ws = 0
        for act in action_list:
            if act.get("type") == "watch":
                pd = act.get("play_duration", 0)
                if isinstance(pd, str):
                    try: ws = float(pd.replace("秒", "").strip())
                    except: ws = 0
                elif isinstance(pd, (int, float)): ws = float(pd)
                break
        questions.append({"type": "continuous", "field": "live_watch_seconds", "true_value": ws})

        liked = commented = sent_gift = followed = shared = False
        for act in action_list:
            t = act.get("type", "")
            if t == "like": liked = True
            elif t == "comment": commented = True
            elif t == "send_gift": sent_gift = True
            elif t == "follow": followed = True
            elif t == "share": shared = True
        questions.append({"type": "binary", "field": "live_liked", "true_value": 1 if liked else 0})
        questions.append({"type": "binary", "field": "live_commented", "true_value": 1 if commented else 0})
        questions.append({"type": "binary", "field": "live_sent_gift", "true_value": 1 if sent_gift else 0})
        questions.append({"type": "binary", "field": "live_followed", "true_value": 1 if followed else 0})
        questions.append({"type": "binary", "field": "live_shared", "true_value": 1 if shared else 0})

        if action.get("context", {}).get("is_shop_live"):
            clicked_cart = False
            for act in action_list:
                if act.get("type") == "click_cart" and act.get("is_click_cart_action"):
                    clicked_cart = True; break
            questions.append({"type": "binary", "field": "live_clicked_cart", "true_value": 1 if clicked_cart else 0})

    elif action_type == "搜索行为":
        keyword = None
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword"); break
        if keyword:
            questions.append({"type": "text", "field": "search_keyword", "true_value": keyword})

    elif action_type == "电商客服对话":
        SKIP_PATTERNS = ["评价", "会话转移"]
        dialogue_content = []
        for act in action_list:
            if act.get("type") == "dialogue":
                dialogue_content = act.get("content", []); break

        if dialogue_content:
            target_msg = None; target_idx = -1
            for i in range(len(dialogue_content) - 1, -1, -1):
                msg = dialogue_content[i]
                if msg.get("role") != "user": continue
                c = msg.get("content", "")
                if not c or not c.strip(): continue
                if any(p in c for p in SKIP_PATTERNS): continue
                if len(c.strip()) <= 2: continue
                target_msg = c; target_idx = i; break

            if target_msg is not None:
                ctx_dialogue = []
                has_history = False
                for j in range(target_idx):
                    m = dialogue_content[j]
                    if m.get("role") in ["user", "assistant"] and m.get("content", "").strip():
                        has_history = True; ctx_dialogue.append(m)

                context_info = format_action_context(action)
                if not has_history:
                    q_text = (
                        f"这是一次电商客服对话场景。作为该用户，你即将开始与客服对话。\n\n"
                        f"### 订单/咨询背景信息\n{context_info}\n\n"
                        f"请代入你的身份，你想发给客服的第一句话是什么？"
                    )
                else:
                    lines = [f"{'你' if m.get('role')=='user' else '客服'}: {m.get('content', '')}" for m in ctx_dialogue[-20:]]
                    q_text = (
                        f"这是一段电商客服对话记录。\n\n"
                        f"### 订单/咨询背景信息\n{context_info}\n\n"
                        f"### 对话历史记录\n" + "\n".join(lines) + "\n\n"
                        f"请代入你的身份，结合上下文语境，你接下来会对客服说什么？"
                    )
                questions.append({
                    "type": "text", "field": "next_user_message", "true_value": target_msg,
                    "question": q_text, "has_dialogue_history": has_history,
                })

    return questions

def build_cot_prompt_with_dynamic_persona(
    l1: L1BaselinePersona, l2_drifts: dict, l2_summary: str, l3: L3ShortTermState,
    behavior_sequence: List[Dict], target_event: Dict, scene_global_rate: float,
    max_history_events: int = 30, question_info: Dict = None
) -> str:
    """Build a binary CoT prompt."""
    persona_txt = fuse_dynamic_persona(l1, l2_drifts, l2_summary, l3)
    current_ts = _parse_timestamp(target_event.get("timestamp", ""))
    seq_txt = _format_behavior_sequence(behavior_sequence, current_ts, max_events=max_history_events)
    content_str = format_action_context(target_event, is_target=True)
    scene = target_event.get("type", "unknown")
    user_scene_rate = l1.scene_stats.get(scene, {}).get("action_rate", scene_global_rate)

    act_type = ""
    if question_info:
        act_name = _YES_NO_QUESTIONS.get(question_info.get("field", ""), "是否执行该行为")
        act_type = FIELD_TO_ACT_TYPE.get(question_info.get("field", ""))
    else:
        act_name = "未知行为"
        if "action" in target_event and target_event["action"]:
            acts = target_event["action"]
            if isinstance(acts, dict): acts = [acts]
            act_name = TARGET_ACTION_NAMES.get(acts[0].get("type", ""), acts[0].get("type", ""))

    specific_rate_str = "无历史数据"
    if act_type:
        user_specific_rate = l1.scene_stats.get(scene, {}).get("specific_rates", {}).get(act_type, 0.0)
        specific_rate_str = f"{user_specific_rate:.1%}"

    hour = _get_hour_from_ts(current_ts)

    return render_prompt_template(
        "dynamic_binary_cot.txt",
        persona_txt=persona_txt,
        max_history_events=max_history_events,
        seq_txt=seq_txt,
        scene=scene,
        act_name=act_name,
        content_str=content_str,
        user_scene_rate=user_scene_rate,
        specific_rate_str=specific_rate_str,
        scene_global_rate=scene_global_rate,
        hour=hour,
        fatigue_level=l3.fatigue_level,
        current_emotion=l3.current_emotion,
        intent_strength=l3.intent_strength,
    )

def build_continuous_cot_prompt(
    l1: L1BaselinePersona, l2_drifts: dict, l2_summary: str, l3: L3ShortTermState,
    behavior_sequence: List[Dict], target_event: Dict, question_info: Dict,
    max_history_events: int = 30
) -> str:
    """Build a continuous-value CoT prompt."""
    persona_txt = fuse_dynamic_persona(l1, l2_drifts, l2_summary, l3)
    current_ts = _parse_timestamp(target_event.get("timestamp", ""))
    seq_txt = _format_behavior_sequence(behavior_sequence, current_ts, max_events=max_history_events)
    content_str = format_action_context(target_event, is_target=True)
    scene = target_event.get("type", "unknown")
    field = question_info.get("field", "")
    q_text = _CONTINUOUS_QUESTIONS.get(field, f"你预计该用户会在这里停留多少秒？（{field}）")
    hour = _get_hour_from_ts(current_ts)

    avg_watch = l1.scene_stats.get(scene, {}).get("avg_watch_seconds", 0.0)
    avg_watch_str = f"{avg_watch:.1f} 秒" if avg_watch > 0 else "无历史数据"

    return render_prompt_template(
        "dynamic_continuous_cot.txt",
        persona_txt=persona_txt,
        max_history_events=max_history_events,
        seq_txt=seq_txt,
        scene=scene,
        avg_watch_str=avg_watch_str,
        content_str=content_str,
        hour=hour,
        q_text=q_text,
        fatigue_level=l3.fatigue_level,
        current_emotion=l3.current_emotion,
    )

def build_text_cot_prompt(
    l1: L1BaselinePersona, l2_drifts: dict, l2_summary: str, l3: L3ShortTermState,
    behavior_sequence: List[Dict], target_event: Dict, question_info: Dict,
    max_history_events: int = 30
) -> str:
    """Build a text-generation CoT prompt."""
    persona_txt = fuse_dynamic_persona(l1, l2_drifts, l2_summary, l3)
    current_ts = _parse_timestamp(target_event.get("timestamp", ""))
    seq_txt = _format_behavior_sequence(behavior_sequence, current_ts, max_events=max_history_events)
    field = question_info.get("field", "")
    hour = _get_hour_from_ts(current_ts)

    if field == "search_keyword":
        return render_prompt_template(
            "dynamic_text_search_cot.txt",
            persona_txt=persona_txt,
            max_history_events=max_history_events,
            seq_txt=seq_txt,
            hour=hour,
        )

    elif field == "next_user_message":
        q_text = question_info.get("question", "")
        return render_prompt_template(
            "dynamic_text_customer_service_cot.txt",
            persona_txt=persona_txt,
            max_history_events=max_history_events,
            seq_txt=seq_txt,
            current_emotion=l3.current_emotion,
            q_text=q_text,
            hour=hour,
        )

    else:
        content_str = format_action_context(target_event, is_target=True)
        return render_prompt_template(
            "dynamic_text_generic_cot.txt",
            persona_txt=persona_txt,
            seq_txt=seq_txt,
            content_str=content_str,
        )

def build_cot_prompt_for_question(
    l1: L1BaselinePersona, l2_drifts: dict, l2_summary: str, l3: L3ShortTermState,
    behavior_sequence: List[Dict], target_event: Dict, question_info: Dict,
    scene_global_rate: float = 0.15, max_history_events: int = 30
) -> str:
    """Dispatch to the prompt builder for the question type."""
    q_type = question_info.get("type", "binary")

    if q_type == "binary":
        return build_cot_prompt_with_dynamic_persona(
            l1=l1, l2_drifts=l2_drifts, l2_summary=l2_summary, l3=l3,
            behavior_sequence=behavior_sequence, target_event=target_event,
            scene_global_rate=scene_global_rate, max_history_events=max_history_events,
            question_info=question_info
        )
    elif q_type == "continuous":
        return build_continuous_cot_prompt(
            l1=l1, l2_drifts=l2_drifts, l2_summary=l2_summary, l3=l3,
            behavior_sequence=behavior_sequence, target_event=target_event,
            question_info=question_info, max_history_events=max_history_events
        )
    elif q_type == "text":
        return build_text_cot_prompt(
            l1=l1, l2_drifts=l2_drifts, l2_summary=l2_summary, l3=l3,
            behavior_sequence=behavior_sequence, target_event=target_event,
            question_info=question_info, max_history_events=max_history_events
        )
    else:
        raise ValueError(f"Unknown question type: {q_type}")

def compute_dynamic_persona_for_training(all_events: List[Dict], target_event: Dict, l1_baseline: L1BaselinePersona, l2_window_days: int = 14) -> tuple:
    """Compute persona drift and short-term state for a target event."""
    cutoff_ts = _parse_timestamp(target_event.get("timestamp", ""))

    l2_window_start = cutoff_ts - l2_window_days * 24 * 3600000
    l2_events = [e for e in all_events if l2_window_start <= _parse_timestamp(e.get("timestamp", "")) < cutoff_ts]
    l2_drifts = compute_l2_trend(l2_events, l1_baseline, window_days=l2_window_days)
    l2_summary = generate_l2_summary_with_llm(l2_drifts, l1_baseline.natural_language_summary)

    l3_window_start = cutoff_ts - 3 * 3600000
    l3_events = [e for e in all_events if l3_window_start <= _parse_timestamp(e.get("timestamp", "")) < cutoff_ts]
    l3_state = compute_l3_state(l3_events, target_event)

    return l2_drifts, l2_summary, l3_state
