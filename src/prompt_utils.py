from functools import lru_cache
from pathlib import Path
from typing import Dict
import re

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt_template(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


def render_prompt_template(filename: str, **values) -> str:
    return load_prompt_template(filename).format(**values)


def _safe_numeric(value):
    """Convert numeric-looking values while preserving non-numeric text."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except (ValueError, TypeError):
            pass
        match = re.match(r'^(\d+\.?\d*)', s)
        if match:
            num_str = match.group(1)
            try:
                if '.' in num_str:
                    return float(num_str)
                return int(num_str)
            except (ValueError, TypeError):
                pass
        return value
    return value

_NUMERIC_CONTEXT_FIELDS = {
    "duration", "fans_user_num",
    "show_cnt", "play_cnt", "complete_play_cnt",
    "like_cnt", "comment_cnt", "share_cnt", "collect_cnt", "download_cnt", "follow_cnt",
    "report_cnt", "reduce_similar_cnt",
    "item_price", "item_num_180d_sales", "item_num_good_assess",
    "item_num_180d_buyer", "item_num_180d_rebuyer", "item_num_180d_cart",
    "live_total_user_cnt", "live_total_view_cnt", "live_total_view_duration",
    "live_like_cnt", "live_comment_cnt",
}

def clean_repetitive_text(text: str) -> str:
    """Collapse noisy repeated spans in long text."""
    if not isinstance(text, str) or len(text) < 20:
        return text

    cleaned = re.sub(r'([^\w\u4e00-\u9fa5])\1{3,}', r'\1', text)

    cleaned = re.sub(r'(.{1,5}?)\1{4,}', r'\1\1', cleaned)
    return cleaned

def format_action_context(action: Dict, is_target: bool = False) -> str:
    """Format the context attached to one behavior event."""
    raw_context = action.get("context", {})

    context = {
        k: (_safe_numeric(v) if k in _NUMERIC_CONTEXT_FIELDS else v)
        for k, v in raw_context.items()
    }
    action_type = action.get("type", "")

    if action_type == "视频浏览":
        parts = []

        if "caption" in context and context["caption"]:
            parts.append(f"这是一个标题为《{context['caption']}》的视频")
        else:
            parts.append("这是一个视频")

        if "fans_user_num" in context and context["fans_user_num"] > 0:
            fans_num = context["fans_user_num"]
            is_verified = context.get("is_verified", False)
            verification_type = context.get("verification_type", "")

            if fans_num >= 10000:
                fans_str = f"{fans_num / 10000:.1f}万"
            else:
                fans_str = f"{fans_num}"

            author_info = f"作者拥有 {fans_str} 粉丝"
            if is_verified and verification_type:
                author_info += f"，并且通过了{verification_type}认证"
            elif is_verified:
                author_info += "，并且已通过平台认证"
            parts.append(author_info)
        elif context.get("is_verified", False):
            verification_type = context.get("verification_type", "")
            if verification_type:
                parts.append(f"作者已通过{verification_type}认证")
            else:
                parts.append("作者已通过平台认证")

        if "duration" in context and context["duration"]:
            duration_minutes = context['duration'] // 60
            duration_seconds = context['duration'] % 60
            if duration_minutes > 0:
                parts.append(f"视频时长为 {duration_minutes} 分 {duration_seconds} 秒")
            else:
                parts.append(f"视频时长为 {duration_seconds} 秒")

        popularity = []

        if "show_cnt" in context and context["show_cnt"] > 0:
            popularity.append(f"{context['show_cnt']}次曝光")

        if "play_cnt" in context and context["play_cnt"] > 0:
            popularity.append(f"{context['play_cnt']}次播放")

        if "complete_play_cnt" in context and context["complete_play_cnt"] > 0:
            popularity.append(f"{context['complete_play_cnt']}次完整播放")

        if "like_cnt" in context and context["like_cnt"] > 0:
            popularity.append(f"{context['like_cnt']}个赞")

        if "comment_cnt" in context and context["comment_cnt"] > 0:
            popularity.append(f"{context['comment_cnt']}条评论")

        if "share_cnt" in context and context["share_cnt"] > 0:
            popularity.append(f"{context['share_cnt']}次分享")

        if "collect_cnt" in context and context["collect_cnt"] > 0:
            popularity.append(f"{context['collect_cnt']}次收藏")

        if "download_cnt" in context and context["download_cnt"] > 0:
            popularity.append(f"{context['download_cnt']}次下载")
        if "follow_cnt" in context and context["follow_cnt"] > 0:
            popularity.append(f"{context['follow_cnt']}次关注")

        if popularity:
            parts.append("该视频目前收获了 " + "、".join(popularity))

        if "ocr" in context and context["ocr"]:
            ocr_text = clean_repetitive_text(context['ocr'])
            if len(ocr_text) > 1000:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text[:1000]}...")
            else:
                parts.append(f"视频画面中识别到的文字内容为：{ocr_text}")

        if "asr" in context and context["asr"]:
            asr_text = clean_repetitive_text(context['asr'])
            if len(asr_text) > 1000:
                parts.append(f"视频中的语音内容为：{asr_text[:1000]}...")
            else:
                parts.append(f"视频中的语音内容为：{asr_text}")

        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"发布时间是 {context['upload_timestamp']}")

        return "。".join(parts) + "。" if parts else "视频信息缺失。"

    elif action_type == "商城购物":
        parts = []

        if "item_title" in context and context["item_title"]:
            parts.append(f"这是一件商品，名称为{context['item_title']}")
        else:
            parts.append("这是一件商品")

        if "item_desc" in context and context["item_desc"]:
            desc = clean_repetitive_text(context["item_desc"])
            if len(desc) > 1000:
                parts.append(f"商品描述：{desc[:1000]}...")
            else:
                parts.append(f"商品描述：{desc}")

        if "product_source" in context and context["product_source"]:
            parts.append(f"商品来自{context['product_source']}")

        if "item_price" in context and context["item_price"]:
            price = context['item_price']
            if price > 0:
                parts.append(f"售价为 {price} 元")
            else:
                parts.append("售价未知")

        category_info = []
        if "category_level1_name" in context and context["category_level1_name"]:
            category_info.append(context["category_level1_name"])
        if "category_level2_name" in context and context["category_level2_name"]:
            category_info.append(context["category_level2_name"])
        if "category_level3_name" in context and context["category_level3_name"]:
            category_info.append(context["category_level3_name"])

        if category_info:
            parts.append(f"商品具体类别为：{'，'.join(category_info)}")

        industry_info = []
        if "industry_level1_name" in context and context["industry_level1_name"]:
            industry_info.append(context["industry_level1_name"])
        if "industry_level2_name" in context and context["industry_level2_name"]:
            industry_info.append(context["industry_level2_name"])
        if "industry_level3_name" in context and context["industry_level3_name"]:
            industry_info.append(context["industry_level3_name"])

        if industry_info:
            parts.append(f"所属行业为：{'，'.join(industry_info)}")

        sales_info = []
        if "item_num_180d_sales" in context and context["item_num_180d_sales"] > 0:
            sales = context["item_num_180d_sales"]
            if sales >= 10000:
                sales_info.append(f"近180天已售出 {sales/10000:.1f}万件")
            else:
                sales_info.append(f"近180天已售出 {sales} 件")

        if "item_num_good_assess" in context and context["item_num_good_assess"] > 0:
            assess = context["item_num_good_assess"]
            if assess >= 10000:
                sales_info.append(f"{assess/10000:.1f}万条好评")
            else:
                sales_info.append(f"{assess}条好评")

        if "item_num_180d_buyer" in context and context["item_num_180d_buyer"] > 0:
            buyer = context["item_num_180d_buyer"]
            if buyer >= 10000:
                sales_info.append(f"{buyer/10000:.1f}万人购买")
            else:
                sales_info.append(f"{buyer}人购买")

        if "item_num_180d_rebuyer" in context and context["item_num_180d_rebuyer"] > 0:
            rebuyer = context["item_num_180d_rebuyer"]
            if rebuyer >= 10000:
                sales_info.append(f"{rebuyer/10000:.1f}万人复购")
            else:
                sales_info.append(f"{rebuyer}人复购")

        if sales_info:
            parts.append("该商品" + "、".join(sales_info))

        if "item_num_180d_cart" in context and context["item_num_180d_cart"] > 0:
            cart_num = context["item_num_180d_cart"]
            if cart_num >= 10000:
                parts.append(f"近180天有 {cart_num/10000:.1f}万 人加入购物车")
            else:
                parts.append(f"近180天有 {cart_num} 人加入购物车")

        return "。".join(parts) + "。" if parts else "商品信息缺失。"

    elif action_type == "广告推荐":
        parts = []

        if "product" in context and context["product"]:
            product_info = f"这是一条推广{context['product']}的广告"
            if "industry_main" in context and context["industry_main"]:
                product_info += f"，属于{context['industry_main']}"
                if "industry_sub" in context and context["industry_sub"] and context["industry_sub"] != context.get("industry_main"):
                    product_info += f"行业中的{context['industry_sub']}品类"
                else:
                    product_info += "行业"
            parts.append(product_info)
        else:
            parts.append("这是一条广告")

        if "is_verified" in context and context["is_verified"]:
            if "verification_type" in context and context["verification_type"]:
                parts.append(f"广告主已认证，认证类型为{context['verification_type']}")
            else:
                parts.append("广告主已认证")

        if "fans_user_num" in context and context["fans_user_num"]:
            parts.append(f"广告主粉丝数为 {context['fans_user_num']}")
        elif "fans_user_num_op_range" in context and context["fans_user_num_op_range"]:
            parts.append(f"广告主粉丝数范围为{context['fans_user_num_op_range']}")

        if "caption" in context and context["caption"]:
            caption = clean_repetitive_text(context['caption'])
            if len(caption) > 500:
                parts.append(f"广告标题为：{caption[:500]}...")
            else:
                parts.append(f"广告标题为：{caption}")

        if "photo_type" in context and context["photo_type"]:
            parts.append(f"广告类型为{context['photo_type']}")

        if "duration" in context and context["duration"]:
            duration = context["duration"]
            if duration >= 60:
                minutes = duration // 60
                seconds = duration % 60
                if seconds > 0:
                    parts.append(f"广告时长为 {minutes} 分 {seconds} 秒")
                else:
                    parts.append(f"广告时长为 {minutes} 分钟")
            else:
                parts.append(f"广告时长为 {duration} 秒")

        if "upload_timestamp" in context and context["upload_timestamp"]:
            parts.append(f"广告上传时间为 {context['upload_timestamp']}")

        if "show_cnt" in context and context["show_cnt"] and context["show_cnt"] > 0:
            parts.append(f"该广告已经向用户曝光过 {context['show_cnt']} 次")

        if "play_cnt" in context and context["play_cnt"] and context["play_cnt"] > 0:
            parts.append(f"广告播放次数为 {context['play_cnt']}")

        if "complete_play_cnt" in context and context["complete_play_cnt"] and context["complete_play_cnt"] > 0:
            parts.append(f"广告被完整播放 {context['complete_play_cnt']} 次")

        interaction_parts = []
        if "like_cnt" in context and context["like_cnt"] and context["like_cnt"] > 0:
            interaction_parts.append(f"获得点赞 {context['like_cnt']} 次")
        if "comment_cnt" in context and context["comment_cnt"] and context["comment_cnt"] > 0:
            interaction_parts.append(f"获得评论 {context['comment_cnt']} 次")
        if "share_cnt" in context and context["share_cnt"] and context["share_cnt"] > 0:
            interaction_parts.append(f"被分享 {context['share_cnt']} 次")
        if "collect_cnt" in context and context["collect_cnt"] and context["collect_cnt"] > 0:
            interaction_parts.append(f"被收藏 {context['collect_cnt']} 次")
        if "download_cnt" in context and context["download_cnt"] and context["download_cnt"] > 0:
            interaction_parts.append(f"被下载 {context['download_cnt']} 次")
        if "follow_cnt" in context and context["follow_cnt"] and context["follow_cnt"] > 0:
            interaction_parts.append(f"被关注 {context['follow_cnt']} 次")
        if interaction_parts:
            parts.append(f"广告互动数据：{', '.join(interaction_parts)}")

        negative_parts = []
        if "report_cnt" in context and context["report_cnt"] and context["report_cnt"] > 0:
            negative_parts.append(f"被举报 {context['report_cnt']} 次")
        if "reduce_similar_cnt" in context and context["reduce_similar_cnt"] and context["reduce_similar_cnt"] > 0:
            negative_parts.append(f"减少类似推荐 {context['reduce_similar_cnt']} 次")
        if negative_parts:
            parts.append(f"广告负面反馈：{', '.join(negative_parts)}")

        if "ocr_text" in context and context["ocr_text"]:
            ocr_text = clean_repetitive_text(context['ocr_text'])
            if len(ocr_text) > 10000:
                parts.append(f"广告画面中的文字内容为：{ocr_text[:10000]}")
            else:
                parts.append(f"广告画面中的文字内容为：{ocr_text}")

        if "asr_text" in context and context["asr_text"]:
            asr_text = clean_repetitive_text(context['asr_text'])
            if len(asr_text) > 10000:
                parts.append(f"广告中的语音内容为：{asr_text[:10000]}")
            else:
                parts.append(f"广告中的语音内容为：{asr_text}")

        return "。".join(parts) + "。" if parts else "广告信息缺失。"

    elif action_type == "直播间":
        parts = []

        if "live_title" in context and context["live_title"]:
            live_intro = f"这是一个标题为{context['live_title']}的直播间"
        else:
            live_intro = "这是一个直播间"

        if "live_category" in context and context["live_category"]:
            live_intro += f"，直播类型是{context['live_category']} 场景"

        if "live_game_name" in context and context["live_game_name"]:
            live_intro += f"，正在直播《{context['live_game_name']}》"

        if "is_shop_live" in context:
            if context['is_shop_live']:
                live_intro += "，主播正在进行带货直播"
            else:
                live_intro += "，这是一场娱乐性的非带货直播"

        parts.append(live_intro)

        popularity_info = []

        if "live_total_user_cnt" in context and context["live_total_user_cnt"]:
            user_cnt = context['live_total_user_cnt']
            if user_cnt >= 10000:
                popularity_info.append(f"累计 {user_cnt/10000:.1f}万 人次观看")
            else:
                popularity_info.append(f"累计 {user_cnt} 人次观看")

        if "live_total_view_cnt" in context and context["live_total_view_cnt"]:
            view_cnt = context['live_total_view_cnt']
            if view_cnt >= 10000:
                popularity_info.append(f"总观看量达到 {view_cnt/10000:.1f}万")
            else:
                popularity_info.append(f"总观看量 {view_cnt}")

        if popularity_info:
            parts.append("该直播已有 " + "、".join(popularity_info))

        if "live_total_view_duration" in context and context["live_total_view_duration"]:
            duration_seconds = context['live_total_view_duration']
            duration_hours = duration_seconds / 3600
            if duration_hours >= 10000:
                parts.append(f"累计观看时长达到 {duration_hours/10000:.1f}万 小时")
            elif duration_hours >= 1:
                parts.append(f"累计观看时长 {duration_hours:.1f} 小时")
            else:
                duration_minutes = duration_seconds / 60
                parts.append(f"累计观看时长 {duration_minutes:.0f} 分钟")

        if "live_cover_content" in context and context["live_cover_content"]:
            parts.append(f"直播封面的内容是：{clean_repetitive_text(context['live_cover_content'])}")

        interaction_info = []
        if "live_like_cnt" in context and context["live_like_cnt"] > 0:
            like_cnt = context['live_like_cnt']
            if like_cnt >= 10000:
                interaction_info.append(f"{like_cnt/10000:.1f}万个赞")
            else:
                interaction_info.append(f"{like_cnt}个赞")

        if "live_comment_cnt" in context and context["live_comment_cnt"] > 0:
            comment_cnt = context['live_comment_cnt']
            if comment_cnt >= 10000:
                interaction_info.append(f"{comment_cnt/10000:.1f}万条评论")
            else:
                interaction_info.append(f"{comment_cnt}条评论")

        if interaction_info:
            parts.append("直播间目前有 " + "、".join(interaction_info))

        if "items" in context and context["items"] and len(context["items"]) > 0:
            items = context["items"]
            item_details = []
            for item in items:
                item_title = item.get("title", "")
                item_price = item.get("price", "")
                if item_title and item_price:
                    item_details.append(f"{item_title}售价 {item_price} 元")
                elif item_title:
                    item_details.append(f"{item_title}")
            if item_details:
                if len(item_details) == 1:
                    parts.append(f"主播正在推荐 1 件商品：{item_details[0]}")
                else:
                    parts.append(f"主播正在推荐 {len(item_details)} 件商品：{', '.join(item_details)}")

        return "。".join(parts) + "。" if parts else "直播信息缺失。"

    elif action_type == "搜索行为":
        parts = []

        keyword = None
        query_category = None

        action_list = action.get("action", [])
        for act in action_list:
            if act.get("type") == "search":
                keyword = act.get("keyword")
                query_category = act.get("query_category")
                break

        if not keyword and "keyword" in context:
            keyword = context["keyword"]
        if not query_category and "query_category" in context:
            query_category = context["query_category"]

        if keyword:
            if is_target:
                search_info = "用户打开了搜索框"
            else:
                search_info = f"用户在搜索框中输入了关键词 {keyword}"
            if query_category:
                if query_category == "查询型":
                    search_info += "，希望查找相关信息"
                elif query_category == "浏览型":
                    search_info += "，想要浏览相关内容"
                else:
                    search_info += f"，搜索意图是{query_category}"
            parts.append(search_info)
        else:
            parts.append("用户进行了搜索")

        return "。".join(parts) + "。" if parts else "搜索信息缺失。"

    elif action_type == "电商客服对话":
        parts = []

        if "ticket_category" in context and context["ticket_category"]:
            parts.append(f"这是一次{context['ticket_category']}类型的咨询")
        else:
            parts.append("这是一次客服对话")

        order_details = []
        if "product_name" in context and context["product_name"]:
            order_details.append(f"涉及商品为《{context['product_name']}》")

        if "product_category_info" in context and context["product_category_info"]:
            order_details.append(f"商品类别是{context['product_category_info']}")

        price_info = []
        if "item_price" in context and context["item_price"]:
            price_info.append(f"单价 {context['item_price']} 元")

        if "item_qty" in context and context["item_qty"]:
            qty = context['item_qty']
            try:
                if isinstance(qty, (int, float)):
                    if qty == int(qty):
                        price_info.append(f"购买数量 {int(qty)} 件")
                    else:
                        price_info.append(f"购买数量 {qty} 件")
                else:
                    price_info.append(f"购买数量 {qty} 件")
            except:
                price_info.append(f"购买数量 {qty} 件")

        if "express_fee" in context and context["express_fee"]:
            try:
                fee = float(context["express_fee"])
                if fee > 0:
                    price_info.append(f"运费 {fee} 元")
            except:
                pass

        if price_info:
            order_details.append("，".join(price_info))

        if order_details:
            parts.append("；".join(order_details))

        if "pay_order_time" in context and context["pay_order_time"]:
            parts.append(f"该订单的下单时间是 {context['pay_order_time']}")

        return "。".join(parts) + "。" if parts else "客服对话信息缺失。"

    return str(context)

def format_action_result(action: Dict) -> str:
    """Format the observed result of one behavior event."""
    action_type = action.get("type", "")
    action_list = action.get("action", [])
    results = []

    if action_type == "视频浏览":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":

                watch_details = []

                if "play_duration" in act:
                    play_duration = act["play_duration"]

                    if isinstance(play_duration, str):
                        watch_details.append(f"观看了 {play_duration}")

                    elif isinstance(play_duration, (int, float)):
                        watch_sec = play_duration
                        if watch_sec >= 60:
                            minutes = int(watch_sec // 60)
                            seconds = int(watch_sec % 60)
                            if seconds > 0:
                                watch_details.append(f"观看了 {minutes} 分 {seconds} 秒")
                            else:
                                watch_details.append(f"观看了 {minutes} 分钟")
                        else:

                            if watch_sec == int(watch_sec):
                                watch_details.append(f"观看了 {int(watch_sec)} 秒")
                            else:
                                watch_details.append(f"观看了 {watch_sec:.1f} 秒")
                elif "watch_seconds" in act:
                    watch_sec = act["watch_seconds"]
                    if watch_sec >= 60:
                        minutes = int(watch_sec // 60)
                        seconds = int(watch_sec % 60)
                        if seconds > 0:
                            results.append(f"观看了 {minutes} 分 {seconds} 秒")
                        else:
                            results.append(f"观看了 {minutes} 分钟")

                watch_features = []

                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')}")

                if act.get("is_fast_forward_play"):
                    watch_features.append("使用了快进播放操作")

                if act.get("is_backward_play"):
                    watch_features.append("进行了回退观看操作")

                if act.get("is_enlarge_play"):
                    watch_features.append("放大了视频画面操作")

                if act.get("completed") or act.get("is_complete_play"):
                    watch_features.append("完整看完了整个视频")

                if watch_features:
                    watch_details.append("，".join(watch_features))

                if watch_details:
                    results.append("、".join(watch_details))

            elif act_type == "like":

                results.append("对视频进行点赞操作")

            elif act_type == "comment":

                comment_detail = act.get("comment_detail_list", "")
                if comment_detail:
                    results.append(f"发表了评论：{comment_detail}")
                else:
                    results.append("发表了评论")

                if act.get("comment_stay_duration"):
                    results.append(f"在评论区停留了 {act.get('comment_stay_duration')}")
                if act.get("is_at_friend_in_comment"):
                    results.append("在评论区提及了好友")

            elif act_type == "share":

                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享给朋友，成功分享了{share_cnt}次")
                else:
                    results.append("分享了视频")

            elif act_type == "collect":

                collect_cnt = act.get("collect_cnt")
                if collect_cnt:
                    results.append(f"收藏了该视频，成功收藏了{collect_cnt}次")
                else:
                    results.append("收藏了该视频")

            elif act_type == "download":

                download_cnt = act.get("download_cnt")
                if download_cnt:
                    results.append(f"下载了该视频，成功下载了{download_cnt}次")
                else:
                    results.append("下载了该视频")

            elif act_type == "follow":

                results.append("关注了视频作者")

            elif act_type == "unfollow":

                if act.get("is_unfollow_action"):
                    results.append("取消关注了视频作者")

            elif act_type == "dislike":

                if act.get("reduce_similar_cnt") or act.get("reduce_simliar_cnt"):
                    results.append("选择了不感兴趣")

            elif act_type == "report":

                if act.get("report_cnt"):
                    results.append("进行了举报")

    elif action_type == "商城购物":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "cart":

                if act.get("is_add_to_cart"):
                    results.append("把商品加入了购物车")
                else:
                    results.append("浏览了商品但未加入购物车")

            elif act_type == "purchase":

                if act.get("order_success") or act.get("paid") or act.get("is_pay"):
                    results.append("成功下单购买")
                else:
                    results.append("未购买该商品")

    elif action_type == "广告推荐":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":

                watch_seconds = 0
                if "watch_seconds" in act:
                    watch_seconds = act.get("watch_seconds", 0)
                elif "play_duration" in act:
                    play_duration = act.get("play_duration")
                    if isinstance(play_duration, str):
                        try:
                            watch_seconds = float(play_duration.replace("秒", "").strip())
                        except:
                            watch_seconds = 0
                    elif isinstance(play_duration, (int, float)):
                        watch_seconds = float(play_duration)

                watch_features = []
                if watch_seconds > 0:
                    watch_features.append(f"观看了 {watch_seconds:.1f} 秒")

                if act.get("played_loop_cnt"):
                    watch_features.append(f"循环播放了 {act.get('played_loop_cnt')} 次")

                if act.get("is_complete_play"):
                    watch_features.append("看完了整个广告")

                if watch_features:
                    results.append("、".join(watch_features))

            elif act_type == "like":
                if act.get("like_cnt"):
                    results.append(f"为广告点赞 {act.get('like_cnt')} 次")
                else:
                    results.append("为广告点赞")

            elif act_type == "comment":

                results.append("在广告下发表了评论")

            elif act_type == "share":

                share_cnt = act.get("share_cnt")
                if share_cnt:
                    results.append(f"分享了广告 {share_cnt} 次")
                else:
                    results.append("分享了广告")

            elif act_type == "conversion":
                if act.get("conversion_cnt"):
                    results.append(f"点击了广告产生转化，共 {act.get('conversion_cnt')} 次")

            elif act_type == "activation":
                if act.get("activation_cnt"):
                    results.append(f"激活了应用或服务，共 {act.get('activation_cnt')} 次")

            elif act_type == "purchase":

                purchase_info = []
                if act.get("pay_cnt"):
                    purchase_info.append(f"支付了 {act.get('pay_cnt')} 次")
                if act.get("pay_purchase_amt"):
                    purchase_info.append(f"支付金额 {act.get('pay_purchase_amt')} 元")
                if purchase_info:
                    results.append("通过广告" + "、".join(purchase_info))

            elif act_type == "submit":

                if act.get("form_submit_total_cnt"):
                    results.append(f"提交了表单 {act.get('form_submit_total_cnt')} 次")

            elif act_type == "follow":
                results.append("关注了广告主")

            elif act_type == "unfollow":
                results.append("取消关注了广告主")

            elif act_type == "dislike":
                if act.get("report_cnt"):
                    results.append("举报了广告")

    elif action_type == "直播间":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":

                play_duration = act.get("play_duration", 0)
                if isinstance(play_duration, str):
                    try:
                        watch_seconds = float(play_duration.replace("秒", "").strip())
                    except:
                        watch_seconds = 0
                elif isinstance(play_duration, (int, float)):
                    watch_seconds = float(play_duration)
                else:
                    watch_seconds = 0

                watch_info = []
                if watch_seconds > 0:
                    if watch_seconds >= 60:
                        minutes = int(watch_seconds // 60)
                        seconds = int(watch_seconds % 60)
                        if seconds > 0:
                            watch_info.append(f"在直播间停留了 {minutes} 分 {seconds} 秒")
                        else:
                            watch_info.append(f"在直播间停留了 {minutes} 分钟")
                    else:
                        watch_info.append(f"在直播间停留了 {int(watch_seconds)} 秒")

                if act.get("play_count"):
                    watch_info.append(f"观看了 {act.get('play_count')} 次")

                if watch_info:
                    results.append("，".join(watch_info))

            elif act_type == "like":
                if act.get("like_cnt"):
                    like_cnt = act.get("like_cnt")
                    if like_cnt >= 10000:
                        results.append(f"为主播点赞 {like_cnt/10000:.1f}万 次")
                    else:
                        results.append(f"为主播点赞 {like_cnt} 次")
                else:
                    results.append("为主播点赞")

            elif act_type == "comment":
                comment_info = []
                if act.get("comment_cnt"):
                    cnt = act.get("comment_cnt")
                    comment_info.append(f"发送了 {cnt} 条弹幕")

                if act.get("comment_content"):
                    comments = act.get("comment_content")
                    if isinstance(comments, list) and len(comments) > 0:

                        display_comments = comments[:3]
                        comment_text = "、".join([f"「{c}」" for c in display_comments])
                        if len(comments) > 3:
                            comment_info.append(f"评论内容包括：{comment_text} 等")
                        else:
                            comment_info.append(f"评论内容：{comment_text}")

                if comment_info:
                    results.append("在直播间" + "，".join(comment_info))
                else:
                    results.append("在直播间发送了弹幕")

            elif act_type == "send_gift":
                if act.get("gift_amount"):
                    gift_amount = act.get("gift_amount")
                    results.append(f"给主播送了价值 {gift_amount} 元的礼物")

            elif act_type == "follow":
                if act.get("is_follow_action"):
                    results.append("关注了主播")

            elif act_type == "unfollow":
                if act.get("is_unfollow_action"):
                    results.append("取消关注了主播")

            elif act_type == "share":
                if act.get("share_cnt"):
                    results.append(f"分享了直播间 {act.get('share_cnt')} 次")

            elif act_type == "dislike":
                dislike_info = []
                if act.get("report_cnt"):
                    dislike_info.append("举报了直播间")
                if act.get("reduce_simliar_cnt"):
                    dislike_info.append("选择了不感兴趣")
                if dislike_info:
                    results.append("、".join(dislike_info))

            elif act_type == "click_cart":

                if act.get("is_click_cart_action"):
                    item_title = act.get("item_title", "")
                    item_price = act.get("item_price", "")
                    if item_title and item_price:
                        results.append(f"把商品 {item_title}（售价{item_price}元）加入了购物车")
                    elif item_title:
                        results.append(f"把商品 {item_title} 加入了购物车")
                    else:
                        results.append("点击了购物车")

            elif act_type == "join_group":
                if act.get("join_fans_group_cnt"):
                    cnt = act.get("join_fans_group_cnt")
                    results.append(f"加入了粉丝团（{cnt}次）")

    elif action_type == "搜索行为":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "search":

                pass

            elif act_type == "show":

                if act.get("show_cnt"):
                    results.append(f"系统展示了 {act.get('show_cnt')} 条搜索结果")

            elif act_type == "click":
                if act.get("click_cnt"):
                    results.append(f"点击了搜索结果 {act.get('click_cnt')} 次")
                else:
                    results.append("点击了搜索结果")

    elif action_type == "电商客服对话":
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "dialogue":

                if act.get("content"):
                    dialogue_list = act.get("content")
                    if isinstance(dialogue_list, list):
                        user_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "user")
                        assistant_msgs = sum(1 for msg in dialogue_list if msg.get("role") == "assistant")
                        results.append(f"与客服进行了对话（用户发送 {user_msgs} 条消息，客服回复 {assistant_msgs} 条）")

            elif act_type == "purchase":

                if act.get("paid"):
                    results.append("最终完成了支付")
                else:
                    results.append("未完成支付")

    else:
        for act in action_list:
            act_type = act.get("type", "")

            if act_type == "watch":
                if "play_duration" in act or "watch_seconds" in act:
                    results.append("观看了内容")
            elif act_type == "like":
                results.append("点赞")
            elif act_type == "comment":
                results.append("评论")
            elif act_type == "share":
                results.append("分享")
            elif act_type == "collect":
                results.append("收藏")
            elif act_type == "download":
                results.append("下载")
            elif act_type == "purchase":
                results.append("购买")
            elif act_type == "click":
                results.append("点击")

    return "，".join(results) if results else "仅浏览未进行其他操作"
