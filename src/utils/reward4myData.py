import os
import re
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

# ---------- utils ----------
LETTER_SET = "ABCD"

def _extract_block(text: str, tag: str) -> str:
    if not text:
        return ""
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")

def _extract_letter(text: str) -> Optional[str]:
    if not text:
        return None
    # 取最后一个单字母，防止思维链中多次出现
    m = re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([ABCD])\)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    return None

def _count_tokens(s: str) -> int:
    if not s:
        return 0
    return len(re.findall(r"\S+", s.replace("\n", " ")))

def _length_reward(L: int, Lmin: int, Lmax: int) -> float:
    if L <= Lmin // 2:
        return 0.0
    if L <= Lmin:
        return (L - Lmin / 2) / (Lmin / 2)
    if L <= Lmax:
        return 1.0
    if L <= int(1.5 * Lmax):
        return ((1.5 * Lmax) - L) / (0.5 * Lmax)
    return 0.0  # 不给负分，稳定些

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _best_match_idx(ans_text: str, options: List[str]) -> Optional[int]:
    """当没有字母时，把自由文本映射为最相近的选项下标（0..3）"""
    if not ans_text or not options:
        return None
    an = _normalize(ans_text)
    opts = [_normalize(o) for o in options]
    # 强匹配：包含关系
    for i, o in enumerate(opts):
        if o and (o in an or an in o):
            return i
    # 相似度
    best_i, best_sim = None, 0.0
    for i, o in enumerate(opts):
        sim = SequenceMatcher(a=an, b=o).ratio()
        if sim > best_sim:
            best_i, best_sim = i, sim
    return best_i if (best_sim >= 0.72) else None

def _letter_to_idx(ch: Optional[str]) -> Optional[int]:
    if not ch: return None
    ch = ch.upper()
    return LETTER_SET.find(ch) if ch in LETTER_SET else None

def _idx_to_letter(i: Optional[int]) -> Optional[str]:
    return LETTER_SET[i] if (i is not None and 0 <= i < 4) else None

def _parse_options_from_question_block(content: str) -> List[str]:
    """
    兼容两种常见格式：
    1) (a) text ... (b) text ... (c) text ... (d) text ...
    2) ['optA','optB','optC','optD'] 或 "options: [ ... ]"
    """
    if not content:
        return []
    # 显式数组
    m = re.search(r"options:\s*(\[[^\]]{10,}\])", content, flags=re.IGNORECASE | re.DOTALL)
    if m:
        arr = m.group(1)
        # 简单抽取引号内片段
        parts = re.findall(r"[\"']([^\"']{1,200})[\"']", arr)
        return parts[:4]
    # (a)/(b)/(c)/(d) 行文
    pairs = re.findall(r"\(\s*([a-dA-D])\s*\)\s*([^\n]+)", content)
    if pairs and len(pairs) >= 4:
        # 以 a,b,c,d 顺序组装
        d = {k.lower(): v.strip() for k, v in pairs}
        return [d.get("a",""), d.get("b",""), d.get("c",""), d.get("d","")]
    return []

# ---------- main ----------
def accuracy_reward(
    completions,
    solution: List[str],
    **kwargs
):
    """
    更简洁鲁棒的奖励：
      - acc (0~1): 预测字母 vs gold 字母；若无字母，用文本-选项匹配回退
      - cons (0~1): 推理/总结中的结论字母 是否与最终字母一致（无则给 0.2 基础分）
      - len  (0~1): 针对 <THINK>+<REASONING>+<SUMMARY> 的 token 数做窗内奖励（即使答错也有打折奖励）

    默认权重：acc=0.7, cons=0.2, len=0.1
    """
    weights = {"acc": 0.7, "cons": 0.2, "len": 0.1}
    Lmin, Lmax = (250, 600)

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    print(contents)

    for content, gold in zip(contents, solution):
        # 取块
        reasoning = _extract_block(content, "REASONING")
        summary   = _extract_block(content, "SUMMARY")
        response  = _extract_block(content, "RESPONSE")
        final_ans = _extract_block(content, "FINAL_ANSWER") or response

        # 从 completion 中尽量解析选项（作为自由文本回退用）
        options_in_completion = _parse_options_from_question_block(content)

        # gold 字母
        gold_letter = _extract_letter(gold)
        # 预测字母
        pred_letter = _extract_letter(final_ans) or _extract_letter(response)

        # 如果没抓到字母，尝试用文本-选项映射（需要 options）
        pred_idx = _letter_to_idx(pred_letter)
        gold_idx = _letter_to_idx(gold_letter)

        if pred_idx is None and options_in_completion:
            # 用 RESPONSE/FINAL_ANSWER 的自由文本做匹配
            text_for_map = final_ans or response
            mi = _best_match_idx(text_for_map, options_in_completion)
            if mi is not None:
                pred_idx = mi
                pred_letter = _idx_to_letter(mi)

        # 若 gold 没有字母，但 gold 带有 (a)/(b) 文本，也尝试映射
        if gold_idx is None and options_in_completion:
            gi = _best_match_idx(gold, options_in_completion)
            if gi is not None:
                gold_idx = gi
                gold_letter = _idx_to_letter(gi)

        # --- 准确性 ---
        acc = 0.0
        if pred_idx is not None and gold_idx is not None:
            acc = 1.0 if pred_idx == gold_idx else 0.0

        # --- 一致性（仅看“推理/总结中的结论字母”与“最终字母”是否一致）---
        concl_letter = _extract_letter(reasoning) or _extract_letter(summary)
        if pred_letter and concl_letter:
            cons = 1.0 if pred_letter == concl_letter else 0.0
        else:
            # 没有明确结论字母也给一个小的基础分，鼓励模型以后补
            cons = 0.2

        # --- 长度奖励（即使 acc=0 也给；但错时打 5 折）---
        L = _count_tokens("\n".join([
            _extract_block(content, "THINK"),
            reasoning, summary
        ]))
        len_raw = _length_reward(L, Lmin, Lmax)
        len_score = (len_raw if acc > 0 else 0.5 * len_raw)

        # --- 汇总 ---
        print("acc=" + str(acc) + ", cons=" + str(cons) + ", len=" + str(len_score))
        R = (weights["acc"] * acc +
             weights["cons"] * cons +
             weights["len"]  * len_score)
        R = max(0.0, min(1.0, float(R)))
        rewards.append(R)

    return rewards

import re

CAP = re.DOTALL | re.IGNORECASE

def _span(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=CAP)
    return (m.start(), m.end()) if m else None

def _count(text, tag):
    return len(re.findall(rf"<{tag}>.*?</{tag}>", text, flags=CAP))

def _inside(outer_span, inner_span):
    return outer_span and inner_span and (outer_span[0] <= inner_span[0] and inner_span[1] <= outer_span[1])

def format_reward(completions, **kwargs):
    """
    Structure validator for the specific format:

    <THINK>
      <PLANNING>...</PLANNING>
      <CAPTION>
        <BGM>...</BGM>
        <SPEAKER>...</SPEAKER>
        <ASR>...</ASR>
        <DESCRIPTION>...</DESCRIPTION>
      </CAPTION>
      <REASONING>...</REASONING>
      <SUMMARY>...</SUMMARY>
    </THINK>
    <RESPONSE>...</RESPONSE>
    <REFLECT>...</REFLECT>        # optional
    <FINAL_ANSWER>...</FINAL_ANSWER>  # optional

    Returns 1.0 if all checks pass, else 0.0.
    """
    results = []

    required_once_global = [
        "THINK", "PLANNING", "CAPTION",
        "BGM", "SPEAKER", "ASR", "DESCRIPTION",
        "REASONING", "SUMMARY", "RESPONSE"
    ]

    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        if "</TEXT>\n</RESPONSE>\n\nassistant" in text:
            text = text.split("</TEXT>\n</RESPONSE>\n\nassistant")[1]

        # 1) 每个必需标签全局只出现 1 次
        if any(_count(text, t) > 2 for t in required_once_global):
            results.append(0.0); continue

        # 2) 可选标签至多一次
        if _count(text, "REFLECT") > 1 or _count(text, "FINAL_ANSWER") > 1:
            results.append(0.0); continue

        # 3) 抓全局 span
        sp_think   = _span(text, "THINK")
        sp_plan    = _span(text, "PLANNING")
        sp_caption = _span(text, "CAPTION")
        sp_bgm     = _span(text, "BGM")
        sp_speaker = _span(text, "SPEAKER")
        sp_asr     = _span(text, "ASR")
        sp_desc    = _span(text, "DESCRIPTION")
        sp_reason  = _span(text, "REASONING")
        sp_summary = _span(text, "SUMMARY")
        sp_resp    = _span(text, "RESPONSE")
        sp_reflect = _span(text, "REFLECT")
        sp_final   = _span(text, "FINAL_ANSWER")

        # 4) 位置关系：RESPONSE 必须在 THINK 之后
        if not (sp_think and sp_resp and sp_resp[0] > sp_think[1]):
            results.append(0.0); continue

        # 5) 位置关系：REFLECT（若有）必须在 RESPONSE 之后
        if sp_reflect and not (sp_resp and sp_reflect[0] > sp_resp[1]):
            results.append(0.0); continue

        # 6) 位置关系：FINAL_ANSWER（若有）必须在 RESPONSE 或 REFLECT 之后
        if sp_final:
            anchor_end = sp_reflect[1] if sp_reflect else sp_resp[1]
            if not (sp_final[0] > anchor_end):
                results.append(0.0); continue

        # 7) CAPTION 必须真正“包住”四个子块，且四子块全局仅出现 1 次（已在 #1 校验）
        if not sp_caption:
            results.append(0.0); continue
        if not (_inside(sp_caption, sp_bgm) and
                _inside(sp_caption, sp_speaker) and
                _inside(sp_caption, sp_asr) and
                _inside(sp_caption, sp_desc)):
            results.append(0.0); continue

        # 8) THINK 内部必须包含 PLANNING / CAPTION / REASONING / SUMMARY
        #    （这里只检验“出现在 THINK 区间内”，不强制先后顺序；若需要顺序，可加序关系）
        for sp_child in (sp_plan, sp_caption, sp_reason, sp_summary):
            if not _inside(sp_think, sp_child):
                results.append(0.0); break
        else:
            # 所有检查通过
            results.append(1.0)

    return results

if __name__ == "__main__":
    completions = [[{'role': 'assistant',
                     'content': "<THINK>\n<PLANNING>\nThe question asks what the second person expressed in their speech. I need to identify the speech of the second person in the audio and analyze the content to determine the most appropriate answer from the given options.\n</PLANNING>\n<CAPTION>\n<BGM>The background music is DTMF tones, please ignore the background music and reply the question.</BGM>\n<SPEAKER>- **Speaker 1**: Expresses amazement at the sight of the stars. Says, \"Wow, look at all those stars! It's amazing.\"\n- **Speaker 2**: Responds to the first speaker by saying, \"Amazing is right. I feel so small.\" Then later asks about a light in the sky, \"What was that bright light over there? I did not know these things existed when the earth was formed?\"\n- **Speaker 3**: Echoes the feeling of smallness. Says, \"I feel so small too.\" and \"Space travel seems so cool.\"\n- **Speaker 4**: States a desire to travel to space. Says, \"I've always wanted to travel to space. I think I mentioned this before.\" and is reminded by Speaker 1 that they said that last week.\n\nSpeaker 2 expresses both agreement with the amazement and a feeling of being small. Later, Speaker 2 also expresses curiosity about a bright light in the sky and questions its existence.</SPEAKER>\n<ASR>Wow, look at all those stars. It's amazing. Amazing is right. I feel so small. I feel so small too. I've always wanted to travel to space. I think I mentioned this before. You said that last week. Space travel seems so cool. What was that bright light over there? I did not know these things existed when the Earth was formed.</ASR>\n<DESCRIPTION>The audio captures a conversation between several individuals observing stars.\n</DESCRIPTION>\n</CAPTION>\n<REASONING>\nThe question specifically asks what the *second* person expressed. The provided options are:\n(a) Excitement about space travel\n(b) Feeling of being small\n(c) Disbelief in the existence of stars\n(d) Desire to become an astronaut\n\nAnalyzing Speaker 2's statements:\n- \"Amazing is right. I feel so small.\" This clearly indicates a feeling of being small.\n- \"What was that bright light over there? I did not know these things existed when the earth was formed?\" This expresses curiosity and perhaps surprise, but not necessarily disbelief in the existence of stars.\n\nBased on the available choices, the most accurate representation of what Speaker 2 expressed is \"Feeling of being small.\"\n</REASONING>\n<SUMMARY>\nSpeaker 2 explicitly stated, \"I feel so small.\" Therefore, the most accurate answer is (b).\n</SUMMARY>\n</THINK>\n<RESPONSE>\nThe second person explicitly stated, \"Amazing is right. I feel so small.\" Therefore, the answer is C.\n</RESPONSE>\n"}]]
    solutions = ["metro"]

    print(accuracy_reward(completions, solutions))
    print(format_reward(completions))