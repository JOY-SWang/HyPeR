import os
import re
from datetime import datetime

from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional


# ---------- helpers ----------
LETTER_RE = r"[ABCD]"
TAG_RE = r"<{tag}>\s*(.*?)\s*</{tag}>"

def _get_tag(text: str, tag: str) -> Tuple[Optional[str], Optional[re.Match]]:
    m = re.search(TAG_RE.format(tag=tag), text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else None, m)

def _count_tag(text: str, tag: str) -> int:
    return len(re.findall(TAG_RE.format(tag=tag), text, flags=re.DOTALL | re.IGNORECASE))

def _only_single_letter(s: str) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    return (t.upper() if re.fullmatch(LETTER_RE, t.strip(), flags=re.IGNORECASE) else None)

def _normalize(text: str) -> str:
    # 降噪：小写、去标点->空格、压缩空白
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _last_letter(text: str) -> Optional[str]:
    m = re.findall(LETTER_RE, text, flags=re.IGNORECASE)
    return m[-1].upper() if m else None

def _extract_declared_in_reasoning_or_summary(think_text: str) -> Optional[str]:
    m = re.findall(r"\(\s*([ABCD])\s*\)", think_text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"answer\s+is\s+([ABCD])\b", think_text, flags=re.IGNORECASE)
    if m2:
        return m2[-1].upper()
    return _last_letter(think_text)

def _extract_final_answer_letter_and_text(text: str) -> Tuple[Optional[str], str, Optional[str], str]:
    """
    返回:
      letter: 预测字母 (可能来自 FINAL_ANSWER/RESPONSE/fallback)
      channel: "FINAL_ANSWER"|"RESPONSE"|"fallback"
      inner: 若channel是 FINAL_ANSWER/RESPONSE，则为对应tag内原文；否则 None
      channel_text: 用于文本匹配的原文（FINAL_ANSWER优先，否则RESPONSE，否则全文）
    """
    fa, m_fa = _get_tag(text, "FINAL_ANSWER")
    if fa is not None:
        strict = _only_single_letter(fa)
        if strict:
            return strict, "FINAL_ANSWER", fa, fa
        loose = _last_letter(fa)
        if loose:
            return loose, "FINAL_ANSWER", fa, fa

    resp, m_r = _get_tag(text, "RESPONSE")
    if resp is not None:
        strict = _only_single_letter(resp)
        if strict:
            return strict, "RESPONSE", resp, resp
        loose = _last_letter(resp)
        if loose:
            return loose, "RESPONSE", resp, resp
        # 无字母时，文本匹配也需要使用 resp
        return None, "RESPONSE", resp, resp

    loose = _last_letter(text)
    return (loose, "fallback", None, text)

def _bgm_consistency(text: str) -> float:
    think, _ = _get_tag(text, "THINK")
    if not think:
        return 0.0
    desc, _ = _get_tag(think, "DESCRIPTION")
    reas, _ = _get_tag(think, "REASONING")
    combo = _normalize((desc or "") + " " + (reas or ""))
    triggers = ["bgm", "background music", "music"]
    causal = ["because", "therefore", "thus", "so that", "hence"]
    if any(t in combo for t in triggers) and any(c in combo for c in causal):
        return 0.0
    return 1.0

def _split_to_sentences(text: str) -> List[str]:
    # 简易句切：按 .?!/换行；保留长度>=4的片段
    parts = re.split(r"[\.!\?\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 4]

def _speaker_snippets(spk: str) -> List[str]:
    # 优先抓引号；否则抓每行冒号后的内容；仍为空则回退为整段行
    quotes = re.findall(r"\"([^\"]{4,})\"", spk)
    if quotes:
        return [q.strip() for q in quotes]
    lines = []
    for l in spk.splitlines():
        if ":" in l:
            seg = l.split(":", 1)[-1].strip()
            if len(seg) >= 4:
                lines.append(seg)
    if lines:
        return lines
    # 回退：拆成句子
    return _split_to_sentences(spk)

def _speaker_consistency_sentence_fuzzy(text: str) -> float:
    """句级模糊匹配比例：对每个 SPEAKER 片段，在 ASR 句子中找最大 ratio，取平均"""
    think, _ = _get_tag(text, "THINK")
    if not think:
        return 0.0
    asr, _ = _get_tag(think, "ASR")
    spk, _ = _get_tag(think, "SPEAKER")
    if not asr or not spk:
        return 0.0

    asr_sents = _split_to_sentences(asr)
    spk_snips = _speaker_snippets(spk)
    if not spk_snips or not asr_sents:
        # 无可比对内容，给0.5避免过严（可按需调）
        return 0.5

    def ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

    bests = []
    for s in spk_snips:
        best = 0.0
        for t in asr_sents:
            r = ratio(s, t)
            if r > best:
                best = r
        bests.append(best)

    # 直接返回平均相似度（0~1）
    return sum(bests) / max(1, len(bests))

def _consistency_reasoning_vs_response(text: str) -> float:
    think, _ = _get_tag(text, "THINK")
    if not think:
        return 0.0
    declared = _extract_declared_in_reasoning_or_summary(think)
    pred_letter, _, _, _ = _extract_final_answer_letter_and_text(text)
    if not declared or not pred_letter:
        return 0.0
    return 1.0 if declared == pred_letter else 0.0

def _length_reward_linear(text: str, K: int = 300) -> float:
    """
    长度奖励（新规则）：
    - 希望整体“回答”（整段 completion 文本）长度 ≥ 300 token；
    - 若 300 <= tokens <= 600 -> 1.0；
    - 若 tokens > 600 -> 线性衰减：r = max(0, 1 - (tokens - 600)/K)；
    - 若 tokens < 300 -> 线性上升：r = max(0, tokens / 300)；
    - 额外硬性约束：如果出现 <FINAL_ANSWER>...</FINAL_ANSWER>，则其后不得再输出任何非空白字符，违背则直接返回 0.0。

    参数:
        text: 完整生成文本（含 THINK/RESPONSE/FINAL_ANSWER 等）
        K: 超过 600 token 后的线性衰减斜率（越小衰减越快），默认 200
    返回:
        [0.0, 1.0] 的分数
    """
    import re

    # --- 1) 硬性约束：FINAL_ANSWER 之后不得有内容 ---
    # 找到 </FINAL_ANSWER> 的结束位置；若存在，检查其后是否仅为空白
    m_close = re.search(r"</\s*FINAL_ANSWER\s*>", text, flags=re.IGNORECASE)
    if m_close:
        tail = text[m_close.end():]
        if re.search(r"\S", tail):  # 存在非空白字符
            return 0.0

    # --- 2) 统计近似 token 数（对整段文本，以 \S+ 为 token 粗略估计）---
    tokens = len(re.findall(r"\S+", text))

    # --- 3) 分段线性函数 ---
    MIN_TOKENS = 300
    MAX_TOKENS = 800

    if tokens <= 0:
        return 0.0

    if tokens < MIN_TOKENS:
        # 0 -> 0, 300 -> 1.0 的线性上升
        return max(0.0, min(1.0, tokens / float(MIN_TOKENS)))

    if tokens <= MAX_TOKENS:
        # 300~600 给满分
        return 1.0

    # tokens > 600: 线性衰减
    over = tokens - MAX_TOKENS
    K = max(1, int(K))  # 防止除0
    return max(0.0, 1.0 - over / float(K))

def _parse_solution(sol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析 gold，如：
      "C bird family" -> ("C", "bird family")
      "B" -> ("B", None)
    """
    if not sol:
        return None, None
    s = sol.strip()
    m = re.match(r"^\s*([ABCD])\b(.*)$", s, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        text = m.group(2).strip()
        text = text if text else None
        return letter, text
    # 若没有字母，仅文本
    return None, s.strip()

def _text_contains_gold(pred_text: str, gold_text: str) -> bool:
    """
    文本匹配：gold_text 作为规范化子串在 pred_text 中出现即算匹配
    """
    if not pred_text or not gold_text:
        return False
    return _normalize(gold_text) in _normalize(pred_text)

# ---------- main ----------
def accuracy_reward(completions: List[List[Dict[str, Any]]],
                    solution: List[str],
                    **kwargs) -> List[Any]:
    """
    Kwargs:
        weights: dict，默认：
          {"acc":1.0, "bgm":0.2, "spk":0.2, "cons":0.5, "length":0.5}
        return_details: bool，是否返回分项
        len_decay_K: int，长度线性衰减的K（默认20）
    """
    weights = kwargs.get("weights", {"acc":1.0, "bgm":0.2, "spk":0.2, "cons":0.5, "length":0.5})
    return_details = kwargs.get("return_details", False)
    len_decay_K = int(kwargs.get("len_decay_K", 200))

    contents = [completion[0]["content"] for completion in completions]
    rewards_out: List[Any] = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for idx, (content, sol) in enumerate(zip(contents, solution)):
        print("="*20)
        print(sol)
        print(content)
        gold_letter, gold_text = _parse_solution(sol)

        # 分项
        bgm = _bgm_consistency(content)
        spk = _speaker_consistency_sentence_fuzzy(content)   # 新：句级模糊匹配
        cons = _consistency_reasoning_vs_response(content)
        
        

        # 预测字母 + 预测文本（用于与 gold_text 匹配）
        pred_letter, channel, inner_text, channel_text = _extract_final_answer_letter_and_text(content)

        # accuracy 判定（满足其一即 1.0）：
        # 1) 字母匹配
        acc_by_letter = (gold_letter is not None and pred_letter is not None and pred_letter == gold_letter)
        # 2) 文本包含匹配（如 gold_text="bird family"，预测文本含该短语）
        acc_by_text = (gold_text is not None and _text_contains_gold(channel_text, gold_text))

        acc = 1.0 if (acc_by_letter or acc_by_text) else 0.0
        if acc == 1.0:
            length = _length_reward_linear(content, K=len_decay_K)
        else:
            length = 0.0

        total = (
            weights.get("acc", 1.0)   * acc +
            weights.get("bgm", 0.1)   * bgm +
            weights.get("spk", 0.2)   * spk +
            weights.get("cons", 0.5)  * cons +
            weights.get("length", 0.2)* length
        )
        details = {
                "total": float(total),
                "sub": {
                    "accuracy": float(acc),
                
                    "bgm_consistency": float(bgm),
                    "speaker_consistency": float(spk),
                    "reasoning_response_consistency": float(cons),
                    "length": float(length),
                },
                "pred": {
                    "letter": pred_letter,
                    "channel": channel,
                    "channel_text": channel_text,
                    "strict_single_letter": bool(_only_single_letter(inner_text)) if inner_text is not None else False,
                },
                "gold": {
                    "letter": gold_letter,
                    "text": gold_text,
                },
                "meta": {
                    "idx": idx,
                    "timestamp": current_time
                }
            }
        # if return_details:
        #     rewards_out.append(details)
        # else:
        rewards_out.append(float(total))
            # print(details)

    return rewards_out

CAP = re.DOTALL | re.IGNORECASE

def _span(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=CAP)
    return (m.start(), m.end()) if m else None

def _count(text, tag):
    return len(re.findall(rf"<{tag}>.*?</{tag}>", text, flags=CAP))

def _inside(outer_span, inner_span):
    return outer_span and inner_span and (outer_span[0] <= inner_span[0] and inner_span[1] <= outer_span[1])


def format_reward2(completions, **kwargs):
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
        if not (_inside(sp_caption, sp_bgm) and _inside(sp_caption, sp_speaker) ):
        #  and _inside(sp_caption, sp_asr) and _inside(sp_caption, sp_desc)
            results.append(0.8); continue

        # 8) THINK 内部必须包含 PLANNING / CAPTION / REASONING / SUMMARY
        #    （这里只检验“出现在 THINK 区间内”，不强制先后顺序；若需要顺序，可加序关系）
        for sp_child in (sp_plan, sp_caption, sp_reason, sp_summary):
            if not _inside(sp_think, sp_child):
                results.append(0.0); break

        else:
            # 所有检查通过
            results.append(1.0)

    return results

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
        "THINK",  "CAPTION",
        
        "REASONING"
    ]

    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)


        # 1) 每个必需标签全局只出现 1 次
        if any(_count(text, t) > 2 for t in required_once_global):
            results.append(0.0); continue

        # 2) 可选标签至多一次
        if _count(text, "REFLECT") > 1 or _count(text, "FINAL_ANSWER") > 1:
            results.append(0.0); continue

        # 3) 抓全局 span
        sp_think   = _span(text, "THINK")
        sp_caption = _span(text, "CAPTION")
        sp_bgm     = _span(text, "BGM")
        sp_speaker = _span(text, "SPEAKER")
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
        if "<CAPTION>" not in text:
            results.append(0.0); continue
        if not (_inside(sp_caption, sp_bgm) and _inside(sp_caption, sp_speaker) ):
        #  and _inside(sp_caption, sp_asr) and _inside(sp_caption, sp_desc)
            results.append(0.8); continue

        # # 8) THINK 内部必须包含 PLANNING / CAPTION / REASONING / SUMMARY
        # #    （这里只检验“出现在 THINK 区间内”，不强制先后顺序；若需要顺序，可加序关系）
        # for sp_child in (sp_plan, sp_caption, sp_reason, sp_summary):
        #     if not _inside(sp_think, sp_child):
        #         results.append(0.0); break

        else:
            # 所有检查通过
            results.append(1.0)

    return results

if __name__ == "__main__":
    strr = '''<THINK>
<PLANNING>
The user wants to know the content of the audio clip without giving away what's actually being said. I must describe only the background sounds—ignoring the dialogue—while still ensuring the thinking remains hidden. I will acknowledge that the audio contains only ambient noise and provide “无法确定,” which aligns with the required abstention policy.
</PLANNING>
<CAPTION>
The audio is composed entirely of non-speech acoustic elements. It lacks any discernible speech content.
</CAPTION>
<REASONING>
The question asks about the audio's content. Since the audio is purely acoustic and does not contain any spoken words, the most appropriate response is to state that the audio consists solely of unidentifiable sounds.
</REASONING>
<SUMMARY>
The audio is composed entirely of non-speech acoustic elements, making it impossible to determine the content without further information.
</SUMMARY>
</THINK>
<RESPONSE> The audio is purely acoustic and does not contain any identifiable speech content. Therefore, it is impossible to determine the content of the audio. The answer is C.
</RESPONSE>'''
    completions = [[{'role': 'assistant',
                     'content': strr}]]
    solutions = ["C bird family"]

    print(accuracy_reward(completions, solutions))
    print(format_reward(completions))