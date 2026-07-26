import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load standard environment variables
load_dotenv()

# We try to import the official google-genai library.
# If it's not installed or fails, we provide a mock/fallback so the app doesn't crash on start.
try:
    from google import genai
    from google.genai import types
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False

# Placeholder written by .env.example; treated as "no key configured".
PLACEHOLDER_KEY = "your_gemini_api_key_here"

# =====================================================================
# Pydantic Schemas for Structured Output
# =====================================================================

class QuestionSchema(BaseModel):
    chinese_sentence: str = Field(
        description="A natural daily life Chinese sentence that has a blank for the target English verb. Ensure it perfectly matches the given daily life scenario."
    )
    english_context: str = Field(
        description="A surrounding English dialog or context (1-2 sentences before or after the blank sentence) to provide rich conversational context."
    )
    blanked_sentence: str = Field(
        description="The target English sentence where the target verb is replaced by 6 underscores (e.g. 'Could you please ______ the window?'). Do NOT put letters in the blank itself."
    )
    correct_verbs: List[str] = Field(
        description="All acceptable English verbs in their exact correct form/tense for the blank. Include the base form and common inflections if applicable. E.g. ['close', 'shut']."
    )
    clue: str = Field(
        description="A subtle hint in Chinese helping the user find the verb (e.g., '这个词表示关闭，也可以指店铺停止营业')"
    )


class EvaluationSchema(BaseModel):
    is_correct: bool = Field(
        description="True if the user's answer is grammatically correct and contextually idiomatic for the blank, even if it's a synonym not in the initial correct_verbs list. False otherwise."
    )
    is_tense_error: bool = Field(
        description="True if the user chose the correct verb base but used the incorrect tense, conjugation, or inflection (e.g. wrote 'write' instead of 'written'). False otherwise."
    )
    recommended_verbs: List[str] = Field(
        description="A list of 1-3 best English verbs that fit this specific sentence, in their correct forms."
    )
    feedback: str = Field(
        description="A friendly, detailed evaluation in Chinese. Explain why the user's input is correct or incorrect, point out grammatical nuances (such as prepositions or tenses), and provide 1-2 examples of everyday collocations of the correct verb."
    )

def _regular_inflections(base: str) -> set:
    """
    Builds the plausible regular inflections of a base verb (-s / -ed / -ing).
    Used only by the offline grader to tell "wrong word" apart from "wrong form";
    when the Gemini API is configured, the model judges the form instead.
    """
    base = base.strip().lower()
    if not base:
        return set()

    forms = {base}
    if base.endswith("e"):
        forms.update({base + "d", base[:-1] + "ing", base + "s"})
    elif len(base) > 2 and base.endswith("y") and base[-2] not in "aeiou":
        forms.update({base[:-1] + "ied", base + "ing", base[:-1] + "ies"})
    elif base.endswith(("s", "x", "z", "ch", "sh")):
        forms.update({base + "ed", base + "ing", base + "es"})
    else:
        forms.update({base + "ed", base + "ing", base + "s"})
        # Doubled final consonant: consonant-vowel-consonant ending (plan -> planned)
        if (len(base) >= 3 and base[-1] not in "aeiouwxy"
                and base[-2] in "aeiou" and base[-3] not in "aeiou"):
            forms.update({base + base[-1] + "ed", base + base[-1] + "ing"})
    return forms


# =====================================================================
# Gemini Client Wrapper
# =====================================================================

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the Gemini Client.
        If api_key is not provided, it falls back to the GEMINI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = None

        if GEMINI_SDK_AVAILABLE and self.api_key:
            # Clean key from extra spaces/quotes
            clean_key = self.api_key.strip().replace('"', '').replace("'", "")
            if clean_key and clean_key != PLACEHOLDER_KEY:
                try:
                    self.client = genai.Client(api_key=clean_key)
                except Exception as e:
                    print(f"Error instantiating Gemini client: {e}")

    def is_configured(self) -> bool:
        """Returns True if the client is fully initialized and ready to make API calls."""
        return self.client is not None

    def generate_question(self, verb: str, definition: str, scenario: str) -> QuestionSchema:
        """
        Generates a testing question based on a target verb and scenario.
        Includes a robust fallback mock question generator in case of network or API key issues.
        """
        if not self.is_configured():
            return self._get_mock_question(verb, definition, scenario)

        prompt = f"""
        You are a professional ESL (English as a Second Language) teacher. Your goal is to generate an interactive, everyday practice question.

        Target English Verb to test: "{verb}" (definition: "{definition}").
        Lifestyle Scenario / Context: "{scenario}".

        Generate a realistic daily life situation that a native speaker would experience.
        Requirements:
        1. The blanked sentence MUST naturally require the verb "{verb}" or its inflections/tenses.
        2. Create a vivid conversation context or a specific scene to make the language feel alive.
        3. Make sure the difficulty matches daily, practical communication.
        4. Supply a comprehensive list of acceptable English verbs (in the correct tense) for this exact blank in 'correct_verbs'.
        5. Provide a helpful hint ('clue') in Chinese that guides the user to think of this verb.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuestionSchema,
                    temperature=0.9, # High temperature to keep things creative and non-repetitive
                ),
            )
            # The SDK automatically parses the JSON response into Pydantic if response_schema is provided.
            # However, we should be robust and verify we can return it.
            if response.parsed:
                return response.parsed

            # Fallback if parsed fails but we have text
            return QuestionSchema(**json.loads(response.text))

        except Exception as e:
            print(f"Gemini API Error in generate_question: {e}")
            return self._get_mock_question(verb, definition, scenario, error_msg=str(e))

    def evaluate_answer(self, question_data: dict, user_answer: str, target_verb: str) -> EvaluationSchema:
        """
        Evaluates the user's answer against the generated question using Gemini.
        If the client is not configured, performs a basic local string comparison.
        """
        if not self.is_configured():
            return self._get_mock_evaluation(question_data, user_answer, target_verb)

        prompt = f"""
        You are an encouraging and professional ESL teacher evaluating a student's answer.

        Testing Details:
        - Chinese Sentence: {question_data.get('chinese_sentence')}
        - English Context: {question_data.get('english_context')}
        - Blanked English Sentence: {question_data.get('blanked_sentence')}
        - Target English Verb (Expected): {target_verb}
        - Accepted Verbs List: {question_data.get('correct_verbs')}
        - Student's Answer: "{user_answer}"

        Your task:
        1. Evaluate if the student's answer "{user_answer}" is grammatically correct and contextually natural for the blank.
        2. VERY IMPORTANT: Do not just do a simple string match. If the student inputs a valid synonym (even if not in the accepted list) that a native English speaker would naturally say in this specific context, mark 'is_correct' as True!
        3. If the student used the correct verb base but got the tense or conjugation wrong (e.g., they wrote 'see' but the blank needs 'saw' or 'seen'), mark 'is_correct' as False, and mark 'is_tense_error' as True.
        4. Write friendly, detailed educational feedback in Chinese:
           - Congratulate them if they got it right, and explain why.
           - If it's a tense error, gently explain how to conjugate it correctly.
           - If it's incorrect, explain the nuance and provide 1-2 daily life collocations or examples of how to use the correct verb.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EvaluationSchema,
                    temperature=0.3, # Low temperature for accurate and consistent grading
                ),
            )
            if response.parsed:
                return response.parsed

            return EvaluationSchema(**json.loads(response.text))

        except Exception as e:
            print(f"Gemini API Error in evaluate_answer: {e}")
            return self._get_mock_evaluation(question_data, user_answer, target_verb, error_msg=str(e))

    # =====================================================================
    # Local Offline Fallbacks (Mock Mode)
    # =====================================================================

    def _get_mock_question(self, verb: str, definition: str, scenario: str, error_msg: str = "") -> QuestionSchema:
        """Local fallback question generator when Gemini is offline or API key is missing."""
        has_key_warning = " (⚠️ 提示：未检测到有效的 Gemini API Key，当前运行在离线模拟模式。)" if not error_msg else f" (⚠️ API调用出错：{error_msg}，已切换到本地模拟。)"

        # Simple hand-written variations for a few common verbs; everything else
        # falls back to the generic template below.
        if verb == "afford":
            blanked = "I really can't ______ to buy such a high-end laptop."
            chinese = f"【场景：{scenario}】请使用动词 '{verb}'（{definition}）来造句。比如你想说：我实在【{definition}】这么高端的笔记本电脑。{has_key_warning}"
            context = "I bought a new computer yesterday, but it is too expensive."
        elif verb == "borrow":
            blanked = "Can I ______ your umbrella for a second? It is pouring outside."
            chinese = f"【场景：{scenario}】请使用动词 '{verb}'（{definition}）来造句。比如询问朋友：我能【{definition}】一下你的雨伞吗？外面在下大雨。{has_key_warning}"
            context = "It started raining suddenly after work."
        elif verb == "apologize":
            blanked = "You should ______ to your sister for breaking her favorite mug."
            chinese = f"【场景：{scenario}】请使用动词 '{verb}'（{definition}）来造句。比如对朋友说：你应当为打破你妹妹最喜欢的马克杯而向她【{definition}】。{has_key_warning}"
            context = "My friend is arguing with his sister."
        else:
            # Generic template
            blanked = f"Please ______ this action carefully so we can proceed."
            chinese = f"【场景：{scenario}】请用动词 '{verb}'（{definition}）来填空。英文意思是：请仔细【{definition}】这个步骤，以便我们继续。{has_key_warning}"
            context = "We are going through the instructions step-by-step."

        return QuestionSchema(
            chinese_sentence=chinese,
            english_context=context,
            blanked_sentence=blanked,
            correct_verbs=[verb],
            clue=f"动词原形为 '{verb}'，中文意思是：{definition}。"
        )

    def _get_mock_evaluation(self, question_data: dict, user_answer: str, target_verb: str, error_msg: str = "") -> EvaluationSchema:
        """Local offline answer grader."""
        ans_clean = user_answer.strip().lower() if user_answer else ""
        expected_clean = target_verb.lower().strip()

        # Any verb the question declared acceptable counts as correct offline.
        accepted = {expected_clean}
        accepted.update(
            v.strip().lower() for v in question_data.get("correct_verbs") or [] if v
        )

        is_correct = ans_clean in accepted
        # Same word, wrong form: the answer is an inflection of an accepted verb.
        is_tense_error = not is_correct and any(
            ans_clean in _regular_inflections(v) for v in accepted
        )

        if is_correct:
            feedback = f"🎉 太棒了！回答正确。在当前语境中，使用 '{user_answer}' 是非常地道和合理的选择。"
        elif is_tense_error:
            feedback = f"⚠️ 动词选择正确，但是【时态/形式】不完全对。目标动词是 '{target_verb}'，在当前句子结构中需要注意正确的变形形式（如单三、过去式或分词形式）。"
        else:
            feedback = f"❌ 回答不准确。在这个句子中，最佳的动词是 '{target_verb}'（{question_data.get('clue', '')}）。\n你可以仔细阅读 Clue (提示) 或查看正确答案，然后再试一次！"

        if error_msg:
            feedback += f"\n\n*(注意：因API调用发生异常：'{error_msg}'，该结果由本地引擎离线评定)*"
        else:
            feedback += "\n\n*(注意：未配置 Gemini API Key，该结果由本地引擎离线评定)*"

        return EvaluationSchema(
            is_correct=is_correct,
            is_tense_error=is_tense_error,
            recommended_verbs=[target_verb],
            feedback=feedback
        )
