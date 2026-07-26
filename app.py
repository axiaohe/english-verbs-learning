import json
import os
import random
from datetime import datetime

import pandas as pd
import streamlit as st

# Import local modules
import db_manager
import llm_client
import vocab_pack_manager
from verbs_data import SCENARIOS_SEED

DIFFICULTY_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
FILTER_LEVELS = ["All"] + DIFFICULTY_LEVELS[:-1]
MODEL_OPTIONS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Characters that would otherwise be interpreted as Markdown or HTML. Question text
# comes from the model (or from words the user typed), so it is escaped before display.
_MARKDOWN_ESCAPES = str.maketrans({c: "\\" + c for c in "\\`*_[]<>|$"})


def md_escape(text) -> str:
    """Escapes Markdown/HTML control characters so text renders verbatim."""
    return str(text if text is not None else "").translate(_MARKDOWN_ESCAPES)


st.set_page_config(
    page_title="English Verb Training Agent",
    page_icon=":material/target:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
db_manager.init_db()

# =====================================================================
# Session State Initialization
# =====================================================================
env_model = os.getenv("GEMINI_MODEL", MODEL_OPTIONS[0])

st.session_state.setdefault("gemini_key", os.getenv("GEMINI_API_KEY", ""))
st.session_state.setdefault("model_name", env_model if env_model in MODEL_OPTIONS else MODEL_OPTIONS[0])
st.session_state.setdefault("current_question", None)
st.session_state.setdefault("current_verb", None)
st.session_state.setdefault("current_scenario", None)
st.session_state.setdefault("question_seq", 0)
st.session_state.setdefault("user_submitted", False)
st.session_state.setdefault("evaluation_result", None)
st.session_state.setdefault("show_target_verb", False)
st.session_state.setdefault("difficulty_filter", "All")
st.session_state.setdefault("starred_filter", False)
st.session_state.setdefault("pack_manager_selected_pack", None)
st.session_state.setdefault("ai_generated_preview", None)


def reset_question():
    """Drops the current question so the next run generates a fresh one."""
    st.session_state.current_question = None
    st.session_state.user_submitted = False
    st.session_state.evaluation_result = None
    st.session_state.show_target_verb = False


# =====================================================================
# API Client
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_client(api_key: str, model_name: str) -> llm_client.GeminiClient:
    """One client instance per (key, model) pair, reused across reruns."""
    return llm_client.GeminiClient(api_key=api_key, model_name=model_name)


# The .env.example placeholder is not a usable key
active_api_key = st.session_state.gemini_key
if active_api_key.strip() == llm_client.PLACEHOLDER_KEY:
    active_api_key = ""

client = get_client(active_api_key, st.session_state.model_name)

# =====================================================================
# Sidebar Configuration
# =====================================================================
with st.sidebar:
    st.subheader("练习配置", divider="gray")

    # Widgets are bound to session state by key; on_change clears the pending
    # question so a filter switch takes effect immediately.
    st.selectbox(
        "选择词汇难度 (CEFR Level)",
        options=FILTER_LEVELS,
        key="difficulty_filter",
        on_change=reset_question,
    )
    st.checkbox(
        "仅测试收藏夹中的单词",
        key="starred_filter",
        on_change=reset_question,
    )

    st.subheader("API 凭证管理", divider="gray")

    st.text_input(
        "Google Gemini API key",
        type="password",
        key="gemini_key",
        help="在 https://aistudio.google.com/ 获取免费的 API Key",
    )
    st.selectbox("选择模型级别 (Gemini model)", options=MODEL_OPTIONS, key="model_name")

    if client.is_configured():
        st.success("Gemini API 已连接", icon=":material/cloud_done:")
    else:
        st.warning("离线本地模拟模式 (未配置 API Key)", icon=":material/cloud_off:")

    st.divider()
    st.caption(
        "**小贴士**  \n"
        "1. 本智能代理使用 3 层随机锚点设计避免重复提问。  \n"
        "2. 支持同义词评估：只要动词在语境中自然，即使不是预设答案也判定正确。  \n"
        "3. 全键盘操作：输入答案后按 Enter 提交，看完解析再按一次 Enter 进入下一题。"
    )

    st.divider()
    st.subheader("词包状态", divider="gray")
    enabled_cnt = db_manager.get_enabled_pack_count()
    total_cnt = db_manager.get_total_pack_count()
    st.caption(f"已启用 {enabled_cnt}/{total_cnt} 个词包")
    if total_cnt > 0:
        st.caption(f"词汇总量约 {db_manager.get_vocab_stats()['total_verbs']} 个")

# =====================================================================
# Main Header
# =====================================================================
st.title(":material/target: 英语动词智能训练营")
st.caption("通过最真实的生活场景测试并追踪你的动词运用能力，不再死记硬背。")

tab_practice, tab_notebook, tab_analytics, tab_settings, tab_packs = st.tabs([
    ":material/school: 智能训练",
    ":material/menu_book: 单词备忘录",
    ":material/insights: 学习进度洞察",
    ":material/settings: 设置中心",
    ":material/package_2: 词汇包管理",
])

# =====================================================================
# TAB 1: PRACTICE MODE
# =====================================================================
with tab_practice:
    # 1. Fetch a question if we don't have one
    if st.session_state.current_question is None:
        diff_val = None if st.session_state.difficulty_filter == "All" else st.session_state.difficulty_filter

        test_verb_row = db_manager.get_verb_for_test(
            difficulty_filter=diff_val,
            starred_only=st.session_state.starred_filter,
        )

        if test_verb_row is None:
            if st.session_state.starred_filter:
                st.info("你的收藏夹中还没有该难度的单词。请在【单词备忘录】里收藏一些词汇，或关闭“仅测试收藏夹”。")
            else:
                st.info("没有找到符合当前过滤条件的单词，请在【单词备忘录】中添加词汇。")
        else:
            st.session_state.current_verb = test_verb_row["verb"]
            st.session_state.current_scenario = random.choice(SCENARIOS_SEED)["name"]

            with st.spinner("正在结合日常生活场景，为你量身定制题目…"):
                question_obj = client.generate_question(
                    verb=test_verb_row["verb"],
                    definition=test_verb_row["definition"],
                    scenario=st.session_state.current_scenario,
                )
            st.session_state.current_question = question_obj.model_dump()
            st.session_state.question_seq += 1
            st.session_state.user_submitted = False
            st.session_state.evaluation_result = None

    question = st.session_state.current_question

    if question:
        # A key that changes with every question, so each question gets a brand new,
        # empty input widget instead of us mutating an already-rendered widget's state.
        answer_key = f"answer_input_{st.session_state.question_seq}"

        col_main, col_side = st.columns([2, 1], gap="medium")

        with col_main:
            with st.container(border=True):
                st.caption(f"生活场景 · {md_escape(st.session_state.current_scenario)}")
                st.markdown("**中文语境（请用英文表达这句话）**")
                st.markdown(f"#### {md_escape(question['chinese_sentence'])}")

            with st.container(border=True):
                st.markdown("**上下文对话 (English context)**")
                st.markdown(f"*{md_escape(question['english_context'])}*")
                st.divider()
                st.markdown("**填空句子 (Fill in the blank)**")
                st.markdown(f"### :blue[{md_escape(question['blanked_sentence'])}]")

            submitted = st.session_state.user_submitted
            with st.form(key=f"answer_form_{st.session_state.question_seq}", border=False):
                user_ans = st.text_input(
                    "输入合适的英文动词（注意时态、单三及变形形式）",
                    placeholder="例如: afford / brought / running",
                    key=answer_key,
                    disabled=submitted,
                )
                submit_clicked = st.form_submit_button(
                    "提交并评估",
                    icon=":material/send:",
                    disabled=submitted,
                    type="primary",
                )

            answer = (user_ans or "").strip()
            if submit_clicked and not submitted and not answer:
                st.error("请输入动词答案后再提交。")

            if submit_clicked and not submitted and answer:
                st.session_state.user_submitted = True
                st.session_state.show_target_verb = True  # Auto-reveal so user can compare
                with st.spinner("智能助教正在评估你的答案与用法偏好…"):
                    eval_obj = client.evaluate_answer(
                        question_data=question,
                        user_answer=answer,
                        target_verb=st.session_state.current_verb,
                    )
                    st.session_state.evaluation_result = eval_obj.model_dump()

                    db_manager.update_progress(
                        verb=st.session_state.current_verb,
                        is_correct=eval_obj.is_correct,
                    )
                    db_manager.save_test_history(
                        verb=st.session_state.current_verb,
                        scenario=st.session_state.current_scenario,
                        chinese_sentence=question["chinese_sentence"],
                        expected_answer=st.session_state.current_verb,
                        user_answer=answer,
                        is_correct=eval_obj.is_correct,
                        feedback=eval_obj.feedback,
                    )
                st.rerun()

            # Evaluation results
            result = st.session_state.evaluation_result
            if st.session_state.user_submitted and result:
                if result["is_correct"]:
                    st.success("回答正确", icon=":material/check_circle:")
                elif result["is_tense_error"]:
                    st.warning("单词正确，但时态 / 形态有误", icon=":material/rule:")
                else:
                    st.error("回答错误", icon=":material/cancel:")

                with st.container(border=True):
                    st.markdown("**智能助教解析**")
                    st.markdown(md_escape(result["feedback"]))
                    if result["recommended_verbs"]:
                        st.markdown(
                            "**推荐搭配：** "
                            + " ".join(f"`{md_escape(v)}`" for v in result["recommended_verbs"])
                        )

                # Enter (focus is out of the disabled input) or Alt+N moves on.
                if st.button(
                    "下一题",
                    icon=":material/arrow_forward:",
                    type="primary",
                    width="stretch",
                    shortcut="Enter",
                    help="快捷键：Enter",
                ):
                    reset_question()
                    st.rerun()
            else:
                # Autofocus the answer box so the whole loop stays keyboard-only.
                st.html(
                    f"""
                    <script>
                    (function () {{
                        let tries = 0;
                        const timer = setInterval(function () {{
                            const el = document.querySelector('.st-key-{answer_key} input');
                            if (el && !el.disabled) {{ el.focus(); clearInterval(timer); }}
                            if (++tries > 20) {{ clearInterval(timer); }}
                        }}, 100);
                    }})();
                    </script>
                    """,
                    unsafe_allow_javascript=True,
                )

        with col_side:
            st.markdown("**辅助工具栏**")

            # Exact lookup: a LIKE search would match e.g. "forget" when testing "get".
            verb_row = db_manager.get_verb(st.session_state.current_verb)
            is_starred = bool(verb_row and verb_row["starred"] == 1)

            if st.button(
                "取消收藏该词" if is_starred else "收藏这个单词",
                icon=":material/star:" if is_starred else ":material/star_outline:",
                width="stretch",
            ):
                db_manager.toggle_star(st.session_state.current_verb)
                st.rerun()

            st.info(md_escape(question["clue"]), icon=":material/lightbulb:")

            with st.container(border=True):
                st.markdown("**当前测试词汇**")

                if st.session_state.show_target_verb:
                    # Revealed state — full details
                    st.markdown(f"- 目标动词：`{md_escape(st.session_state.current_verb)}`")
                    if verb_row:
                        st.markdown(f"- 基本释义：{md_escape(verb_row['definition'])}")
                        st.markdown(f"- 难度级别：{md_escape(verb_row['difficulty'])}")
                        st.markdown(f"- 历史尝试次数：{verb_row['attempts']} 次")
                        st.progress(
                            verb_row["mastery_score"] / 100,
                            text=f"熟练度 {verb_row['mastery_score']}/100",
                        )
                    if st.button("🙈 隐藏目标词汇", width="stretch"):
                        st.session_state.show_target_verb = False
                        st.rerun()
                else:
                    # Hidden state — mask the answer, show only stats
                    st.caption("🔒 目标词汇已隐藏")
                    if verb_row:
                        st.markdown(f"- 难度级别：{md_escape(verb_row['difficulty'])}")
                        st.markdown(f"- 历史尝试次数：{verb_row['attempts']} 次")
                        st.progress(
                            verb_row["mastery_score"] / 100,
                            text=f"熟练度 {verb_row['mastery_score']}/100",
                        )
                    if st.button("👁 显示目标词汇", width="stretch", type="primary"):
                        st.session_state.show_target_verb = True
                        st.rerun()

# =====================================================================
# TAB 2: VOCABULARY NOTEBOOK
# =====================================================================
with tab_notebook:
    st.subheader("个人单词库与备忘录")
    st.caption("这里收集了系统预置的常用动词，你也可以手动添加最近遇到的英语生词。")

    list_col, add_col = st.columns([3, 1], gap="medium")

    with list_col:
        sf1, sf2, sf3, sf4 = st.columns([2, 1, 1, 1])
        with sf1:
            search_input = st.text_input("搜索单词或中文释义", placeholder="输入词汇或中文释义…")
        with sf2:
            diff_filter_box = st.selectbox("按难度筛选", options=FILTER_LEVELS, key="notebook_diff_filter")
        with sf3:
            star_filter_box = st.selectbox(
                "按收藏状态筛选",
                options=["全部词汇", "仅显示收藏"],
                key="notebook_star_filter",
            )
        with sf4:
            packs = db_manager.get_all_packs()
            pack_options = ["全部词包"] + [f"{p['display_name']} ({p['verb_count']})" for p in packs]
            pack_filter_ui = st.selectbox("按词包筛选", options=pack_options, key="notebook_pack_filter")

        # Resolve pack filter selection
        selected_pack_filter = None
        if pack_filter_ui != "全部词包":
            # Extract the pack_name from the selection string "display_name (count)"
            selected_display = pack_filter_ui.rsplit(" (", 1)[0]
            for p in packs:
                if p["display_name"] == selected_display:
                    selected_pack_filter = p["pack_name"]
                    break

        verbs_list = db_manager.get_all_verbs(
            difficulty_filter=None if diff_filter_box == "All" else diff_filter_box,
            starred_only=star_filter_box == "仅显示收藏",
            search_query=search_input.strip() or None,
            pack_filter=selected_pack_filter,
        )

        if not verbs_list:
            st.warning("没有找到匹配的动词，试试其他筛选条件吧。")
        else:
            df = pd.DataFrame(verbs_list)
            # Map source_pack to display_name
            pack_name_to_display = {p["pack_name"]: p["display_name"] for p in packs}
            display_df = pd.DataFrame({
                "收藏": df["starred"] == 1,
                "单词": df["verb"],
                "难度": df["difficulty"],
                "中文释义": df["definition"],
                "来源词包": df["source_pack"].map(pack_name_to_display).fillna("自定义/原始"),
                "尝试次数": df["attempts"],
                "正确率": (df["correct_attempts"] / df["attempts"].where(df["attempts"] > 0)).fillna(0),
                "熟练度": df["mastery_score"],
            })

            table = st.dataframe(
                display_df,
                hide_index=True,
                height=460,
                key="vocab_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "收藏": st.column_config.CheckboxColumn(disabled=True, width="small"),
                    "正确率": st.column_config.NumberColumn(format="percent", width="small"),
                    "熟练度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                },
            )

            selected_rows = table.selection.rows
            if selected_rows:
                picked = verbs_list[selected_rows[0]]
                starred_now = picked["starred"] == 1
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"已选中 **{md_escape(picked['verb'])}** — {md_escape(picked['definition'])}")
                    if st.button(
                        "取消收藏" if starred_now else "加入收藏",
                        icon=":material/star:" if starred_now else ":material/star_outline:",
                    ):
                        db_manager.toggle_star(picked["verb"])
                        st.rerun()
            else:
                st.caption("提示：点选表格中的一行即可快速切换该词的收藏状态。")

            # Export button for filtered results
            if verbs_list:
                csv_data = vocab_pack_manager.verbs_to_csv_string(verbs_list, include_progress=True)
                st.download_button(
                    "导出当前筛选结果为 CSV",
                    data=csv_data,
                    file_name="vocabulary_export.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )

    with add_col:
        st.markdown("**添加新动词**")
        st.caption("学到新词汇？加进学习库，系统会自动为它生成练习题。")

        with st.form("add_verb_form", clear_on_submit=True):
            new_v = st.text_input("英文动词（原形）", placeholder="e.g. acquire")
            new_d = st.selectbox("难度级别", options=DIFFICULTY_LEVELS)
            new_def = st.text_area("中文含义 / 常用释义", placeholder="e.g. 获得，习得")
            add_submit = st.form_submit_button("保存到单词库", icon=":material/save:", width="stretch")

        if add_submit:
            if not new_v.strip() or not new_def.strip():
                st.error("单词名称和释义不能为空。")
            else:
                success, msg = db_manager.add_custom_verb(new_v.strip(), new_d, new_def.strip())
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# =====================================================================
# TAB 3: LEARNING ANALYTICS & HISTORY
# =====================================================================
with tab_analytics:
    st.subheader("学习数据洞察与历史归档")

    stats = db_manager.get_vocab_stats()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("已练习动词", f"{stats['practiced_verbs']}/{stats['total_verbs']}", border=True)
    m2.metric("重点收藏夹", stats["starred_verbs"], border=True)
    m3.metric("累积练习题数", stats["total_tests_taken"], border=True)
    m4.metric("整体回答正确率", f"{stats['accuracy_rate']}%", border=True)

    st.divider()

    chart_col, mistake_col = st.columns(2, gap="medium")

    with chart_col:
        st.markdown("**词汇熟练度分布**")
        mastery_df = pd.DataFrame({
            "熟练度阶段": ["精通 (>=80)", "中等 (40-79)", "起步 (1-39)", "尚未练习"],
            "单词数量": [
                stats["master_count"],
                stats["intermediate_count"],
                stats["beginner_count"],
                stats["unpracticed_count"],
            ],
        })
        st.bar_chart(mastery_df, x="熟练度阶段", y="单词数量", horizontal=True)

    with mistake_col:
        st.markdown("**重点攻坚：易错动词**")
        st.caption("练习过 2 次以上但正确率不高于 50% 的词，训练时请重点关注。")

        troubled = [
            {
                "单词": v["verb"],
                "释义": v["definition"],
                "练习次数": v["attempts"],
                "正确率": v["correct_attempts"] / v["attempts"],
                "熟练度": v["mastery_score"],
            }
            for v in db_manager.get_all_verbs()
            if v["attempts"] >= 2 and v["correct_attempts"] / v["attempts"] <= 0.5
        ]

        if not troubled:
            st.success("暂无易错单词，继续加油！", icon=":material/thumb_up:")
        else:
            troubled_df = pd.DataFrame(troubled).sort_values("熟练度")
            st.dataframe(
                troubled_df,
                hide_index=True,
                column_config={"正确率": st.column_config.NumberColumn(format="percent")},
            )

    st.divider()

    st.markdown("**历史测试档案**")
    history_logs = db_manager.get_test_history(limit=100)

    if not history_logs:
        st.info("尚无历史答题记录，去【智能训练】开始第一题吧。")
    else:
        hist_df = pd.DataFrame(history_logs)
        display_hist = pd.DataFrame({
            "时间": pd.to_datetime(hist_df["created_at"], errors="coerce"),
            "动词": hist_df["verb"],
            "生活场景": hist_df["scenario"],
            "中文句子": hist_df["chinese_sentence"],
            "你的回答": hist_df["user_answer"],
            "判定": hist_df["is_correct"] == 1,
            "智能解析反馈": hist_df["feedback"],
        })
        st.dataframe(
            display_hist,
            hide_index=True,
            height=400,
            column_config={
                "时间": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "判定": st.column_config.CheckboxColumn(disabled=True, width="small"),
            },
        )

# =====================================================================
# TAB 4: SETTINGS & MAINTENANCE
# =====================================================================
with tab_settings:
    st.subheader("设置与维护中心")

    conn_col, data_col = st.columns(2, gap="medium")

    with conn_col:
        st.markdown("**API 连接状态**")
        st.caption("API Key 与模型在左侧边栏配置，此处仅用于确认当前生效的设置。")

        with st.container(border=True):
            st.markdown(f"- 当前模型：`{st.session_state.model_name}`")
            if client.is_configured():
                st.success("Gemini 客户端已就绪，题目与批改均由模型生成。", icon=":material/cloud_done:")
            else:
                st.warning(
                    "未检测到有效的 API Key，应用运行在离线模拟模式：题目来自本地模板，"
                    "批改仅做词形比对。",
                    icon=":material/cloud_off:",
                )

    with data_col:
        st.markdown("**数据维护**")
        st.caption("清空练习状态、熟练度与答题历史，从零开始。收藏夹会被保留。")
        st.warning("此操作不可逆，将永久删除所有答题历史并把熟练度归零。", icon=":material/warning:")

        confirm_reset = st.checkbox("我已确认并理解该操作不可逆。")
        if st.button(
            "清空所有历史与熟练度",
            icon=":material/delete_forever:",
            width="stretch",
            disabled=not confirm_reset,
        ):
            db_manager.clear_all_history()
            reset_question()
            st.success("数据重置成功，你现在是一个初学者啦！")
            st.rerun()

# =====================================================================
# TAB 5: VOCABULARY PACK MANAGER
# =====================================================================
with tab_packs:
    st.subheader("词汇包管理")
    st.caption("启用或禁用内置词包、从文件导入词汇、或使用 AI 智能生成个性化词包。")

    # Refresh pack list
    all_packs = db_manager.get_all_packs()
    pack_name_to_display = {p["pack_name"]: p["display_name"] for p in all_packs}

    # ---- Pack Overview Cards ----
    st.markdown("**已安装的词包**")
    if not all_packs:
        st.info("尚无任何词包。请从下方导入或生成词包。")
    else:
        cols_per_row = 3
        for i in range(0, len(all_packs), cols_per_row):
            row_packs = all_packs[i : i + cols_per_row]
            card_cols = st.columns(cols_per_row, gap="medium")
            for j, pack in enumerate(row_packs):
                with card_cols[j]:
                    with st.container(border=True):
                        # Header row: name + tags
                        tag = ""
                        if pack["is_builtin"]:
                            tag = "内置"
                        elif pack["is_ai_generated"]:
                            tag = "AI 生成"
                        else:
                            tag = "自定义"

                        st.markdown(f"**{pack['display_name']}**  `{tag}`")
                        st.caption(pack.get("description", "")[:80] + "…" if len(pack.get("description", "")) > 80 else pack.get("description", ""))

                        st.markdown(f"分类: **{pack.get('category', '未分类')}**  |  {pack['verb_count']} 个动词")

                        # Enable/disable toggle
                        enabled = pack["is_enabled"] == 1
                        toggle_label = "已启用" if enabled else "已禁用"
                        if st.button(
                            toggle_label,
                            key=f"toggle_pack_{pack['pack_name']}",
                            icon=":material/toggle_on:" if enabled else ":material/toggle_off:",
                            width="stretch",
                        ):
                            db_manager.enable_pack(pack["pack_name"], not enabled)
                            st.rerun()

                        # Select for detail view
                        if st.button(
                            "查看详情",
                            key=f"detail_pack_{pack['pack_name']}",
                            icon=":material/info:",
                            width="stretch",
                        ):
                            st.session_state.pack_manager_selected_pack = pack["pack_name"]
                            st.rerun()

    # ---- Pack Detail (expanded when a pack is selected) ----
    selected_pack_name = st.session_state.pack_manager_selected_pack
    if selected_pack_name:
        selected_pack = next((p for p in all_packs if p["pack_name"] == selected_pack_name), None)
        if selected_pack:
            st.divider()
            st.markdown(f"### 词包详情: {selected_pack['display_name']}")

            detail_c1, detail_c2 = st.columns([2, 1], gap="medium")
            with detail_c1:
                st.markdown(f"**描述:** {selected_pack['description']}")
                st.markdown(f"**分类:** {selected_pack['category']}  |  **版本:** {selected_pack['version']}  |  **动词数:** {selected_pack['verb_count']}")

                # Verbs in this pack
                pack_verbs = db_manager.get_all_verbs(pack_filter=selected_pack_name)
                if pack_verbs:
                    pack_df = pd.DataFrame(pack_verbs)
                    st.dataframe(
                        pack_df[["verb", "difficulty", "definition"]],
                        hide_index=True,
                        height=300,
                        use_container_width=True,
                    )

            with detail_c2:
                # Export this pack
                pack_dict = db_manager.export_pack_to_dict(selected_pack_name)
                if pack_dict:
                    json_str = vocab_pack_manager.verbs_to_pack_json(
                        verbs=pack_dict["verbs"],
                        pack_name=pack_dict["pack_name"],
                        display_name=pack_dict["display_name"],
                        description=pack_dict["description"],
                        category=pack_dict["category"],
                    )
                    st.download_button(
                        "导出词包为 JSON",
                        data=json_str,
                        file_name=f"{selected_pack_name}.json",
                        mime="application/json",
                        icon=":material/download:",
                        width="stretch",
                    )
                    csv_str = vocab_pack_manager.verbs_to_csv_string(pack_dict["verbs"])
                    st.download_button(
                        "导出词包为 CSV",
                        data=csv_str,
                        file_name=f"{selected_pack_name}.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        width="stretch",
                    )

                # Delete non-builtin pack
                if not selected_pack["is_builtin"]:
                    st.markdown("---")
                    if st.button(
                        "删除此词包 (不可恢复)",
                        icon=":material/delete:",
                        width="stretch",
                        type="secondary",
                    ):
                        db_manager.delete_pack(selected_pack_name)
                        # Also delete the file
                        filename = selected_pack_name + ".json"
                        vocab_pack_manager.delete_pack_file(filename)
                        st.session_state.pack_manager_selected_pack = None
                        st.success(f"词包 '{selected_pack['display_name']}' 已删除。")
                        st.rerun()

            if st.button("收起详情", icon=":material/collapse_content:", width="stretch"):
                st.session_state.pack_manager_selected_pack = None
                st.rerun()

    # ---- Import / AI Generate Section ----
    st.divider()
    st.markdown("### 导入或生成词包")

    import_col, ai_col = st.columns(2, gap="large")

    with import_col:
        st.markdown("**从文件导入**")
        st.caption("支持 CSV / Excel / JSON 格式的词汇文件导入。")

        uploaded_file = st.file_uploader(
            "选择文件 (CSV / JSON / Excel)",
            type=["csv", "json", "xlsx"],
            key="pack_file_uploader",
            label_visibility="collapsed",
        )

        with st.form("import_pack_form", clear_on_submit=True):
            pack_name_input = st.text_input("词包名称", placeholder="e.g. 我的自定义词包")
            pack_cat_input = st.text_input("分类标签", placeholder="e.g. 日常口语 / 专业术语")
            import_submitted = st.form_submit_button(
                "导入词包", icon=":material/upload:", width="stretch"
            )

        if import_submitted and uploaded_file is not None:
            if not pack_name_input.strip():
                st.error("请输入词包名称。")
            else:
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name.lower()

                if filename.endswith(".json"):
                    # JSON pack import
                    try:
                        data = json.loads(file_bytes.decode("utf-8"))
                        ok, errors = vocab_pack_manager.validate_pack_structure(data)
                        if not ok:
                            st.error(f"JSON 格式错误: {'; '.join(errors)}")
                        else:
                            pack_name = data.get("pack_name", pack_name_input.strip())
                            display_name = data.get("display_name", pack_name_input.strip())
                            desc = data.get("description", "")
                            category = data.get("category", pack_cat_input.strip() or "自定义")

                            db_manager.register_pack(pack_name, display_name, desc, category, "1.0")
                            imported, skipped = db_manager.import_pack_verbs(pack_name, data["verbs"])
                            save_path = vocab_pack_manager.save_pack_to_file(data, f"{pack_name}.json")
                            st.success(f"导入完成：新增 {imported} 个动词，跳过 {skipped} 个已存在的动词。")
                            st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"无法解析 JSON 文件: {e}")
                elif filename.endswith(".xlsx"):
                    # XLSX / Excel import
                    pack_name = pack_name_input.strip().replace(" ", "_").lower() or "imported_vocabulary"
                    verbs, errors = vocab_pack_manager.parse_xlsx_verb_file(file_bytes)
                    if errors:
                        for err in errors[:5]:
                            st.warning(err)
                        if len(errors) > 5:
                            st.warning(f"…还有 {len(errors) - 5} 个错误未显示。")

                    if not verbs:
                        st.error("没有有效的动词数据可导入。")
                    else:
                        display_name = pack_name_input.strip() or "导入的词汇表"
                        category = pack_cat_input.strip() or "自定义"
                        pack_data = {
                            "pack_name": pack_name,
                            "display_name": display_name,
                            "description": f"从 Excel 文件导入的自定义词包 ({category})。",
                            "category": category,
                            "language": "en-zh",
                            "version": "1.0",
                            "author": "user",
                            "verbs": verbs,
                        }
                        save_path = vocab_pack_manager.save_pack_to_file(pack_data, f"{pack_name}.json")
                        db_manager.register_pack(pack_name, display_name, pack_data["description"], category, "1.0")
                        imported, skipped = db_manager.import_pack_verbs(pack_name, verbs)
                        st.success(f"导入完成：新增 {imported} 个动词，跳过 {skipped} 个已存在的动词。")
                        st.rerun()
                else:
                    # CSV import
                    pack_name = pack_name_input.strip().replace(" ", "_").lower()
                    verbs, errors = vocab_pack_manager.parse_csv_verb_file(file_bytes)
                    if errors:
                        for err in errors[:5]:
                            st.warning(err)
                        if len(errors) > 5:
                            st.warning(f"…还有 {len(errors) - 5} 个错误未显示。")

                    if not verbs:
                        st.error("没有有效的动词数据可导入。")
                    else:
                        display_name = pack_name_input.strip() or "自定义词包"
                        category = pack_cat_input.strip() or "自定义"
                        pack_data = {
                            "pack_name": pack_name,
                            "display_name": display_name,
                            "description": f"从文件导入的自定义词包 ({category})。",
                            "category": category,
                            "language": "en-zh",
                            "version": "1.0",
                            "author": "user",
                            "verbs": verbs,
                        }
                        save_path = vocab_pack_manager.save_pack_to_file(pack_data, f"{pack_name}.json")
                        db_manager.register_pack(pack_name, display_name, pack_data["description"], category, "1.0")
                        imported, skipped = db_manager.import_pack_verbs(pack_name, verbs)
                        st.success(f"导入完成：新增 {imported} 个动词，跳过 {skipped} 个已存在的动词。")
                        st.rerun()

    with ai_col:
        st.markdown("**AI 智能生成词包**")
        st.caption("输入一个主题，让 Gemini AI 自动为你生成分类词汇包。")

        with st.form("ai_generate_pack_form", clear_on_submit=False):
            ai_topic = st.text_input(
                "主题关键词",
                placeholder="e.g. 雅思写作高频动词 / 医疗英语 / 科技面试",
            )
            ai_count = st.slider("目标动词数量", min_value=30, max_value=200, value=80, step=10)
            ai_difficulties = st.multiselect(
                "参考难度等级（可多选，不选则自动分布）",
                options=DIFFICULTY_LEVELS,
                default=["B1", "B2"],
            )
            ai_submitted = st.form_submit_button(
                "AI 生成词包",
                icon=":material/auto_awesome:",
                width="stretch",
                disabled=not client.is_configured(),
                help="需要配置 Gemini API Key" if not client.is_configured() else "使用 Gemini AI 生成词包",
            )

        if ai_submitted and ai_topic.strip():
            with st.spinner(f"正在调用 Gemini 生成 '{ai_topic}' 词包（约需 10-30 秒）…"):
                result = vocab_pack_manager.generate_pack_with_ai(
                    client=client,
                    topic=ai_topic.strip(),
                    count=ai_count,
                    difficulty_prefs=ai_difficulties if ai_difficulties else None,
                )
            if result is None:
                st.error("AI 生成失败，请检查 API Key 或稍后重试。")
            else:
                st.session_state.ai_generated_preview = result
                st.rerun()

        # Show AI preview if available
        ai_preview = st.session_state.ai_generated_preview
        if ai_preview:
            verbs_preview = ai_preview.get("verbs", [])
            st.success(f"AI 已生成 {len(verbs_preview)} 个动词")
            st.markdown(f"**词包名称:** {ai_preview['display_name']}")
            st.markdown(f"**描述:** {ai_preview['description']}")
            preview_df = pd.DataFrame(verbs_preview)
            st.dataframe(
                preview_df[["verb", "difficulty", "definition"]],
                hide_index=True,
                height=200,
                use_container_width=True,
            )

            preview_c1, preview_c2 = st.columns(2)
            with preview_c1:
                if st.button("确认保存词包", icon=":material/save:", width="stretch", type="primary"):
                    pack_name = ai_preview["pack_name"]
                    save_path = vocab_pack_manager.save_pack_to_file(ai_preview, f"{pack_name}.json")
                    db_manager.register_pack(
                        pack_name,
                        ai_preview["display_name"],
                        ai_preview["description"],
                        ai_preview.get("category", "AI 生成"),
                        ai_preview.get("version", "1.0"),
                        is_ai_generated=True,
                    )
                    imported, skipped = db_manager.import_pack_verbs(pack_name, verbs_preview)
                    st.session_state.ai_generated_preview = None
                    st.success(f"词包 '{ai_preview['display_name']}' 已保存！新增 {imported} 个动词。")
                    st.rerun()
            with preview_c2:
                if st.button("放弃并重试", icon=":material/refresh:", width="stretch"):
                    st.session_state.ai_generated_preview = None
                    st.rerun()

    # ---- Export All Section ----
    st.divider()
    st.markdown("### 全局导出")
    export_c1, export_c2 = st.columns(2)
    with export_c1:
        all_verbs = db_manager.export_all_verbs_to_dict()
        csv_all = vocab_pack_manager.verbs_to_csv_string(all_verbs, include_progress=True)
        st.download_button(
            "导出全部词汇为 CSV (含进度)",
            data=csv_all,
            file_name=f"all_vocabulary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with export_c2:
        json_all = vocab_pack_manager.verbs_to_pack_json(
            verbs=all_verbs,
            pack_name="full_export",
            display_name="全部词汇导出",
            description="从个人单词库导出的完整词汇集合，含所有已启用词包的动词。",
            category="export",
        )
        st.download_button(
            "导出全部词汇为 JSON",
            data=json_all,
            file_name=f"all_vocabulary_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
