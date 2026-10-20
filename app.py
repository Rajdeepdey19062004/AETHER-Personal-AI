"""Streamlit chat interface with separate conversations and scoped attachments."""

from pathlib import Path
from uuid import uuid4

import streamlit as st

from auth_store import (
    DB_PATH, account_status, authenticate, change_password, create_user, init_db,
    list_users, set_temporary_password, set_user_active, user_count,
)
from chat_store import load_chats, save_chats
from document_manager import (
    calculate_sha256,
    get_all_documents,
    index_uploaded_document,
    remove_document,
)
from document_store import index_files_exist, load_documents, load_embedding_backend
from evaluation import check_answer_quality
from health_check import run_health_check
from logger_config import get_logger
from rag_engine import generate_answer
from study_assistant import (
    evaluate_quiz,
    generate_notes,
    generate_quiz,
    generate_study_plan,
    save_generated_text,
)
from user_context import set_user_id

SUPPORTED_TYPES = ["pdf", "txt", "md", "docx"]
logger = get_logger("app")

st.set_page_config(page_title="AETHER Personal AI", page_icon="🧠", layout="wide")
init_db()

if "aether_user" not in st.session_state:
    st.title("🧠 AETHER Personal AI")
    existing_accounts = user_count()
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        st.caption("Use your AETHER username and password.")
        with st.form("aether_login_form"):
            login_username = st.text_input("Username", max_chars=64, key="login_username")
            login_password = st.text_input("Password", type="password", max_chars=128, key="login_password")
            login_submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if login_submitted:
            account = authenticate(login_username, login_password)
            if account:
                st.session_state.aether_user = account
                st.rerun()
            st.error("Username or password is incorrect.")
    with register_tab:
        if existing_accounts == 0:
            st.info("Create the first account to get started.")
        else:
            st.info("Create your own account. Your chats and files will be private to your account.")
        with st.form("aether_register_form"):
            register_username = st.text_input("Choose a username", max_chars=64, key="register_username")
            register_password = st.text_input("Choose a password (10+ characters)", type="password", max_chars=128, key="register_password")
            confirm_password = st.text_input("Confirm password", type="password", max_chars=128, key="confirm_password")
            register_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if register_submitted:
            if register_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    account = create_user(register_username, register_password)
                    st.session_state.aether_user = account
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
    st.stop()

signed_in_user = st.session_state.aether_user
status = account_status(signed_in_user["id"])
if not status or not status["is_active"]:
    del st.session_state["aether_user"]
    st.error("This account is inactive. Contact the administrator.")
    st.stop()
signed_in_user["must_change_password"] = bool(status["must_change_password"])
set_user_id(signed_in_user["id"])
if signed_in_user["must_change_password"]:
    st.title("Set a new password")
    st.info("An administrator reset your password. Enter the temporary password, then choose a new one.")
    with st.form("forced_password_change"):
        current_password = st.text_input("Temporary password", type="password")
        new_password = st.text_input("New password (10+ characters)", type="password")
        confirm_new_password = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Save new password", type="primary")
    if submitted:
        if new_password != confirm_new_password:
            st.error("The new passwords do not match.")
        else:
            try:
                change_password(signed_in_user["id"], current_password, new_password)
                st.session_state.aether_user["must_change_password"] = False
                st.success("Password updated.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    st.stop()
st.title("🧠 AETHER — Personal AI")
st.caption(f"Signed in as {signed_in_user['username']} · Your chats and files are private to your account.")


def new_chat() -> dict:
    return {"id": uuid4().hex, "title": "New chat", "messages": [], "document_ids": []}


def sync_chat_attachments(chat_id: str, widget_key: str) -> None:
    chat = next((item for item in st.session_state.chats if item["id"] == chat_id), None)
    if chat is not None:
        chat["document_ids"] = list(st.session_state.get(widget_key, []))
        save_chats(st.session_state.chats)


def attach_uploaded_files(chat_id: str, widget_key: str) -> None:
    """Index newly selected uploads and attach them to the active conversation."""
    chat = next((item for item in st.session_state.chats if item["id"] == chat_id), None)
    if chat is None:
        return

    uploaded = st.session_state.get(widget_key) or []
    processed_by_chat = st.session_state.setdefault("processed_uploads", {})
    processed = set(processed_by_chat.get(chat_id, []))
    notices = []
    for uploaded_file in uploaded:
        content = uploaded_file.getvalue()
        digest = calculate_sha256(content)
        if digest in processed:
            continue
        safe_name = Path(uploaded_file.name.replace("\\", "/")).name
        try:
            result = index_uploaded_document(safe_name, content)
            document_id = result.get("document_id")
            if document_id and document_id not in chat["document_ids"]:
                chat["document_ids"].append(document_id)
            notices.append(f"Attached {safe_name} ({result['status']}).")
            processed.add(digest)
        except Exception as error:
            logger.exception("Document upload/indexing failed for %s", safe_name)
            notices.append(f"Could not attach {safe_name}: {error}")

    processed_by_chat[chat_id] = list(processed)
    attachment_widget_key = f"attached-docs-{chat_id}"
    if attachment_widget_key in st.session_state:
        st.session_state[attachment_widget_key] = list(chat["document_ids"])
    st.session_state["upload_notices"] = notices
    save_chats(st.session_state.chats)


if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if not st.session_state.chats:
    st.session_state.chats = [new_chat()]
if "active_chat_id" not in st.session_state or not any(
    chat["id"] == st.session_state.active_chat_id for chat in st.session_state.chats
):
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]

chats = st.session_state.chats
active_chat = next(chat for chat in chats if chat["id"] == st.session_state.active_chat_id)

with st.sidebar:
    st.caption(f"Account: {signed_in_user['username']}")
    if st.button("Sign out", use_container_width=True):
        del st.session_state["aether_user"]
        st.rerun()
    if signed_in_user.get("is_admin"):
        with st.expander("Admin portal"):
            st.caption("Manage accounts. Passwords are never shown; reset creates a temporary password.")
            try:
                accounts = list_users(signed_in_user["id"])
                st.dataframe(
                    [{"Username": row["username"], "Role": "Admin" if row["is_admin"] else "User",
                      "Status": "Active" if row["is_active"] else "Disabled", "Created": row["created_at"]}
                     for row in accounts],
                    hide_index=True, use_container_width=True,
                )
                regular_accounts = [row for row in accounts if not row["is_admin"]]
                if regular_accounts:
                    selected_id = st.selectbox(
                        "Select user", [row["id"] for row in regular_accounts],
                        format_func=lambda user_id: next(row["username"] for row in regular_accounts if row["id"] == user_id),
                    )
                    selected_user = next(row for row in regular_accounts if row["id"] == selected_id)
                    status_col, reset_col = st.columns(2)
                    with status_col:
                        action = "Disable" if selected_user["is_active"] else "Enable"
                        if st.button(action, key="admin_toggle_user", use_container_width=True):
                            set_user_active(signed_in_user["id"], selected_id, not selected_user["is_active"])
                            st.rerun()
                    with reset_col:
                        if st.button("Reset password", key="admin_reset_password", use_container_width=True):
                            st.session_state.admin_temp_password = {
                                "username": selected_user["username"],
                                "password": set_temporary_password(signed_in_user["id"], selected_id),
                            }
                    temporary = st.session_state.get("admin_temp_password")
                    if temporary:
                        st.warning(f"Temporary password for {temporary['username']} (visible in this admin session):")
                        st.code(temporary["password"])
                        st.caption("Share it with the user securely. They must change it at their next login.")
                        if st.button("Hide temporary password", key="hide_admin_temp_password"):
                            del st.session_state["admin_temp_password"]
                            st.rerun()
                else:
                    st.caption("No regular user accounts yet.")
            except (PermissionError, ValueError) as error:
                st.error(str(error))
    with st.expander("Account database"):
        st.caption(f"Database file: {DB_PATH}")
        st.caption("Passwords are saved as salted scrypt hashes in the password_hash column, not as readable passwords.")
        st.caption("The database also stores account records and chat history. Uploaded files and search indexes are stored under data/users/ in a separate folder per account.")
    reasoning_mode = st.radio(
        "AI reasoning",
        ["Fast", "Smart Thinking"],
        help="Fast uses the installed Qwen2.5 3B model. Smart Thinking uses Qwen3 4B with deliberate reasoning and a larger answer budget.",
    )
    smart_thinking = reasoning_mode == "Smart Thinking"
    if smart_thinking:
        st.caption("Smart Thinking is more capable for complex tasks, but may take longer on this computer.")
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        chat = new_chat()
        chats.insert(0, chat)
        st.session_state.active_chat_id = chat["id"]
        save_chats(chats)
        st.rerun()

    for chat in chats:
        selected = chat["id"] == active_chat["id"]
        if st.button(
            ("● " if selected else "") + chat.get("title", "New chat"),
            key=f"open-chat-{chat['id']}",
            use_container_width=True,
        ):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()

    delete_col, clear_col = st.columns(2)
    with delete_col:
        if st.button("Delete chat", use_container_width=True):
            chats.remove(active_chat)
            if not chats:
                chats.append(new_chat())
            st.session_state.active_chat_id = chats[0]["id"]
            save_chats(chats)
            st.rerun()
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            active_chat["messages"] = []
            active_chat["title"] = "New chat"
            save_chats(chats)
            st.rerun()

    st.divider()
    st.header("Files for this chat")
    uploaded_files = st.file_uploader(
        "Upload a file to attach it to this chat",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"uploads-{active_chat['id']}",
        on_change=attach_uploaded_files,
        args=(active_chat["id"], f"uploads-{active_chat['id']}"),
    )
    for notice in st.session_state.pop("upload_notices", []):
        if notice.startswith("Could not"):
            st.error(notice)
        else:
            st.success(notice)
    if uploaded_files:
        st.caption("Files attach automatically when upload finishes. Remove them under Attached documents.")

    try:
        indexed_documents = get_all_documents()
    except Exception as error:
        indexed_documents = []
        st.error(f"Could not load indexed documents: {error}")

    documents_by_id = {doc["document_id"]: doc for doc in indexed_documents}
    active_chat["document_ids"] = [
        doc_id for doc_id in active_chat.get("document_ids", []) if doc_id in documents_by_id
    ]
    if indexed_documents:
        attachment_widget_key = f"attached-docs-{active_chat['id']}"
        selected_docs = st.multiselect(
            "Attached documents",
            options=list(documents_by_id),
            default=active_chat["document_ids"],
            format_func=lambda doc_id: documents_by_id[doc_id].get("filename", doc_id),
            key=attachment_widget_key,
            on_change=sync_chat_attachments,
            args=(active_chat["id"], attachment_widget_key),
            help="Only these documents can provide evidence in this chat. Remove one here to detach it.",
        )
        active_chat["document_ids"] = selected_docs
    else:
        st.caption("No indexed files yet. General questions still work.")

    with st.expander("Document library"):
        if indexed_documents:
            for document in indexed_documents:
                filename = document.get("filename", "Unnamed document")
                st.caption(f"{filename} · {document.get('chunk_count', 0)} chunks")
                confirm = st.checkbox(
                    f"Confirm permanent deletion: {filename}",
                    key=f"confirm-delete-{document['document_id']}",
                )
                if confirm and st.button(
                    f"Delete file and index", key=f"delete-{document['document_id']}"
                ):
                    try:
                        remove_document(document["document_id"], delete_file=True)
                        for chat in chats:
                            chat["document_ids"] = [
                                item for item in chat.get("document_ids", [])
                                if item != document["document_id"]
                            ]
                            widget_key = f"attached-docs-{chat['id']}"
                            if widget_key in st.session_state:
                                del st.session_state[widget_key]
                        save_chats(chats)
                        st.success(f"Deleted {filename} from the library and chats.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not delete {filename}: {error}")
        else:
            st.caption("No indexed documents.")

    if index_files_exist():
        try:
            indexed_chunks, _ = load_documents()
            backend = load_embedding_backend()
            label = "local fallback" if backend == "local-hash-v1" else "semantic model"
            st.caption(f"Index ready: {len(indexed_chunks)} passages · {label}")
        except Exception as error:
            st.error(f"Index cannot be loaded: {error}")

tab_chat, tab_notes, tab_quiz, tab_plan, tab_health = st.tabs(
    ["💬 Chat", "📝 Notes", "🎯 Quiz", "📅 Study Plan", "🩺 System Health"]
)

with tab_chat:
    st.subheader(active_chat.get("title", "New chat"))
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources Used"):
                    for index, source in enumerate(message["sources"], start=1):
                        citation = source.get("citation", index)
                        st.markdown(f"**[{citation}] {source['source']}**")
                        st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source["text"])

    question = st.chat_input("Ask about attached files or anything else...")
    if question:
        active_chat["messages"].append({"role": "user", "content": question})
        if active_chat.get("title") == "New chat":
            active_chat["title"] = question[:36] + ("…" if len(question) > 36 else "")
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            thinking_status = st.empty()
            thinking_status.caption("Reasoning through the question…" if smart_thinking else "Generating answer…")
            response_area = st.empty()
            streamed_text = []
            chars_since_refresh = [0]

            def show_token(piece: str) -> None:
                streamed_text.append(piece)
                thinking_status.empty()
                chars_since_refresh[0] += len(piece)
                if chars_since_refresh[0] >= 32:
                    response_area.markdown("".join(streamed_text) + " ▌")
                    chars_since_refresh[0] = 0

            result = generate_answer(
                question, top_k=4, document_ids=active_chat.get("document_ids", []),
                history=[item for item in active_chat["messages"][:-1]
                         if not (item.get("role") == "assistant" and
                                 (item.get("sources") or set(item.get("used_document_ids", []))
                                  - set(active_chat.get("document_ids", []))))],
                on_token=show_token,
                smart_thinking=smart_thinking,
            )
            answer, sources = str(result["answer"]), result["sources"]
            thinking_status.empty()
            response_area.markdown(answer)
            if result.get("used_document_ids"):
                quality = check_answer_quality(answer, sources)
                if not quality["passed"]:
                    st.warning("This document-based answer has no complete source attribution. Check the passages before relying on it.")
            if sources:
                with st.expander("Sources Used"):
                    for index, source in enumerate(sources, start=1):
                        citation = source.get("citation", index)
                        st.markdown(f"**[{citation}] {source['source']}**")
                        st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source["text"])
        active_chat["messages"].append({"role": "assistant", "content": answer,
                                        "sources": sources,
                                        "used_document_ids": result.get("used_document_ids", [])})
        save_chats(chats)
        st.rerun()

with tab_notes:
    st.subheader("📝 AI Notes Generator")
    notes_topic = st.text_input("Topic", key="notes_topic", placeholder="Example: Machine Learning")
    detail_level = st.selectbox("Detail level", ["Quick Revision", "Detailed", "Exam Revision"])
    if st.button("Generate Notes"):
        try:
            with st.spinner("Generating notes from attached documents..."):
                generated = generate_notes(notes_topic, detail_level, active_chat.get("document_ids", []), smart_thinking)
            st.session_state.generated_notes = generated
        except Exception as error:
            st.error(str(error))
    if st.session_state.get("generated_notes"):
        generated = st.session_state.generated_notes
        st.markdown(generated["notes"])
        st.download_button("Download Notes (.txt)", generated["notes"],
                           file_name="AETHER_Study_Notes.txt", mime="text/plain")
        if st.button("Save Notes locally"):
            try:
                path = save_generated_text(generated["topic"], generated["notes"])
                st.success(f"Saved to: {path}")
            except Exception as error:
                st.error(str(error))
        with st.expander("Sources used"):
            for source in generated["sources"]:
                st.markdown(f"**{source['source']}**")
                st.write(source["text"])

with tab_quiz:
    st.subheader("🎯 Topic or Exam Quiz")
    st.caption("Enter a topic or exam name to generate a quiz. Attach documents if you want questions grounded in them.")
    quiz_topic = st.text_input("Topic or exam name", key="quiz_topic", placeholder="Example: Cyber Security or NEET Biology")
    question_count = st.slider("Number of questions", 1, 10, 5)
    if st.button("Generate Quiz"):
        try:
            with st.spinner("Planning questions and checking the topic..."):
                st.session_state.quiz_data = generate_quiz(
                    quiz_topic, question_count, active_chat.get("document_ids", []), smart_thinking)
            st.session_state.quiz_submitted = False
            st.session_state.pop("quiz_result", None)
        except Exception as error:
            st.error(str(error))
    quiz_data = st.session_state.get("quiz_data")
    if quiz_data:
        questions = quiz_data["questions"]
        with st.form("quiz_form"):
            answers = {}
            for index, item in enumerate(questions):
                st.markdown(f"**{index + 1}. {item['question']}**")
                answers[index] = st.radio("Select your answer", list("ABCD"),
                    format_func=lambda key, options=item["options"]: f"{key}. {options[key]}",
                    key=f"quiz-answer-{index}")
            submitted = st.form_submit_button("Submit Quiz")
        if submitted:
            st.session_state.quiz_result = evaluate_quiz(questions, answers)
            st.session_state.quiz_submitted = True
        if st.session_state.get("quiz_submitted"):
            result = st.session_state.quiz_result
            st.metric("Score", f"{result['correct']} / {result['total']}")
            st.metric("Percentage", f"{result['percentage']}%")
            for item in result["results"]:
                st.success(item["question"]) if item["is_correct"] else st.error(item["question"])
                st.write(f"Your answer: {item['your_answer']} · Correct: {item['correct_answer']}")
                st.caption(item["explanation"])

with tab_plan:
    st.subheader("📅 AI Study Plan Generator")
    st.caption("A topic or exam name is enough. Attach a syllabus or notes for a document-grounded plan.")
    subject = st.text_input("Subject, topic, or exam name", key="study_subject", placeholder="Example: Machine Learning or GATE CSE")
    days = st.number_input("Days available", 1, 90, 7)
    minutes_per_day = st.number_input("Study minutes per day", 10, 720, 120, step=10)
    if st.button("Create Study Plan"):
        try:
            with st.spinner("Building a sequenced plan with practice and review checkpoints..."):
                st.session_state.study_plan_result = generate_study_plan(
                    subject, int(days), int(minutes_per_day), active_chat.get("document_ids", []), smart_thinking)
            st.session_state.study_plan_subject = subject
        except Exception as error:
            st.error(str(error))
    plan_result = st.session_state.get("study_plan_result")
    if plan_result:
        st.markdown(plan_result["plan"])
        st.download_button("Download Study Plan", plan_result["plan"],
                           file_name="AETHER_Study_Plan.txt", mime="text/plain")
        if st.button("Save Study Plan locally"):
            try:
                path = save_generated_text(st.session_state.get("study_plan_subject", "Study_Plan"), plan_result["plan"])
                st.success(f"Saved to: {path}")
            except Exception as error:
                st.error(str(error))
        with st.expander("Sources used"):
            for source in plan_result["sources"]:
                st.markdown(f"**{source['source']}**")
                st.write(source["text"])

with tab_health:
    st.header("🩺 AETHER System Health")
    if st.button("Run Health Check"):
        report = run_health_check()
        st.subheader("Overall Status: " + report["overall_status"].upper())

        index_status = report["index"]
        st.write("### Document Index")
        if index_status["status"] == "ok":
            st.success("Index is structurally consistent.")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Documents", index_status["documents"])
            metric_columns[1].metric("Chunks", index_status["chunks"])
            metric_columns[2].metric("Embeddings", index_status["embeddings"])
        else:
            st.error(index_status["message"])

        ollama_status = report["ollama"]
        st.write("### Local AI Service")
        if ollama_status["status"] == "ok":
            st.success("Ollama is reachable.")
            st.write("Available models:", ollama_status["models"])
        else:
            st.error(ollama_status["message"])
