"""
Streamlit app for English Pronunciation Assessment using a multi-task Conformer model.

Features:
  • Upload audio and transcript to evaluate pronunciation.
  • Displays CTC-based phoneme recognition and phoneme-level error labels.
  • Highlights correct, substituted, and deleted phonemes in color.
  • Includes word-level error breakdown and result export as JSON.
"""

import streamlit as st
import os
import time
import torch
import json

from inference import run_inference

st.set_page_config(
    page_title="English Pronunciation Assessment",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK_ID = 0

if "page" not in st.session_state:
    st.session_state.page = "upload"
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "results" not in st.session_state:
    st.session_state.results = {}


def safe_rerun():
    st.rerun()


def build_colored_sequence(phones, errors):
    color_map = {"Correct": "green", "Substituted": "blue", "Deleted": "red"}
    tokens = []
    for ph, err in zip(phones, errors):
        color = color_map.get(err, "black")
        tokens.append(f"<span style='color:{color}'>{ph}</span>")
    return " ".join(tokens)


def display_color_legend():
    legend_html = """
    <p><strong>Color Legend:</strong></p>
    <ul style="list-style:none; padding-left:0;">
      <li><span style="color:green;">●</span>Correct</li>
      <li><span style="color:blue;">●</span>Substituted</li>
      <li><span style="color:red;">●</span>Deleted</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


# Upload Page
if st.session_state.page == "upload":
    st.title("English Pronunciation Assessment")
    st.write("Upload Audio & Enter Transcript.")
    audio_file = st.file_uploader("Upload Audio (WAV/MP3):", type=["wav", "mp3"])
    transcript_text = st.text_area("Enter Transcript:")
    if st.button("Submit"):
        if not audio_file:
            st.error("Please provide an audio file!")
        else:
            base_filename, ext = os.path.splitext(audio_file.name)
            audio_path = r"E:\IRP\sample\audio"
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            st.session_state.audio_path = audio_path
            st.session_state.transcript_text = transcript_text.strip()
            st.session_state.page = "progress"
            safe_rerun()

# Inference Progress Page
elif st.session_state.page == "progress":
    st.title("Running")
    prog = st.progress(0)
    time.sleep(0.5); prog.progress(20)
    st.write("Running ...")
    try:
        results = run_inference(
            audio_file=st.session_state.audio_path,
            transcript_text=st.session_state.transcript_text,
            phoneme_map_path=r"E:\IRP\backend\data\preprocessed_five\phoneme_map.json",
            model_checkpoint=r"E:\IRP\models\model_epoch_55.pt",
            device=DEVICE,
            blank_id=BLANK_ID
        )
        st.session_state.results = results
        st.write("Completed.")
    except Exception as e:
        st.session_state.results = {"error": f"Inference error: {e}"}
    prog.progress(100)
    st.session_state.page = "results"
    st.write("Done. Click below to see the results.")
    if st.button("Go to Results"):
        safe_rerun()

# Results Page
elif st.session_state.page == "results":
    st.title("Results")
    results = st.session_state.results

    if "error" in results:
        st.error(results["error"])
        if st.button("Go Back"):
            st.session_state.page = "upload"
            safe_rerun()
    else:
        st.subheader("Uploaded Audio")
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path)
        else:
            st.write("Audio file not found.")

        st.subheader("Transcript")
        st.write(st.session_state.transcript_text)

        st.subheader("Pronunciation Assessment Results")
        summary = results.get("summary", {})

        # ——— Detailed Stats (wide, 2‑col) ———
        with st.expander("Detailed Stats"):
            # Summary on top
            st.markdown("### Summary")
            st.write(summary.get("plain_text", ""))

            # Fetch sequences
            canonical   = results.get("canonical_phones", [])
            canon_err   = results.get("canonical_error_labels", [])
            recognized  = results.get("recognized_phones", [])
            word_data   = results.get("word_breakdown", [])

            # Two‑column layout (3:2)
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("### Reference Phonemes + Error Labels")
                if canonical and canon_err:
                    colored_html = build_colored_sequence(canonical, canon_err)
                    st.markdown(colored_html, unsafe_allow_html=True)
                    display_color_legend()
                else:
                    st.write("(No canonical phones or error labels.)")

            with col2:
                st.markdown("### Recognized Phoneme Sequence")
                if recognized:
                    st.write(" ".join(recognized))
                else:
                    st.write("(No recognized phonemes.)")

            # Word‑level Breakdown table
            st.markdown("---")
            st.markdown("### Word-level Breakdown")
            if word_data:
                table_rows = []
                for wb in word_data:
                    table_rows.append({
                        "Word":               wb["word"],
                        "Canonical Phones":   " ".join(wb["phones"]),
                        "Error Labels":       " ".join(wb["error_labels"]),
                    })
                st.table(table_rows)
            else:
                st.write("No word-level breakdown available.")

        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(results, indent=2),
            file_name="inference_results.json",
            mime="application/json"
        )

        if st.button("Go Back"):
            st.session_state.page = "upload"
            safe_rerun()
