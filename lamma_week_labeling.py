import streamlit as st
import torch
import pandas as pd
import io
from transformers import pipeline
from itertools import combinations

# Model ID
MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Persistent session state to retain the dataframes
if "STATEMENT_DATAFRAME" not in st.session_state:
    st.session_state.STATEMENT_DATAFRAME = pd.DataFrame(columns=["speaker", "statement", "label"])
if "AGRUMENT_DATAFRAME" not in st.session_state:
    st.session_state.AGRUMENT_DATAFRAME = pd.DataFrame(columns=["speaker1", "statement1", "speaker2", "statement2", "label"])

# Initialize the model pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        }
    )

pipe = load_pipeline()

def compare_agrument_statements(speaker1:str, statement1_list: list, speaker2: str = None, statement2_list: list = None):
    for i, statement1 in enumerate(statement1_list):
        indices_to_compare = [i-1, i, i+1]
        for j in indices_to_compare:
            if 0 <= j < len(statement2_list):
                statement2 = statement2_list[j]
                if statement1 != statement2:
                    _, result = classify_arguments(speaker1, statement1, speaker2, statement2)
                    classify_arguments_dataframe(speaker1, statement1, speaker2, statement2, result)

def upload_dataset_classify(uploaded_file, speaker1: str, speaker2: str, column_list: list):
    df = pd.read_csv(uploaded_file)
    if speaker2 == "":
        statement1_list = df[df[column_list[0]] == speaker1][column_list[1]].tolist()
        for statement1 in statement1_list:
            _, result = classify_statement(speaker1, statement1)
            classify_statement_dataframe(speaker1, statement1, result)
    else:
        statement1_list = df[df[column_list[0]] == speaker1][column_list[1]].tolist()
        statement2_list = df[df[column_list[0]] == speaker2][column_list[1]].tolist()
        compare_agrument_statements(speaker1, statement1_list, speaker2, statement2_list)

def local_dataset_agrument(df: pd.DataFrame):
    speaker_list = df["speaker"].unique()
    # Get statements for each speaker
    statements_by_speaker = {
        speaker: df[(df["speaker"] == speaker) & (df["label"] == "Argument")]["statement"].tolist()
        for speaker in speaker_list
    }

    # If there is only one speaker, remove duplicates and compare their statements with each other
    if len(speaker_list) == 1:
        speaker = speaker_list[0]
        statement_list = list(set(statements_by_speaker[speaker]))  # Remove duplicates
        compare_agrument_statements(speaker, statement_list, speaker, statement_list)
    else:
        # Compare statements across different speakers
        for speaker1, speaker2 in combinations(speaker_list, 2):
            statement1_list = statements_by_speaker[speaker1]
            statement2_list = statements_by_speaker[speaker2]
            compare_agrument_statements(speaker1, statement1_list, speaker2, statement2_list)

def classify_statement(speaker: str, statement: str):
    messages = [
        {"role": "system:\n", "content": "You are a helpful assistant!\n"},
        {"role": "user:\n", "content": "I will provide you with a statement from a debate. The format is `speaker : speaker_statement`.\n"},
        {"role": "user:\n", "content": "Your task is to determine if the statement is a Fact or an Argument.\n"},
        {"role": "user:\n", "content": "Fact: A statement that presents verifiable, objective information or an event that occurred.\n"},
        {"role": "user:\n", "content": "Argument: A statement that presents reasoning or tries to persuade someone to believe or act in a certain way.\n"},
        {"role": "user:\n", "content": f"Please classify the following statement as a Fact or Argument, only give the label: {speaker} : {statement}\n"},
    ]
    return call_model(messages)

def classify_arguments(speaker1: str,speaker1_statement: str, speaker2: str,speaker2_statement: str):
    messages = [
    {"role": "system\n", "content": "You are a helpful assistant!\n"},
    {"role": "user\n", "content": "I will provide you with two statements from a debate. The format is `speaker1 : speaker_statement1 <-> speaker2 : speaker_statement2`.\n"},
    {"role": "user\n", "content": "Your task is to classify if the two statements as either Restatement, Counterargument, or Neutral.\n"},
    {"role": "user\n", "content": "Restatement: The second statement restates or reinforces the first.\n"},
    {"role": "user\n", "content": "Counterargument: The second statement opposes the first.\n"},
    {"role": "user\n", "content": "Neutral: No clear relationship between the statements.\n"},
    {"role": "user\n", "content": f"Please classify the following two statements as Restatement, Counterargument, or Neutral, only give the label: {speaker1} : {speaker1_statement}  <-> {speaker2} : {speaker2_statement}\n"}

    ]
    return call_model(messages)

def classify_statement_dataframe(speaker: str, statement: str, label: str):
        new_row = pd.DataFrame([{
            "speaker": speaker,
            "statement": statement,
            "label": label
        }])
        st.session_state.STATEMENT_DATAFRAME = pd.concat([st.session_state.STATEMENT_DATAFRAME, new_row], ignore_index=True)

def classify_arguments_dataframe(speaker1: str, statement1: str, speaker2: str, statement2: str, label: str):
        new_row = pd.DataFrame([{
            "speaker1": speaker1,
            "statement1": statement1,
            "speaker2": speaker2,
            "statement2": statement2,
            "label": label
        }])
        st.session_state.AGRUMENT_DATAFRAME = pd.concat([st.session_state.AGRUMENT_DATAFRAME, new_row], ignore_index=True)

def get_csv_download_link(df, filename="data.csv"):
    # Convert the dataframe to CSV
    csv = df.to_csv(index=False)
    # Use io to create a download link
    buffer = io.StringIO(csv)
    return buffer.getvalue()

def call_model(messages):
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=[pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    clean_prompt = prompt.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
    outputs=outputs[0]['generated_text'][len(prompt):].strip() # Delete the prompt

    return clean_prompt, outputs

# Streamlit UI
st.title("Llama Weak Labeling: Argument vs. Fact Classification")

# Persistent session state to retain outputs
if "statement_result" not in st.session_state:
    st.session_state.statement_prompt = ""
    st.session_state.statement_result = ""

if "arguments_result" not in st.session_state:
    st.session_state.arguments_prompt = ""
    st.session_state.arguments_result = ""

## Classify Statements UI
st.markdown("## Classify statements")
speaker = st.text_input("Speaker's Name:")
statement = st.text_area("Speaker's Statement:")

st.markdown("### Upload and Classify Dataset")
st.text("Upload your dataset and generate classifications without displaying results.")

speaker = st.text_input("Speaker's Name:", key="speaker_upload")
column_input = st.text_input("Enter column names (separated by ';'):", "speaker;text", key="column_statement")
column_list = [col.strip() for col in column_input.split(";")]
uploaded_file = st.file_uploader("Choose a file", type="csv", key="file_uploader_statement")

if st.button("Classify Statement"):
    if uploaded_file and speaker:
        upload_dataset_classify(uploaded_file, speaker , "", column_list=column_list)
    elif speaker and statement:
        prompt, result = classify_statement(speaker, statement)
        st.session_state.statement_prompt = prompt
        st.session_state.statement_result = result
        classify_statement_dataframe(speaker, statement, result)
    else:
        st.warning("Please enter both a speaker and a statement or enter speaker and upload file")

if st.session_state.statement_result and st.session_state.statement_prompt:
    st.markdown("## Statement Classification Result:")
    st.write("### Prompt:")
    st.write(st.session_state.statement_prompt, language="plaintext")
    st.write("### Result:")
    st.write(st.session_state.statement_result, language="plaintext")

st.markdown("### Download Statement Data")
st.text("The statement classification data updates after each classification.")

csv_data = get_csv_download_link(st.session_state.STATEMENT_DATAFRAME)
st.download_button(
    label="Download Statement Data as CSV",
    data=csv_data,
    file_name="statement_data.csv",
    mime="text/csv"
)

st.markdown("### Dataframe output:")
if st.button("Clear Dataframe", key="clear_statement"):
    st.session_state.STATEMENT_DATAFRAME = pd.DataFrame(columns=["speaker", "statement", "label"])
st.dataframe(st.session_state.STATEMENT_DATAFRAME, use_container_width=True)

## Classify Agruments UI
st.markdown("## Classify Arguments")

st.markdown("### Compare using dataset above")
internal_dataset=st.checkbox("Classify Arguments using Statements dataset")

st.text("To classify arguments, provide the details of both speakers and their statements.")

speaker1 = st.text_input("Speaker 1's Name:")
statement1 = st.text_area("Speaker 1's Statement:")

speaker2 = st.text_input("Speaker 2's Name:")
statement2 = st.text_area("Speaker 2's Statement:")

st.text("# Upload and Classify Dataset")
st.text("Upload your dataset and generate classifications without displaying results")

speaker1 = st.text_input("Speaker 1's Name:", key="speaker1_upload")
speaker2 = st.text_input("Speaker 2's Name:", key="speaker2_upload")
column_input = st.text_input("Enter column names (separated by ';'):", "speaker;statement", key="column_agrument")
column_list = [col.strip() for col in column_input.split(";")]
uploaded_file = st.file_uploader("Choose a file", type="csv", key="file_uploader_agrument")

if st.button("Classify Arguments"):
    if internal_dataset:
        local_dataset_agrument(st.session_state.STATEMENT_DATAFRAME)
    elif uploaded_file and speaker1 and speaker2:
        upload_dataset_classify(uploaded_file, speaker1, speaker2, column_list)
    elif speaker1 and statement1 and speaker2 and statement2:
        prompt, result = classify_arguments(speaker1, statement1, speaker2, statement2)
        st.session_state.arguments_prompt = prompt
        st.session_state.arguments_result = result
        classify_arguments_dataframe(speaker1, statement1, speaker2, statement2, result)
    else:
        st.warning("Please enter both speakers and statements.")

if st.session_state.arguments_result and st.session_state.arguments_prompt:
    st.markdown("## Arguments Classification Result:")
    st.markdown("### Prompt:")
    st.write(st.session_state.arguments_prompt, language="plaintext")
    st.markdown("### Result:")
    st.write(st.session_state.arguments_result, language="plaintext")

st.markdown("### Download Argument Data")
st.text("The argument classification data updates after each classification.")

csv_data = get_csv_download_link(st.session_state.AGRUMENT_DATAFRAME)
st.download_button(
    label="Download Argument Data as CSV",
    data=csv_data,
    file_name="argument_data.csv",
    mime="text/csv"
)

st.markdown("### Dataframe output:")
if st.button("Clear Dataframe", key="clear_agrument"):
    st.session_state.AGRUMENT_DATAFRAME = pd.DataFrame(columns=["speaker1", "statement1", "speaker2", "statement2", "label"])
st.dataframe(st.session_state.AGRUMENT_DATAFRAME, use_container_width=True)