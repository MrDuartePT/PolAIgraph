import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize session state to store outputs
if "uploaded_results" not in st.session_state:
    st.session_state.uploaded_results = None
if "internal_results" not in st.session_state:
    st.session_state.internal_results = None

# Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, head_type="basic", dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = get_classification_head(hidden_size, head_type, dropout_rate)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

@st.cache_data
# Define the same TranfomerClassifier class and Classification head
def get_classification_head(hidden_size, head_type="basic", dropout_rate=0.1):
    if head_type == "basic":
        return nn.Linear(hidden_size, 2)  # Simple linear layer
    elif head_type == "mlp":
        return nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
    elif head_type == "gelu_norm":
        return nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
    else:
        raise ValueError("Unknown head type")

@st.cache_resource
def load_models():
    # Load the models
    ## Fact and Agrument Classifier
    model_fact_agrument_classifier = AutoModelForSequenceClassification.from_pretrained("../models/distilbert_fact_agrument_classifier").to(device)
    tokenizer_fact_agrument_classifier = BertTokenizer.from_pretrained("../models/distilbert_fact_agrument_classifier")

    ## Fact Classifier
    model_fact_classifier = TransformerClassifier("bert-base-uncased", "basic")
    model_fact_classifier.load_state_dict(torch.load("../models/fact_checker_model.pt", map_location=device))
    model_fact_classifier.to(device)
    tokenizer_fact_classifier = AutoTokenizer.from_pretrained("../models/fact_checker_tokenizer")

    ## Agrument Classifier (using pipeline)
    model_agrument_classifier = pipeline("text-classification", model="../models/distilbert_agrument_classifier", device=device)

    return model_fact_agrument_classifier, tokenizer_fact_agrument_classifier, model_fact_classifier, tokenizer_fact_classifier, model_agrument_classifier

# Load models when Streamlit app starts
model_fact_agrument_classifier, tokenizer_fact_agrument_classifier, model_fact_classifier, tokenizer_fact_classifier, model_agrument_classifier = load_models()

def datasets_analyse(df_fact_agrument: pd.DataFrame, df_fact: pd.DataFrame, df_agrument: pd.DataFrame, debate_name: str):
    st.subheader(f"🗣️ Analyzing the debate dataset: {debate_name}\n")
    st.markdown(f"#### 📊 Displaying the analysis plot for the debate...")
    
    # Group by speaker and verdict (True/False) for fact analysis
    speaker_counts = df_fact.groupby(['speaker', 'label']).size().unstack(fill_value=0)
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    #---------------Fact/Agrument distribution---------------
    fact_agrument_counts = df_fact_agrument['label'].value_counts()
    axs[0].pie(fact_agrument_counts, labels=fact_agrument_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(fact_agrument_counts)))
    axs[0].set_title("Fact/Argument Distribution")
    axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    #---------------False/True distribution---------------
    false_true_counts = df_fact['label'].value_counts()
    axs[1].pie(false_true_counts, labels=false_true_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(false_true_counts)))
    axs[1].set_title("Verdict Distribution (True/False)")
    axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    #---------------Argument distribution---------------
    agrument_label_counts = df_fact['label'].value_counts()
    axs[2].pie(agrument_label_counts, labels=agrument_label_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(agrument_label_counts)))
    axs[2].set_title("Argument Distribution")
    axs[2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

    #---------False/True distribution per speaker---------
    fig_1, axs_1 = plt.subplots(1, len(df_fact['speaker'].unique()), figsize=(16, 5))
    for i, speaker in enumerate(df_fact['speaker'].unique()):
        speaker_data = df_fact[df_fact['speaker'] == speaker]['label'].value_counts()
        
        axs_1[i].pie(speaker_data, labels=speaker_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(speaker_data)))
        axs_1[i].set_title(f"{speaker}'s Verdict Distribution")
        axs_1[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig_1)
    
    # Find speaker with most False and most True statements
    if 'False' in speaker_counts.columns:
        most_false_speaker = speaker_counts['False'].idxmax()
        st.write(f"🟥 Speaker with the most False statements: {most_false_speaker}")
    else:
        st.warning("⚠️ No 'False' statements found in the facts dataset.")
    
    if 'True' in speaker_counts.columns:
        most_true_speaker = speaker_counts['True'].idxmax()
        st.write(f"🟩 Speaker with the most True statements: {most_true_speaker}")
    else:
        st.warning("⚠️ No 'True' statements found in the facts dataset.")

    # Median and Average confidence for each dataset
    st.markdown(f"#### 📊 Confidence Statistics:")
    fact_agrument_confidence = df_fact_agrument["confidence"]
    fact_confidence = df_fact["confidence"]
    agrument_confidence = df_agrument["confidence"]

    st.write(f"   - Fact/Argument Dataset: Median Confidence = {fact_agrument_confidence.median():.2f}, Mean Confidence = {fact_agrument_confidence.mean():.2f}")
    st.write(f"   - Fact Dataset: Median Confidence = {fact_confidence.median():.2f}, Mean Confidence = {fact_confidence.mean():.2f}")
    st.write(f"   - Argument Dataset: Median Confidence = {agrument_confidence.median():.2f}, Mean Confidence = {agrument_confidence.mean():.2f}")

# Funtion to Classifie the text
def run_model(text, model, tokenizer, labels):
    # Tokenize the text and add padding/truncation
    encoding = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make the model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1).item()  # Convert logits to predicted class index
        softmax_probs = F.softmax(outputs.logits, dim=1)
        confidence = softmax_probs[0, predictions].item() # get confidence result

    # Output the prediction label based on the model's class index
    predicted_label = labels[predictions]  # Map the predicted index to the label
    return confidence, predicted_label

def create_prediction_dataframe(statements, speaker, model, tokenizer, labels):
    results = []
    for statement in statements:
        confidence, predicted_label = run_model(statement, model, tokenizer, labels)
        results.append({
            "speaker": speaker,
            "statement": statement,
            "label": predicted_label,
            "confidence": confidence
        })
    return pd.DataFrame(results)

def divide_fact_agrument(speaker: str, statement_list: list):
    labels = {0: "Fact", 1: "Argument"}
    return create_prediction_dataframe(statement_list, speaker, model_fact_agrument_classifier, tokenizer_fact_agrument_classifier, labels)

def classification_facts(speaker: str, df: pd.DataFrame):
    labels = {0: "False", 1: "True"}
    fact_statements = df[(df["label"] == "Fact") & (df["speaker"] == speaker)]["statement"].to_list()
    return create_prediction_dataframe(fact_statements, speaker, model_fact_agrument_classifier, tokenizer_fact_agrument_classifier, labels)

def classification_agruments(speaker1: str, speaker2: str, df: pd.DataFrame):
    id2label = {0: "Neutral", 1: "Counterargument", 2: "Restatement"}
    argument_pairs = []

    argument_statements1 = df[(df["label"] == "Argument") & (df["speaker"] == speaker1)]["statement"].to_list()
    argument_statements2 = df[(df["label"] == "Argument") & (df["speaker"] == speaker2)]["statement"].to_list()

    for i, statement1 in enumerate(argument_statements1):
        for j in range(i, len(argument_statements2)):
            statement2 = argument_statements2[j]
            model_result = model_agrument_classifier(f"{statement1} </s></s> {statement2}", truncation=True)
            argument_pairs.append({
                "speaker1": speaker1,
                "statement1": statement1,
                "speaker2": speaker2,
                "statement2": statement2,
                "label": model_result[0]['label'],
                "confidence": model_result[0]['score']
            })

    result_df = pd.DataFrame(argument_pairs)
    #result_df["label"] = result_df["label"].replace(id2label)
    return result_df

# Run on Internal datasets
@st.cache_data
def app_test_internal_datasets():
    # Load datasets
    second_presidential_debate = pd.read_csv(os.path.join('../Datasets', 'us_debates', 'original', 'us_election_2020_2nd_presidential_debate.csv'))
    trump_town_hall_debate = pd.read_csv(os.path.join('../Datasets', 'us_debates', 'original', 'us_election_2020_trump_town_hall.csv'))
    biden_town_hall_debate = pd.read_csv(os.path.join('../Datasets', 'us_debates', 'original', 'us_election_2020_biden_town_hall.csv'))

    # Classify fact and arguments
    second_presidential_fact_agrument = pd.concat([
        divide_fact_agrument("Donald Trump", second_presidential_debate[second_presidential_debate["speaker"] == "Donald Trump"]["text"].to_list()),
        divide_fact_agrument("Joe Biden", second_presidential_debate[second_presidential_debate["speaker"] == "Joe Biden"]["text"].to_list())
    ])

    trump_town_hall_fact_agrument = pd.concat([
        divide_fact_agrument("Savannah Guthrie", trump_town_hall_debate[trump_town_hall_debate["speaker"] == "Savannah Guthrie"]["text"].to_list()),
        divide_fact_agrument("President Trump", trump_town_hall_debate[trump_town_hall_debate["speaker"] == "President Trump"]["text"].to_list())
    ])

    biden_town_hall_fact_agrument = pd.concat([
        divide_fact_agrument("George Stephanopoulos", biden_town_hall_debate[biden_town_hall_debate["speaker"] == "George Stephanopoulos"]["text"].to_list()),
        divide_fact_agrument("Joe Biden", biden_town_hall_debate[biden_town_hall_debate["speaker"] == "Joe Biden"]["text"].to_list())
    ])

    # Classify facts and return results
    second_presidential_fact = pd.concat([
        classification_facts("Donald Trump", second_presidential_fact_agrument),
        classification_facts("Joe Biden", second_presidential_fact_agrument)
    ])

    trump_town_hall_fact = pd.concat([
        classification_facts("Savannah Guthrie", trump_town_hall_fact_agrument),
        classification_facts("President Trump", trump_town_hall_fact_agrument)
    ])

    biden_town_hall_fact = pd.concat([
        classification_facts("George Stephanopoulos", biden_town_hall_fact_agrument),
        classification_facts("Joe Biden", biden_town_hall_fact_agrument)
    ])

    # Classify Agruments and return results
    second_presidential_agrument = classification_agruments("Donald Trump", "Joe Biden", second_presidential_fact_agrument)

    trump_town_hall_agrument = classification_agruments("Savannah Guthrie", "President Trump", trump_town_hall_fact_agrument)

    biden_town_hall_agrument = classification_agruments("George Stephanopoulos", "Joe Biden", biden_town_hall_fact_agrument)

    return (
        second_presidential_fact_agrument, trump_town_hall_fact_agrument, biden_town_hall_fact_agrument, 
        second_presidential_fact, trump_town_hall_fact, biden_town_hall_fact, second_presidential_agrument, 
        trump_town_hall_agrument, biden_town_hall_agrument
    )

# Streamlit app
def app():
    st.title("PolAIgraph")

    st.header("Upload Dataset")
    st.write("Colums: speaker | text")
    speaker1 = st.text_input("Speaker 1's Name:")
    speaker2 = st.text_input("Speaker 2's Name:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if st.button("Start", key="uploaded_dataset"):
        if uploaded_file is None:
            st.error("Please upload a CSV file")
        elif not speaker1 or not speaker2:
            st.error("Please enter both Speaker 1 and Speaker 2 names.")
        else:
            try:
                # Read the uploaded file
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Dataframe Loaded: {df_upload.shape[0]} rows and {df_upload.shape[1]} columns")

                # Process dataset
                uploaded_fact_agrument = pd.concat([
                    divide_fact_agrument(speaker1, df_upload[df_upload["speaker"] == speaker1]["text"].to_list()),
                    divide_fact_agrument(speaker2, df_upload[df_upload["speaker"] == speaker2]["text"].to_list())
                ])
                uploaded_fact = pd.concat([
                    classification_facts(speaker1, uploaded_fact_agrument),
                    classification_facts(speaker2, uploaded_fact_agrument)
                ])
                uploaded_agrument = classification_agruments(speaker1, speaker2, uploaded_fact_agrument)
                st.dataframe(uploaded_agrument)

                st.session_state.uploaded_results = {
                    "fact_agrument": uploaded_fact_agrument,
                    "fact": uploaded_fact,
                    "agrument": uploaded_agrument
                }
            
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.session_state.uploaded_results:
        # Display the datasets and graphs
        st.write("### Upload Debate Dataset - Fact and Argument Classification")
        st.dataframe(uploaded_fact_agrument)

        st.write("### Upload Debate Dataset - Fact Classification")
        st.dataframe(uploaded_fact)

        st.write("### Upload Debate Dataset - Argument Classification")
        st.dataframe(uploaded_agrument)

        datasets_analyse(uploaded_fact_agrument, uploaded_fact, uploaded_agrument, "Uploaded Debate Dataset")
    

    st.header("Application Test (Use Internal Datasets)")
    if st.button("Start", key="internal_dataset"):
        second_presidential_fact_agrument, trump_town_hall_fact_agrument, biden_town_hall_fact_agrument,second_presidential_fact, trump_town_hall_fact, biden_town_hall_fact, second_presidential_agrument, trump_town_hall_agrument, biden_town_hall_agrument = app_test_internal_datasets()

        st.session_state.internal_results = {
            "second_presidential": {
                "fact_agrument": second_presidential_fact_agrument,
                "fact": second_presidential_fact,
                "agrument": second_presidential_agrument,
                "dataset_name": "USA 2020 Election 2nd Presidential Debate"
            },
            "trump_town_hall": {
                "fact_agrument": trump_town_hall_fact_agrument,
                "fact": trump_town_hall_fact,
                "agrument": trump_town_hall_agrument,
                "dataset_name": "USA 2020 Town Hall Trump Debate"
            },
            "biden_town_hall": {
                "fact_agrument": biden_town_hall_fact_agrument,
                "fact": biden_town_hall_fact,
                "agrument": biden_town_hall_agrument,
                "dataset_name": "USA 2020 Town Hall Biden Debate"
            }
        }

    # Display Internal Dataset Results
    if st.session_state.internal_results:
        for debate, results in st.session_state.internal_results.items():
            st.write(f"### {debate.replace('_', ' ').title()} - Fact and Argument Classification")
            st.dataframe(results["fact_agrument"])

            st.write(f"### {debate.replace('_', ' ').title()} - Fact Classification")
            st.dataframe(results["fact"])

            st.write(f"### {debate.replace('_', ' ').title()} - Argument Classification")
            st.dataframe(results["agrument"])

            datasets_analyse(
                results["fact_agrument"],
                results["fact"],
                results["agrument"],
                results["dataset_name"]
            )

# Run the app
if __name__ == "__main__":
    app()
