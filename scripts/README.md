# Getting Started

This project uses [Streamlit](https://streamlit.io/) to build an interactive user interface for various applications.

## Available Scripts

Navigate to the script directory and run the following commands to start the applications:

### Running the Poligraph App

```sh
streamlit run ./Poligraph_App.py
```

This command launches the Poligraph App.

- Open [http://localhost:8501/](http://localhost:8501/) in your browser to access the app.
- The app will automatically reload when you modify the code.
- Instructions on how to interact with the application are provided within the UI.

### Running the Llama Weak Labeling App

```sh
streamlit run ./llama_week_labeling.py
```

This script is responsible for generating labeled data using a weak labeling technique. It utilizes the `unsloth/llama-3-8b-Instruct-bnb-4bit` text generation with custom prompt to create the dataset to train the following models:

- **Fact/Argument Division BERT Model**
- **Argument Classification BERT Model**

- Open [http://localhost:8501/](http://localhost:8501/) in your browser to access the app.
- The app will automatically reload when changes are made to the code.
- Instructions on how to interact with the application are provided within the UI.

## Notes

Ensure you have all dependencies installed before running the scripts. Refer to the project's setup documentation for installation instructions.
