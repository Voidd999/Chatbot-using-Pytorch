import streamlit as st
import chat
import json
import training
import torch
import nn

with st.sidebar:
    st.subheader("Simple Pytorch ChatBot", "Simple Pytorch ChatBot")
    choice = st.radio("Select", ["Upload data", "Train", "Chat"])

if choice == "Upload data":
    st.title("Upload Custom Intents")

    intents = st.file_uploader("Upload JSON Intents", type=["json"])
    download_json_path = "Data/sample-json-intents.json"
    with open(download_json_path, "r") as f:
        st.download_button(
            "Download sample intents",
            f.read(),
            file_name="sample-json-intents.json",
            mime="application/json",
        )
    if intents is not None:
        with open("Data/intents.json", "w") as f:
            f.write(intents.getvalue().decode("utf-8"))
        st.success("Intents uploaded successfully")

        st.json(json.load(intents))

    else:
        st.warning("Please upload intents to begin")


if choice == "Train":
    intents = json.loads(open("Data/intents.json").read())
    st.title("Train Chatbot")
    batch_size = st.number_input(
        "Enter Batch Size", min_value=5, max_value=100, value=40, step=5
    )
    hidden = st.number_input(
        "Enter Hidden Size", min_value=5, max_value=100, value=40, step=5
    )

    epochs = st.number_input(
        "Enter Epochs", min_value=1000, max_value=10000, value=5000, step=1000
    )
    learning_rate = 0.0001

    if st.button("Train"):
        st.text("Training")
        model = training.train(intents, batch_size, hidden, learning_rate, epochs)

        st.success("Model trained successfully")
        torch.save(model, "Data/data.pth")

        with open("Data/data.pth", "rb") as f:
            st.download_button(
                "Download model",
                f,
                file_name="data.pth",
                mime="application/octet-stream",
            )
if choice == "Chat":
    intents = json.loads(open("Data/intents.json").read())
    st.title("Chatbot Demo")
    st.markdown("What can I help you with? Type 'quit' to exit.")

    user_input = st.text_input("You:")
    end = ["quit", "bye"]
    if st.button("Ask", key="ask"):
        if user_input.lower() in end:
            st.text("Session ended")
        else:
            resp,accuracy = chat.get_response(user_input, "data.pth", intents)
            st.text(f"You: {user_input}\n{chat.name}: {resp}")
            st.caption(f'Accuracy : {accuracy:.3f}')
