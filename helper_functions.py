# Third-Party Imports
import streamlit as st
import langchain
from openai import OpenAI
# from index_functions import load_data
# from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
# from llama_index.llms import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Main function to generate responses from OpenAI's API, not considering indexed data
def generate_response(prompt, history, model_name, temperature):
    # Fetching the last message sent by the chatbot from the conversation history
    chatbot_message = history[-1]['content']

    # Fetching the first message that the user sent from the conversation history
    first_message = history[1]['content']

    # Fetching the last message that the user sent from the conversation history
    last_user_message = history[-2]['content']

    # Constructing a comprehensive prompt to feed to OpenAI for generating a response
    full_prompt = f"{prompt}\n\
        ### The original message: {first_message}. \n\
        ### Your latest message: {chatbot_message}. \n\
        ### Previous conversation history for context: {history}"

    # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": last_user_message}
        ]
    )

    # Extracting the generated response content from the API response object
    # full_response = api_response['choices'][0]['message']['content']
    full_response = api_response.choices[0].message.content

    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

def generate_response_index(prompt, history, model_name, temperature, chat_engine):
    # Fetching the last message sent by the chatbot from the conversation history
    chatbot_message = history[-1]['content']

    # Fetching the first message that the user sent from the conversation history
    first_message = history[1]['content']

    # Fetching the last message that the user sent from the conversation history
    last_user_message = history[-2]['content']

    # Constructing a comprehensive prompt to feed to OpenAI for generating a response
    full_prompt = f"{prompt}\n\
        ### The original message: {first_message}. \n\
        ### Your latest message: {chatbot_message}. \n\
        ### Previous conversation history for context: {history}"

    # Initializing a variable to store indexed data relevant to the user's last message
    index_response = ""

    # Fetching relevant indexed data based on the last user message using the chat engine
    response = chat_engine.chat(last_user_message)

    # Storing the fetched indexed data in a variable
    index_response = response.response

    # Adding the indexed data to the prompt to make the chatbot response more context-aware and data-driven
    full_prompt += f"\n### Relevant data from documents: {index_response}"

    # Making an API call to OpenAI to generate a chatbot response based on the constructed prompt
    api_response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": last_user_message}
        ]
    )

    # Extracting the generated response content from the API response object
    # full_response = api_response['choices'][0]['message']['content']
    full_response = api_response.choices[0].message.content

    # Yielding a response object containing the type and content of the generated message
    yield {"type": "response", "content": full_response}

##############################################################################################
    
# Additional functions
    
# def get_thanks_phrase():
#     """
#     Returns a random thanks phrase to be used as part of the CoPilots reply
#     Requires a dictionary of 'thanks_phrase' to work properly
#     """
#     selected_phrase = random.choice(thanks_phrases)
#     return selected_phrase

# def get_initial_message():
#     """
#     Randomize initial message of CoPilot
#     Requires a dictionary of 'initial_message' to work properly
#     """
#     initial_message = random.choice(initial_message_phrases)
#     return initial_message

# def generate_summary(model_name, temperature, summary_prompt):
#     """
#     Generate the summary; used in part of the response
#     """
#     summary_response = openai.ChatCompletion.create(
#         model=model_name,
#         temperature=temperature,
#         messages=[
#             {"role": "system", "content": "You are an expert at summarizing information effectively and making others feel understood"},
#             {"role": "user", "content": summary_prompt}
#         ]
#     )
#     summary = summary_response['choices'][0]['message']['content']
#     print(f"summary: {summary}, model name: {model_name}, temperature: {temperature}")
#     return summary

# def transform_bullets(content):
#     """
#     Enable 'summary' mode in which the CoPilot only responds with bullet points rather than paragraphs
#     """
#     try:
#         prompt = f"Summarize the following content in 3 brief bullet points while retaining the structure and conversational tone (using wording like 'you' and 'your idea'):\n{content}"
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             temperature=.2,
#             messages=[
#                 {"role": "system", "content": prompt}
#             ],
#         )
#         return response['chocies'][0]['message']['content'].strip()
#     except Exception as e:
#         print(response)
#         print("Error in transform_bullets:", e)
#         return content # Return the original content as a fallback

# def get_stage_prompt(stage):
#     """
#     Add relevant stage specific context into prompt
#     """
#     #### NOTE: Needs to be implemented
#     pass

# def grade_response(user_input, assistant_message, idea):
#     """
#     Grade the response based on length, relevancy, and depth of response
#     """
#     #### NOTE: Needs to be implemented
#     pass

# def generate_final_report():
#     """
#     Generate a final 'report' at the end of the conversation, summarizing the convo and providing a final recommendatio
#     """
#     #### NOTE: Needs to be implemented
#     pass