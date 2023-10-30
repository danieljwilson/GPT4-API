import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)


########################
# General Code Pattern #
########################

# response = openai.ChatCompletion.create(
#               model="MODEL_NAME",
#               messages=[{"role": "system", "content": 'SPECIFY HOW THE AI ASSISTANT SHOULD BEHAVE'},
#                         {"role": "user", "content": 'SPECIFY WANT YOU WANT THE AI ASSISTANT TO SAY'}
#               ])

#####################
# Datacamp Tutorial #
#####################
# https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python

# From the IPython.display package, import display and Markdown
from IPython.display import display, Markdown

# Import yfinance as yf
import yfinance as yf

# Define the system message
system_msg = 'You are a helpful assistant who understands data science.'

# Define the user message
user_msg = 'Create a small dataset about total sales over the last year. The format of the dataset should be a data frame with 12 rows and 2 columns. The columns should be called "month" and "total_sales_usd". The "month" column should contain the shortened forms of month names from "Jan" to "Dec". The "total_sales_usd" column should contain random numeric values taken from a normal distribution with mean 100000 and standard deviation 5000. Provide Python code to generate the dataset, then provide the output in the format of a markdown table.'

# Create a dataset using GPT
response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "system", "content": system_msg},
                                                  {"role": "user", "content": user_msg}])

response["choices"][0]["finish_reason"]

# Extract AI assistant's message
display(Markdown(response["choices"][0]["message"]["content"]))


# Helper function to call GPT
def chat(system, user_assistant):
  assert isinstance(system, str), "`system` should be a string"
  assert isinstance(user_assistant, list), "`user_assistant` should be a list"
  system_msg = [{"role": "system", "content": system}]
  user_assistant_msgs = [
      {"role": "assistant", "content": user_assistant[i]} if i % 2 else {"role": "user", "content": user_assistant[i]}
      for i in range(len(user_assistant))]

  msgs = system_msg + user_assistant_msgs
  response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=msgs)
  status_code = response["choices"][0]["finish_reason"]
  assert status_code == "stop", f"The status code was {status_code}."
  return response["choices"][0]["message"]["content"]

# Example use of function
response_fn_test = chat("You are a machine learning expert.",["Explain what a neural network is."])

display(Markdown(response_fn_test))

##########################
# Analyze sample dataset #
##########################

# Assign the content from the response in Task 1 to assistant_msg
assistant_msg = response["choices"][0]["message"]["content"]

# Define a new user message
user_msg2 = 'Using the dataset you just created, write code to calculate the mean of the `total_sales_usd` column. Also include the result of the calculation.'

# Create an array of user and assistant messages
user_assistant_msgs = [user_msg, assistant_msg, user_msg2]

# Get GPT to perform the request
response_calc = chat(system_msg, user_assistant_msgs)

# Display the generated content
display(Markdown(response_calc))
