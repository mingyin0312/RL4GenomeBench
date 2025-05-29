#######--------------------------
## This script will modify the mbox content to 
## the new_QA.json file with each item has (question,answer,context)
#######--------------------------


import mailbox
import csv
import re

def remove_quotes(text):
    # Regular expression to match lines that start with one or more '>' characters followed by a space
    clean_lines = [line for line in text.splitlines() if not re.match(r'^\s*(>+ )', line)]
    return '\n'.join(clean_lines)

def get_email_body(message):
    body = None  # Initialize body to None or an empty string
    if message.is_multipart():
        for part in message.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if (ctype == 'text/plain' and 'attachment' not in cdispo) or (ctype == 'text/html' and 'attachment' not in cdispo):
                body = part.get_payload(decode=True)
                body = body.decode('utf-8', errors='ignore')
                body = remove_quotes(body)
                return body  # Return the cleaned body once found
    else:
        body = message.get_payload(decode=True)
        body = body.decode('utf-8', errors='ignore')
        body = remove_quotes(body)
    return body


# Load the mbox file
mbox = mailbox.mbox('.../mbox') # your local mbox address

# Open a CSV file to write to
with open('testemail.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Message-ID', 'In-Reply-To', 'References', 'Subject', 'From', 'Date', 'To', 'Body'])

    for message in mbox:
        message_id = message['message-id']
        in_reply_to = message['in-reply-to']
        references = message['references']
        subject = message['subject']
        from_email = message['from']
        date = message['date']
        to_email = message['to']
        body = get_email_body(message)

        writer.writerow([message_id, in_reply_to, references, subject, from_email, date, to_email, body])

print("Conversion completed with adjusted email body extraction!")



# %%

import pandas as pd
# Adjust display settings
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 30)     # Show all rows
pd.set_option('display.max_colwidth', None) # Show full content of each cell
pd.set_option('display.width', None)        # Auto-detect the display width
df = pd.read_csv('/xxxx/testemail.csv')

df['Body'] = df['Body'].str.split('You received this message because you are subscribed to the Google Groups').str[0]
pattern = r"On\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+(2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019|2020|2021|2022|2023|2024)\s+at.*?wrote:"

# Split the 'body' column and keep only the part before the unwanted text
df['Body'] = df['Body'].str.split(pattern, n=1, expand=True)[0]
pattern = r"On\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2}\s+(AM|PM)\s+.*?<.*?>\s*\nwrote:"
df['Body'] = df['Body'].str.split(pattern, n=1, expand=True)[0]
df['Body'] = df['Body'].str.replace('\n', '', regex=False)
df['Message-ID'] = df['Message-ID'].str.replace(' ', '', regex=False)
df['Message-ID'] = df['Message-ID'].str.replace('<', '').str.replace('>', '')
df['In-Reply-To'] = df['In-Reply-To'].str.replace(' ', '', regex=False)
df['In-Reply-To'] = df['In-Reply-To'].str.replace('<', '').str.replace('>', '')
df['References'] = df['References'].str.extractall(r'<(.*?)>').groupby(level=0).agg(list)
df['References'] = df['References'].apply(lambda refs: [ref.replace(' ', '') for ref in refs] if isinstance(refs, list) else [])
df.head(30)


# %%

import pandas as pd
# Assuming 'df' is already loaded and formatted with 'References' as lists of message IDs.

# Identifying root messages
roots = df[df['In-Reply-To'].isna()]

# Preparing a list to store threads
threads = []

# Processing each root to find the thread end and reconstruct the thread
for index, root in roots.iterrows():
    thread_id = root['Message-ID']
    thread_subject = root['Subject']
    
    # Find messages with this root ID in their references
    thread_messages = df[df['References'].apply(lambda x: thread_id in x if x else False)]
    thread_body = '***SUBJECT: {subject}***'.format(subject=root['Subject'])

    if not thread_messages.empty:
        # Find the last message by identifying the one with the longest 'References' list
        last_message = thread_messages.loc[thread_messages['References'].apply(len).idxmax()]

        # Initialize the thread body with empty string        message_ids = [root['Message-ID']]  # Start with the root message ID

        # Iterate through the reference IDs to construct the body of the thread
        for msg_id in last_message['References']:
            # Safely retrieve the message, ensure there's at least one message that matches
            message = df[df['Message-ID'] == msg_id].iloc[0] if not df[df['Message-ID'] == msg_id].empty else None
            if message is not None:
                thread_body += ' ***NEW MESSAGE BY {sender}***: {body}'.format(sender=message['From'], body=message['Body'])
                message_ids.append(msg_id)  # Track the message ID

        # Append the last message body after all referenced messages
        thread_body += ' ***NEW MESSAGE BY {sender}***: {body} '.format(sender=last_message['From'], body=last_message['Body'])
        message_ids.append(last_message['Message-ID'])  # Include the last message ID

        # Append to the list of threads
        threads.append({'threadID': thread_id, 'threadSubject': thread_subject, 'Body': thread_body, 'Message-IDs': message_ids})
    else:
        # Handle the case where no messages reference the root
        thread_body += ' ***NEW MESSAGE BY {sender}***: {body}'.format(sender=root['From'], body=root['Body'])
        message_ids = [root['Message-ID']]  # Just the root message ID for standalone threads
        threads.append({'threadID': thread_id, 'threadSubject': thread_subject, 'Body': thread_body, 'Message-IDs': message_ids})

# Convert list to DataFrame
thread_df = pd.DataFrame(threads)

# Displaying the new DataFrame
print(thread_df.head(5))

# %%

import pandas as pd
import json
import openai
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize your OpenAI client
client = OpenAI(api_key='xxxx')

# Assuming 'df' is your DataFrame and it has 'threadID' and 'Body' columns
df = thread_df

system_prompt = "The following text represents an entire email thread that's in chronological order. It started with a ***SUBJECT: {subject}***. It then has different emails by different people, segmented by '***NEW MESSAGE BY {name} <email>***'. I want you to extract as many research/scientific related Q&A pairs as possible. You also want to have a context field that makes it so a new researcher, who is not involved in the Email thread, can read the Q&A and the context field, and then have an understanding of what is going on. It's good for answer to be a solid response to the question based on the email thread, and it'd be nice for for answer to include explanations. The context field is additionally provided to give more macro information about the whole thread. The question and the context field together can help this new researcher understand why this question is taking in place, and has a holistic understanding of the entire Email thread. Your output must be a list of maps (i.e. [{question: 'what is 1+1' answer': '2', context: 'PersonA has been asking PersonB about elementary math', questionFrom: 'daniel, Daniel@gmail.com', answerFrom: 'john, john@gmail.com'}...]. If there's no answer then just keep the answer field as empty string "". FOR Question, Answer, and Context Field, MAKE SURE TO REPLACE REAL NAMES WITH PERSON1, PERSON2... MOST IMPORTANT: REMINDER THAT YOUR REPLY MUST BE A LIST OF MAPS IN THAT FORMAT I DESCRIBED AND NOTHING ELSE."


def fix_json_string(json_like_string):
    # Step 1: Replace single quotes around keys with double quotes
    corrected_json = re.sub(r"\'([a-zA-Z0-9_]+)\'\s*:", r'"\1":', json_like_string)
    
    # Step 2: Add double quotes around unquoted keys
    corrected_json = re.sub(r'(?<!")([a-zA-Z0-9_]+)\s*:(?!")', r'"\1":', corrected_json)
    
    return corrected_json
    
def extract_qa_pairs(row):
    print('ok trying to run openai on this thread')
    thread_text = row['Body']
    extraction_prompt = thread_text
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extraction_prompt}
            ]
        )
        if completion.choices and completion.choices[0].message.content:
            return json.loads(completion.choices[0].message.content)
        else:
            # If no data is present, log or print the empty or invalid response
            print("Received an empty or invalid response from the API.")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        print(completion.choices[0].message.content)
        return []
        

def process_dataframe_concurrently(df):
    macro_json = []
    with ThreadPoolExecutor(max_workers=1000) as executor:
        # Submit all tasks to the executor
        future_to_row = {executor.submit(extract_qa_pairs, row): index for index, row in df.iterrows()}
        for future in as_completed(future_to_row):
            index = future_to_row[future]
            try:
                qa_pairs = future.result()
                macro_json.append({
                    'threadID': df.at[index, 'threadID'],
                    'QAPairs': qa_pairs
                })
                print(f'Completed thread {df.at[index, "threadID"]}')
            except Exception as e:
                print(f"Row {index} generated an exception: {e}")
    return macro_json

# Now process the DataFrame concurrently
results_json = process_dataframe_concurrently(df)

print("Q&A extraction completed and saved to output.json.")


def simplify_qa_data(macro_json):
    simplified_data = []
    missing_answer_count = 0  # Counter for missing 'answer' fields
    missing_answer_field = 0
    for thread in macro_json:
        for qa_pair in thread['QAPairs']:
            # Check if 'answer' field is missing
            if 'answer' not in qa_pair:
                missing_answer_field += 1
                continue
            if qa_pair['answer'] == "":
                print('bruh missing')
                missing_answer_count += 1
            # Try to construct the simplified QA pair and handle missing keys
            try:
                simplified_pair = {
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'context': qa_pair['context']
                }
                simplified_data.append(simplified_pair)
            except KeyError as e:
                # This block catches missing 'question' or 'context' fields
                print(f"KeyError: {e} in qa_pair: {qa_pair}")

    print(f"Missing 'answer' field count: {missing_answer_count}")
    print('missing_answer_field: ', missing_answer_field)
    return simplified_data

# Usage assuming 'results_json' is loaded
simplified_json = simplify_qa_data(results_json)

# Write the results to a file
with open('new_QA.json', 'w') as outfile:
    json.dump(simplified_json, outfile, indent=4)





