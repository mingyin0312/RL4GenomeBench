#######--------------------------
## This script will use Converted_MCQs_gpt4o_full.json file
## as the input and convert it to MCQs_formatted.json,
## where answer key has the format 
## <explanation>{original expert explanation}</explanation> <answer>{correct option}</answer>.
## It also concatenates 'context' with 'question' in question field.
#######--------------------------


import json

# Load the original dataset
with open("Converted_MCQs_gpt4o_full.json", "r") as file:
    data = json.load(file)

# Function to format the answer with step-by-step reasoning (dummy example)
def format_answer(original_answer, final_answer):
    return f"<explanation>{original_answer}</explanation> <answer>{final_answer}</answer>"

# Convert to GENOME-BENCH format
converted_data = []
for entry in data:
    question = entry["Questions with options"]
    reasoning = entry.get("Original answer", "The answer is derived from the given choices.")  # Placeholder reasoning
    final_answer = entry.get("Answer key"),
    context = entry["Context"]


    # Ensure proper formatting
    formatted_answer = format_answer(reasoning, final_answer[0])

    converted_data.append({
        "question": 'Question context: ' + context + ' Question: ' + question,
        "answer": formatted_answer
    })

# Save the new dataset
with open("MCQs_formatted.json", "w") as outfile:
    json.dump(converted_data, outfile, indent=4)

print("Dataset reformatted to GENOME-BENCH format and saved as MCQs_formatted.json")