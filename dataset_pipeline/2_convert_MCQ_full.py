#######--------------------------
## This script will use new_QA.json file
## as the input and use gpt-4o to convert it to 
## multiple-choice question data named 
## Converted_MCQs_gpt4o_full.json
#######--------------------------

import json
import openai
from tqdm import tqdm

# Set up your OpenAI API key
openai.api_key = "xxxx"
# Load the MCQ examples for prompting
with open("MCQ_updated.json", "r") as f:
    mcq_examples = json.load(f)["MCQ_Updated"]

# Load the QA data
with open("new_QA.json", "r") as f:
    final_qa_data = json.load(f)

# Prepare the examples prompt
example_prompts = "\n\n".join(
    f"Example MCQ:\n{mcq['Questions with options']}\nCorrect Answer: {mcq['Key']}"
    for mcq in mcq_examples#
)

# Initialize a list to store generated MCQs
generated_mcqs = []

# Process each question in the QA data
for item in tqdm(final_qa_data, total = len(final_qa_data)):
   
    question = item["question"]
    answer = item["answer"]
    context = item["context"]

    if answer.strip() == "":
        continue
    else:
        # Prepare the prompt
        prompt = f"""
        Below are examples of multiple-choice questions (MCQs) with their formats. Use them to generate a new Single-Choice MCQ based on the provided question and context. The MCQ should include five answer choices (a-e) with one being the correct answer, and it should be identical as provided, following the similar format as the examples. The response should not include anything else except for the generated multiple-choice question itself. Just provide one version, then end the response and don't repeat.

        {example_prompts}

        New Question:
        Question: {question}
        Answer: {answer}
        Context: {context}

        Do not modify the original question in the new MCQ.

        Generate a new MCQ:
        """
        #If the above question is not clear, you can also modify the question by using 'context' information for new questions. However, do not use 'answer' information for it! 
        
        # Generate MCQ using the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an gene-editing expert generating formatted single-choice multiple-choice questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content

        # Process the response to separate options and answer key
        index = generated_text.find("Correct Answer:")
        if index != -1:
            options = generated_text[:index].strip()
            answer_key = generated_text[index+16:].strip()
        else:
            options = generated_text
            answer_key = "N/A"  # Default if no answer key is found

        # Store the result
        generated_mcqs.append({
            "Questions with options": question + " " + options,
            "Answer key": answer_key,
            "Original answer": answer,
            "Context": context
        })


# Save the generated MCQs to a new JSON file
output_file = "Converted_MCQs_gpt4o_full.json"
with open(output_file, "w") as f:
    json.dump(generated_mcqs, f, indent=4)

print(f"Generated MCQs saved to {output_file}")