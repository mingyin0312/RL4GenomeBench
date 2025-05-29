####-----------------------------------------------------------------
#### Converting "Question + Option" questions to natural questions
####-----------------------------------------------------------------

import json
import openai
from tqdm import tqdm

# Set your OpenAI API key
openai.api_key = "xxxx"

# Load the JSON file
input_file = "MCQs_formatted.json" 
output_file = "MCQs_Genome-Bench.json"  

with open(input_file, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

# Function to transform structured question into a natural-sounding question
def generate_natural_question(question_context, question_text):
    prompt = (f"Convert the following structured question into a natural-sounding question while preserving the "
              f"maximal amount of information:\n\n"
              f"Context: {question_context}\n"
              f"Question: {question_text}\n\n"
              f"The output should be a single natural question that integrates the context smoothly without phrases like "
              f"'Person A' or 'Person B'.")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an AI that rewrites structured questions into natural-sounding questions while preserving information."},
                      {"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
       
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating question: {e}")
        return None


# Process each question in the JSON file
processed_questions = []
for item in tqdm(questions_data, desc="Processing Questions", unit="question"):
    structured_question = item["question"]
    
    # Extract context and question text
    try:
        context_start = structured_question.find("Question context: ") + len("Question context: ")
        
        question_start = structured_question.find("Question: ")
        
        context_text = structured_question[context_start:question_start].strip()
        
        question_text = structured_question[question_start + len("Question: "):].strip()
        

        # Generate natural question
        natural_question = generate_natural_question(context_text, question_text)
        if natural_question:
            processed_questions.append({"natural_question": natural_question})
    except Exception as e:
        print(f"Error processing question: {e}")

# Save the transformed questions to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_questions, f, indent=4, ensure_ascii=False)

print(f"Processed questions saved to {output_file}")