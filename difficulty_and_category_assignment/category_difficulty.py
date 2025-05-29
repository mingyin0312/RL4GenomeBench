import json
import os
from llm import LLMChat

# Load the test data
def load_test_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Prompt templates for category and difficulty classification
CATEGORY_PROMPT = """
You are a CRISPR expert assistant. I will give you a CRISPR-related question and answer, and you will assign it to exactly one of the following seven predefined categories based on its core intent. Be precise—choose the category that most closely reflects the primary focus of the question.
Here are the categories:
1. Gene-editing Enzyme Selection
2. GuideRNA Design
3. Cloning & Plasmid Construction
4. Gene-editing Delivery Methods
5. CRISPR Screening & Library Workflows
6. Validation, Troubleshooting & Optimization
7. Practical Considerations & Lab Logistics

For each question, provide your category assignment with a brief (1-2 sentence) explanation of why you selected that category.

Question: {question}
Answer: {answer}

Please format your response following this response format and make sure it is parsable by JSON:
{
    "category": <category>,  # Category name
    "reason": <reason>       # Brief statement on why you picked
}
"""

DIFFICULTY_PROMPT = """
You are a CRISPR expert assistant. I will give you a CRISPR-related question and answer, and you will assign it to exactly one of the following three predefined difficulty levels.

Here are the categories:
1. Easy
2. Medium
3. Hard

For each question, provide your difficulty assignment with a brief (1-2 sentence) explanation of why you selected that level.

Question: {question}
Answer: {answer}

Please format your response following this response format and make sure it is parsable by JSON:
{
    "difficulty": <difficulty>,  # Difficulty name
    "reason": <reason>           # Brief statement on why you picked
}
"""

def main():
    # Path to the test data file
    test_data_path = "MCQs_Genome-Bench-evaluation.json"
    
    # Load test data
    data = load_test_data(test_data_path)
    
    # Process each entry
    results = []
    
    # Variables to track stats
    category_counts = {}
    difficulty_counts = {}
    category_mismatches = 0
    difficulty_mismatches = 0
    entries_with_original_category = 0
    entries_with_original_difficulty = 0
    
    for i, entry in enumerate(data):
        print(f"Processing entry {i+1}/{len(data)}: ID {entry['id']}")
        
        # Extract question and answer
        question = entry['question']
        answer = entry['answer']
        entry_id = entry['id']
        
        # Store original values if they exist
        original_category = entry.get('question type')
        original_difficulty = entry.get('difficulty')
        
        if original_category:
            entries_with_original_category += 1
        if original_difficulty:
            entries_with_original_difficulty += 1
        
        # Generate new category
        try:
            category_prompt = CATEGORY_PROMPT.format(question=question, answer=answer)
            category_response = LLMChat.chat(category_prompt, model_name="gpt4o")
            category = category_response.get('category')
            print(f"  - Generated category: {category}")
            
            if original_category and category != original_category:
                category_mismatches += 1
                print(f"  - MISMATCH: Original category was '{original_category}'")
        except Exception as e:
            print(f"  - Error generating category: {str(e)}")
            category = None
        
        # Generate new difficulty
        try:
            difficulty_prompt = DIFFICULTY_PROMPT.format(question=question, answer=answer)
            difficulty_response = LLMChat.chat(difficulty_prompt, model_name="gpt4o")
            difficulty = difficulty_response.get('difficulty')
            print(f"  - Generated difficulty: {difficulty}")
            
            if original_difficulty and difficulty != original_difficulty:
                difficulty_mismatches += 1
                print(f"  - MISMATCH: Original difficulty was '{original_difficulty}'")
        except Exception as e:
            print(f"  - Error generating difficulty: {str(e)}")
            difficulty = None
        
        # Update counts
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if difficulty:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Add to results
        results.append({
            'id': entry_id,
            'question': question,
            'answer': answer,
            'category': category,
            'difficulty': difficulty
        })
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nCategory Counts:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")
    
    print("\nDifficulty Counts:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff}: {count}")
    

    # Save results to a new file
    output_file = "Genome-Bench-evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} entries and saved to {output_file}")

if __name__ == "__main__":
    main()