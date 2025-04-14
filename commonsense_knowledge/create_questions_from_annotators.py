# First we generate the new questions from the distractors provided by human annotators
import pandas as pd 
import matplotlib.pyplot as plt
import random

# --------------------------------------
def plot_distribution(df):
    # count group by the question id
    df['question_id'].value_counts().value_counts()

    # Get the frequency of how many times each question_id appears
    question_counts = df['question_id'].value_counts()

    # Count how many question_ids appear 4 times, 5 times, etc.
    distribution = question_counts.value_counts().sort_index()

    # Plot
    plt.figure(figsize=(6, 4))
    distribution.plot(kind='bar')
    plt.title('Distribution of the annotations')
    plt.xlabel('Number of human vadidations')
    plt.ylabel('Number of question_ids')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --------------------------------------
def process_user_evaluation_from_df(df, df_questions):
    distractors_per_id = {}
    df = df[
        df['suggested_ingredient_a'].notnull() &
        df['suggested_ingredient_b'].notnull() &
        df['suggested_ingredient_c'].notnull()
    ]
    
    question_set = []

    for _, row in df.iterrows():
        question_text = df_questions[df_questions['question_id'] == row['question_id']]['question_text'].values[0]

        possible_answers = [
            row['suggested_ingredient_a'],
            row['suggested_ingredient_b'],
            row['suggested_ingredient_c'],
            row['correct_answer']
        ]

        random.shuffle(possible_answers)

        shuffled_answers = {
            'answer_a': possible_answers[0].lower(),
            'answer_b': possible_answers[1].lower(),
            'answer_c': possible_answers[2].lower(),
            'answer_d': possible_answers[3].lower(),
            'correct_answer': row['correct_answer'].lower(),
            'correct_letter': ['A', 'B', 'C', 'D'][possible_answers.index(row['correct_answer'])]
        }

        question_set.append({
            'question_id': row['question_id'],
            'question_text': question_text,
            'origin': row['origin'],
            **shuffled_answers
        })

        # Contribute to the distractor lists per question_id 
        
        if row['question_id'] not in distractors_per_id:
            distractors_per_id[row['question_id']] = []

        if pd.notnull(row['suggested_ingredient_a']):
            distractors_per_id[row['question_id']].append(row['suggested_ingredient_a'])
        if pd.notnull(row['suggested_ingredient_b']):
            distractors_per_id[row['question_id']].append(row['suggested_ingredient_b'])
        if pd.notnull(row['suggested_ingredient_c']):
            distractors_per_id[row['question_id']].append(row['suggested_ingredient_c'])

    # remove duplicates in the distractors
    for question_id in distractors_per_id:
        distractors_per_id[question_id] = list(set(distractors_per_id[question_id]))
    return pd.DataFrame(question_set), distractors_per_id

# --------------------------------------

# to validate the question, it has to have three nos
def count_no_answers(row):
    return sum([
        row['eval_answer_a'] == 'no',
        row['eval_answer_b'] == 'no',
        row['eval_answer_c'] == 'no',
        row['eval_answer_d'] == 'no',
    ])

# --------------------------------------

def replace_incorrect_answers(row, ingredient_dict):
    question_id = row['question_id']
    
    
    if question_id not in ingredient_dict or row['question_type'] != 'main ingredient':
        row['was_updated'] = False
        return row

    current_answers = {
        'A': row['answer_a'],
        'B': row['answer_b'],
        'C': row['answer_c'],
        'D': row['answer_d'],
    }

    correct_letter = row['correct_letter']
    used_ingredients = set(current_answers.values())
    available_ingredients = [i for i in ingredient_dict[question_id] if i not in used_ingredients]

    updated = False  

    print(question_id)
    for letter in current_answers:
        eval_col = 'eval_answer_' + letter.lower()
        if letter != correct_letter and row.get(eval_col) == 'yes':
            if not available_ingredients:
                row['was_updated'] = False
                return row
            
            new_ingredient = random.choice(available_ingredients)
            available_ingredients.remove(new_ingredient)
            current_answers[letter] = new_ingredient
            row[eval_col] = 'no'
            updated = True

    # Actualizar respuestas
    row['answer_a'] = current_answers['A']
    row['answer_b'] = current_answers['B']
    row['answer_c'] = current_answers['C']
    row['answer_d'] = current_answers['D']
    row['was_updated'] = updated  

    return row

# --------------------------------------

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Load merged questions file (assuming it covers all questions)
    df_questions = pd.read_csv("../data/annotation_platform/myapp_question.csv")
    df_original_valid_evaluations = pd.read_csv('../data/annotation_platform/just_validated_questions_original.csv')
    df_current_evaluations = pd.read_csv('../data/annotation_platform/myapp_userevaluation_new.csv')
    df_original_distractors = pd.read_csv('../data/annotation_platform/myapp_userevaluation_original.csv')
    df_human_distractors = pd.concat([df_original_distractors, df_current_evaluations], axis=0)

    print(df_current_evaluations.shape, "Number of distractor annotations in the current platform")
    print(df_original_distractors.shape, "Number of distractor annotations in the original platform")
    print(df_human_distractors.shape, "Number of distractor annotations after pd.concat")

    # --- DISTRATORS ---
    # Get the questions using the distractors provided by annotators 
    df_human_distractors = df_human_distractors.drop_duplicates() # remove duplicates

    # in dict_distractors we have the distractors per question_id for fixing the questions that are not correct (from the human validation)
    df_processed, dict_distractors = process_user_evaluation_from_df(df_human_distractors, df_questions) # Now apply the transformation function
    df_processed = df_processed.sort_values(by='question_id').reset_index(drop=True) # Sort and display

    # Save the processed DataFrame to a CSV file
    df_processed.to_csv('../data/annotation_platform/questions_human_distractors.csv', index=False)


    # --- VALIDATIONS ---
    # study the validated questions
    print(df_original_valid_evaluations.shape, "Number of evaluations in the original platform")
    print(df_current_evaluations.shape, "Number of evaluations in the current platform")

    # concat 
    df_evaluations = pd.concat([df_original_valid_evaluations, df_current_evaluations], axis=0)
    print(df_evaluations.shape)

    plot_distribution(df_evaluations)


    # Add column with count of "no"s per row
    df_evaluations['num_no_answers'] = df_evaluations.apply(count_no_answers, axis=1)
    # Now group by question_id and collect the 'num_no_answers' into lists
    no_counts_per_question = df_evaluations.groupby('question_id')['num_no_answers'].apply(list)

    # --------------------------------- CORRECT QUESTIONS ----------------------------------
    # Count how many times the number 3 appears per question
    correct_questions = (
        df_evaluations.groupby('question_id')['num_no_answers']
        .apply(lambda x: (x == 3).sum())
        .loc[lambda s: s >= 2]
    )

    # Number of such questions
    num_correct_questions = len(correct_questions)
    print("A total of", num_correct_questions, "questions are correct.")

    # save the correct questions
    correct_questions = correct_questions.index.tolist()
    # Example: filter rows belonging to correct questions
    df_correct = df_questions[df_questions['question_id'].isin(correct_questions)]
    # Save the DataFrame to a CSV file
    df_correct.to_csv('../data/annotation_platform/questions_correct.csv', index=False)

    # --------------------------------- INCORRECT QUESTIONS ----------------------------------
    # Count how many times the number is NOT 3 per question
    incorrect_questions = (
        df_evaluations.groupby('question_id')['num_no_answers']
        .apply(lambda x: (x != 3).sum())
        .loc[lambda s: s >= 2]
    )

    # Number of such questions
    num_incorrect_questions = len(incorrect_questions)
    print("A total of", num_incorrect_questions, "questions are incorrect.")

    incorrect_questions
    incorrect_question_ids = incorrect_questions.index.tolist()
    # Example: filter rows belonging to incorrect questions
    df_incorrect = df_evaluations[df_evaluations['question_id'].isin(incorrect_question_ids)]
    incorrect_question_ids = list(set(df_incorrect['question_id']))


    # Fix the incorrect questions by sampling incorrect answers from the distractors provided by human annotators
    df_evaluations_fixed = df_evaluations.copy()
    df_evaluations_fixed = df_evaluations_fixed[df_evaluations_fixed['question_id'].isin(incorrect_question_ids)]
    df_evaluations_fixed = df_evaluations_fixed.apply(
        lambda row: replace_incorrect_answers(row, dict_distractors), axis=1
    )

    df_evaluations_fixed = df_evaluations_fixed.drop_duplicates(subset=['question_id'])

    df_evaluations_fixed.to_csv('../data/annotation_platform/questions_fixed.csv', index=False)