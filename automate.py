import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "C:\\Users\\Owner\\OneDrive\\Desktop\\Dan's project"

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

def clean_key(filename):
    name = filename.lower().replace('.csv', '').replace(' ', '_')
    if 'match' in name:
        return f"matches_{re.findall(r'\d+', name)[0]}"
    elif 'midterm' in name and 'mentee' in name:
        return f"mid_mentee_{re.findall(r'\d+', name)[0]}"
    elif 'midterm' in name and 'mentor' in name:
        return f"mid_mentor_{re.findall(r'\d+', name)[0]}"
    elif 'eop' in name and 'mentee' in name:
        return f"eop_mentee_{re.findall(r'\d+', name)[0]}"
    elif 'eop' in name and 'mentor' in name:
        return f"eop_mentor_{re.findall(r'\d+', name)[0]}"
    else:
        return name 

def load_csv_files(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {}
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df.columns = df.columns.str.strip() 
        key = clean_key(f)
        data[key] = df
    return data


def group_by_cohort(data_dict):
    grouped = {}
    for key, df in data_dict.items():
        match = re.search(r'(\d+)$', key)
        if match:
            cohort = match.group(1)
            grouped.setdefault(cohort, {})[key] = df
    return grouped

def merge_mentor_mentee(matches, mentees_data, mentors_sata):
    return(matches[['Mentor Name', 'Mentee Name']]
           .merge(mentees_data, left_on='Mentee Name', right_on='Your Name', how='left')
           .merge(mentors_sata, left_on='Mentor Name', right_on='Your Name', how='left', suffixes=('_mentee', '_mentor'))
           .drop(columns=['Your Name_mentee', 'Your Name_mentor']).fillna(-1))

def distributions(df, title_prefix = ''):
    mentee_cols = [col for col in df if '_mentee' in col and 'Timestamp' not in col]
    mentor_cols = [col for col in df if '_mentor' in col and 'Timestamp' not in col]

    mentee_df = df[['Mentee Name'] + mentee_cols].melt(id_vars = 'Mentee Name', var_name = 'question', value_name = 'response')
    mentee_df['role'] = 'mentee'
    mentee_df['question'] = mentee_df['question'].str.replace('_mentee', '', regex=False)

    mentor_df = df[['Mentor Name'] + mentor_cols].melt(id_vars = 'Mentor Name', var_name = 'question', value_name = 'response')
    mentor_df['role'] = 'mentor'
    mentor_df['question'] = mentor_df['question'].str.replace('_mentor', '', regex=False)

    new_df = pd.concat([mentee_df.rename(columns = {'Mentee Name' : 'name'}), mentor_df.rename(columns = {'Mentor Name': 'name'})], ignore_index=True)
    questions = new_df['question'].unique()
    n_cols = 3
    n_rows = -(-len(questions) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, q in enumerate(questions):
        ax = axes[i]
        temp = new_df[new_df['question'] == q].copy()
        try:
            temp['response'] = pd.to_numeric(temp['response'], errors='coerce')
            temp = temp.dropna(subset=['response'])
            sns.histplot(temp, x='response', hue='role', multiple='stack', ax=ax, kde=True)

        except:
            temp['response'] = temp['response'].astype(str).fillna('Missing')
            sns.countplot(temp, x='response', hue='role', ax=ax)

        ax.set_title(f"{title_prefix} - {q}")
        ax.set_xlabel('Response')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

all_data = load_csv_files(DATA_DIR)
print("✅ CSV files loaded:")
for key in all_data:
    print("-", key)
grouped_data = group_by_cohort(all_data)

for cohort, files in grouped_data.items():
    try:
        matches = files[f'matches_{cohort}']
        mid_mentees = files[f'mid_mentee_{cohort}']
        eop_mentees = files[f'eop_mentee_{cohort}']
        mid_mentors = files[f'mid_mentor_{cohort}']
        eop_mentors = files[f'eop_mentor_{cohort}']
        mid_merged = merge_mentor_mentee(matches, mid_mentees, mid_mentors)
        eop_merged = merge_mentor_mentee(matches, eop_mentees, eop_mentors)
        merged = pd.concat([mid_merged, eop_merged], ignore_index=True)
        cols_to_drop = [col for col in merged.columns if any(x in col for x in ['Satisfaction', 'Suggestion', 'Impact'])]
        cleaned = merged.drop(columns=cols_to_drop, errors='ignore')
        distributions(cleaned, title_prefix=f"Cohort {cohort}")
    except KeyError as e:
        print(f"Missing data for cohort {cohort}: {e}")