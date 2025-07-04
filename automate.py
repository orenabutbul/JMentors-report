import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pandas.api.types import is_numeric_dtype, is_object_dtype

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

def preprocess_df(df):
    cols_to_drop = [0,11] + list(range(13, len(df.columns)))
    df.drop(df.columns[cols_to_drop], axis=1, inplace=True, errors='ignore')
    df.columns = ['Your Name', 
                  'Rate your experience', 
                  'Would you recommend the program?', 
                  'Would you consider serving as a mentor?', 
                  'Were the articles helpful?', 
                  'How often did you meet?', 
                  'Did you meet enough?', 
                  'Describe your relationship',
                  'Comfortable sharing information?',
                  'Valuable experience?',  
                  'Program impact?']
    df.columns = df.columns.str.lower().str.strip()
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    name_col = df.columns[0]
    df[name_col] = df[name_col].str.lower()    
    return df

def preprocess_matches(df):
    df = df[['Mentor Name', 'Mentee Name']].copy()
    df['Mentor Name'] = df['Mentor Name'].str.strip().str.lower()
    df['Mentee Name'] = df['Mentee Name'].str.strip().str.lower()
    df.columns = ['mentor_name', 'mentee_name']
    df.dropna(how = 'all', inplace=True)
    return df

def merge_mentor_mentee(mentor_df, mentee_df, matches):
    merged = pd.merge(matches, mentor_df, left_on='mentor_name', right_on='your name', how='outer')
    merged = pd.merge(merged, mentee_df, left_on='mentee_name', right_on='your name', how='outer', suffixes=('_mentor', '_mentee'))
    merged.drop(columns=['your name_mentor', 'your name_mentee'], inplace=True, errors='ignore')
    return merged

def participation(merged, base = 'rate your experience'):
    mentor_col = f'{base}_mentor'
    mentee_col = f'{base}_mentee'

    mentor_missing = merged[mentor_col].isna().sum()
    mentor_total = len(merged)
    mentor_participated = mentor_total - mentor_missing

    mentee_missing = merged[mentee_col].isna().sum()
    mentee_total = len(merged)
    mentee_participated = mentee_total - mentee_missing
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].pie([mentor_participated, mentor_missing],
               labels=['Participated', 'Missing'],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#66c2a5', '#fc8d62'])
    axs[0].set_title('Mentor Participation Rate')
    axs[0].axis('equal')

    axs[1].pie([mentee_participated, mentee_missing],
               labels=['Participated', 'Missing'],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#8da0cb', '#fc8d62'])
    axs[1].set_title('Mentee Participation Rate')
    axs[1].axis('equal')

    plt.tight_layout()
    plt.show()

def ave_mentor_mentee(mentors, mentees , cohort = ""):
    mentors_mean = mentors.mean(numeric_only = True)
    mentees_mean = mentees.mean(numeric_only = True)
    plot_df = pd.DataFrame({'mentors_avg': mentors_mean.values,
                           'mentees_avg': mentees_mean.values,
                           'question': mentors_mean.index})
    plt.figure(figsize = (10,6))
    sns.scatterplot(data= plot_df, x = 'mentors_avg', y = 'mentees_avg', hue = 'question')
    max_val = max(plot_df[['mentors_avg', 'mentees_avg']].max())
    plt.plot([0, max_val], [0, max_val], '--', color = 'gray')

    plt.title(f'Mentor vs Mentee Averages {cohort}')
    plt.xlabel('Mentor Average Score')
    plt.ylabel('Mentee Average Score')
    plt.axis('equal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def bar_graph(merged):
    mentor_names = merged['mentor_name'].combine_first(merged['your name_mentor']).fillna('Unknown Mentor')
    mentee_names = merged['mentee_name'].combine_first(merged['your name_mentee']).fillna('Unknown Mentee')
    pair_label = mentor_names + ' - ' + mentee_names
    questions = []
    for col in merged.columns:
        if col.endswith('_mentor') and is_numeric_dtype(merged[col]):
            base = col.replace('_mentor', '')
            if base not in questions:
                questions.append(base)
    n_questions = len(questions)
    n_cols = 2
    n_rows = math.ceil(n_questions/ n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize = (20, 8 * n_rows))
    axs = axs.flatten()
    for i,q in enumerate(questions):
        ax = axs[i]
        mentor_col = f'{q}_mentor'
        mentee_col = f'{q}_mentee'
        mentor_vals = pd.to_numeric(merged[mentor_col], errors='coerce')
        mentee_vals = pd.to_numeric(merged[mentee_col], errors='coerce')

        plot_df = pd.DataFrame({'pair': pair_label,
                               'mentor': mentor_vals,
                               'mentee': mentee_vals}).dropna(how='all', subset=['mentor','mentee'])
        plot_df.set_index('pair')[['mentor', 'mentee']].plot(kind='bar', ax=ax, color = ['#66c2a5', '#fc8d62'])

        ax.set_title(q.replace('_', ' ').capitalize())
        ax.set_ylabel('Score')
        ax.set_xlabel('Mentor - Mentee Pair')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        max_val = pd.concat([mentor_vals, mentee_vals], axis = 0).max()
        if max_val > 5:
            ax.set_ylim(0, 10)
        else:
            ax.set_ylim(0,5)
    for i in range(n_questions, n_rows*n_cols):
        row, col =divmod(i, n_cols)
        axs[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def plot_cat(merged):
    questions = []
    for col in merged.columns:
        if col.endswith('_mentor') and is_object_dtype(merged[col]):
            base = col.replace('_mentor', '')
            if base.lower() != 'your name' and base not in questions:
                questions.append(base)
    sns.set(style='whitegrid')
    n=len(questions)
    fig, axs = plt.subplots(n, 1, figsize=(10, 5*n))
    for i, q in enumerate(questions):
        mentor_col = f'{q}_mentor'
        mentee_col = f'{q}_mentee'
        mentor_counts = merged[mentor_col].value_counts().sort_index()
        mentee_counts = merged[mentee_col].value_counts().sort_index()
        all_options = sorted(set(mentor_counts.index).union(mentee_counts.index))
        mentor_counts = mentor_counts.reindex(all_options, fill_value=0)
        mentee_counts = mentee_counts.reindex(all_options, fill_value=0)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Response': all_options,
            'Mentors': mentor_counts.values,
            'Mentees': mentee_counts.values
        })

        plot_df.set_index('Response')[['Mentors', 'Mentees']].plot(
            kind='bar', ax=axs[i], color=['#66c2a5', '#fc8d62']
        )

        axs[i].set_title(q.replace('_', ' ').capitalize())
        axs[i].set_ylabel("Count")
        axs[i].set_xlabel("Response")
        axs[i].legend()
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=44)

    plt.tight_layout()
    plt.show()

def trend_data(records, merged, cohort, timepoint):
    questions = [
        'rate your experience',
        'would you recommend the program?',
        'valuable experience?',
        'program impact?'
    ]
    for q in questions:
        mentor_col = f'{q}_mentor'
        mentee_col = f'{q}_mentee'
        
        merged[mentor_col] = pd.to_numeric(merged[mentor_col], errors='coerce')
        merged[mentee_col] = pd.to_numeric(merged[mentee_col], errors='coerce') 
        
        records.append({
            'cohort': int(cohort),
            'time': timepoint,
            'question': q,
            'group': 'mentor',
            'mean_score': merged[mentor_col].mean(),
            'cohort_time': f"{cohort}_{timepoint}"
        })
        records.append({
            'cohort': int(cohort),
            'time': timepoint,
            'question': q,
            'group': 'mentee',
            'mean_score': merged[mentee_col].mean(),
            'cohort_time': f"{cohort}_{timepoint}"
        })

def plot_trends(trend_df):
    trend_df['sort_key'] = trend_df['cohort'].astype(int) * 10 + trend_df['time'].map({'mid': 0, 'eop': 1})
    trend_df = trend_df.sort_values('sort_key')
    trend_df['cohort_time'] = trend_df['cohort'].astype(str) + '_' + trend_df['time']
    for q in trend_df['question'].unique():
        plt.figure(figsize=(10, 6))
        data = trend_df[trend_df['question'] == q]
        sns.lineplot(
            data=data,
            x='cohort_time',
            y='mean_score',
            hue='group',
            markers=True,
            style='group',
            dashes=False
        )
        plt.title(f"Trend for '{q.replace('_', ' ').capitalize()}' over time")
        plt.xlabel("Cohort")
        plt.ylabel("Average Score")
        plt.ylim(0, 10)
        plt.legend(title='Group & Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

all_data = load_csv_files(DATA_DIR)
grouped_data = group_by_cohort(all_data)

trend_records = []

for cohort, files in grouped_data.items():
    try:
        matches = preprocess_matches(files[f'matches_{cohort}'])
        mid_mentees = preprocess_df(files[f'mid_mentee_{cohort}'])
        eop_mentees = preprocess_df(files[f'eop_mentee_{cohort}'])
        mid_mentors = preprocess_df(files[f'mid_mentor_{cohort}'])
        eop_mentors = preprocess_df(files[f'eop_mentor_{cohort}'])
        mid_merged = merge_mentor_mentee(mid_mentors, mid_mentees, matches)
        eop_merged = merge_mentor_mentee(eop_mentors, eop_mentees, matches)

        trend_data(trend_records, mid_merged, cohort, 'mid')
        trend_data(trend_records, eop_merged, cohort, 'eop')
    except KeyError as e:
        print(f"Missing data for cohort {cohort}: {e}")

trend_records = pd.DataFrame(trend_records)
plot_trends(trend_records)
