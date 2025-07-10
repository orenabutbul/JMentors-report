# app.py
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_object_dtype
import automate

st.title("📊 Mentor-Mentee Program Dashboard")

uploaded_files = st.file_uploader(
    "Upload all CSV files (matches, mentees, mentors)", 
    type="csv", 
    accept_multiple_files=True
)

@st.cache_data
def preprocess_all(grouped_data):
    processed = {}
    for cohort, files in grouped_data.items():
        try:
            matches = automate.preprocess_matches(files[f'matches_{cohort}'])
            mid_mentors = automate.preprocess_df(files[f'mid_mentor_{cohort}'])
            mid_mentees = automate.preprocess_df(files[f'mid_mentee_{cohort}'])
            eop_mentors = automate.preprocess_df(files.get(f'eop_mentor_{cohort}', pd.DataFrame()))
            eop_mentees = automate.preprocess_df(files.get(f'eop_mentee_{cohort}', pd.DataFrame()))
            processed[cohort] = {
                'matches': matches,
                'mid_mentors': mid_mentors,
                'mid_mentees': mid_mentees,
                'eop_mentors': eop_mentors,
                'eop_mentees': eop_mentees,
            }
        except Exception as e:
            st.warning(f"Failed to preprocess cohort {cohort}: {e}")
    return processed

if uploaded_files:
    all_data = automate.load_uploaded_files(uploaded_files) 
    grouped_data = automate.group_by_cohort(all_data)

    preprocessed_data = preprocess_all(grouped_data)

    # Cohort selection
    cohort_options = sorted(preprocessed_data.keys())
    selected_cohort = st.selectbox("Choose a cohort:", cohort_options)

    # Pull data for selected cohort
    files = preprocessed_data[selected_cohort]
    matches = files['matches']
    mid_mentors = files['mid_mentors']
    mid_mentees = files['mid_mentees']
    eop_mentors = files['eop_mentors']
    eop_mentees = files['eop_mentees']

    # Merge and compute trend data
    mid_merged = automate.merge_mentor_mentee(mid_mentors, mid_mentees, matches)
    eop_merged = automate.merge_mentor_mentee(eop_mentors, eop_mentees, matches)

    trend_records = []
    automate.trend_data(trend_records, mid_merged, selected_cohort, 'mid')
    automate.trend_data(trend_records, eop_merged, selected_cohort, 'eop')

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Participation", 
        "📍 Overall Mentors-Mentees Comparisons", 
        "📊 Mentor vs Mentee per Question", 
        "📦 Mentor-Mentee Word Answers", 
        "📈 Trends"
    ])

    with tab1:
        st.header("Participation")
        st.subheader("Midpoint Participation")
        automate.participation(mid_merged)
        st.subheader("End of Program Participation")
        automate.participation(eop_merged)

    with tab2:
        st.header("Overall Mentors-Mentees Comparisons")
        st.subheader("Midpoint Average Scores")
        automate.ave_mentor_mentee(mid_mentors, mid_mentees, cohort=selected_cohort)
        automate.survey_avg(mid_mentors, mid_mentees, cohort=selected_cohort)

        st.subheader("End of Program Average Scores")
        automate.ave_mentor_mentee(eop_mentors, eop_mentees, cohort=selected_cohort)
        automate.survey_avg(eop_mentors, eop_mentees, cohort=selected_cohort)

    with tab3:
        st.header("Mentor vs Mentee per Question")
        st.subheader("Midpoint Comparisons")
        automate.bar_graph(mid_merged)
        st.subheader("End of Program Comparisons")
        automate.bar_graph(eop_merged)

    with tab4:
        st.header("Categorical Response Distribution")
        st.subheader("Midpoint Categorical Responses")
        automate.plot_cat(mid_merged)
        st.subheader("End of Program Categorical Responses")
        automate.plot_cat(eop_merged)

    with tab5:
        st.header("Trends Across Cohorts")
        all_trend_records = []

        for cohort, files in preprocessed_data.items():
            try:
                mid_merged = automate.merge_mentor_mentee(files['mid_mentors'], files['mid_mentees'], files['matches'])
                eop_merged = automate.merge_mentor_mentee(files['eop_mentors'], files['eop_mentees'], files['matches'])

                automate.trend_data(all_trend_records, mid_merged, cohort, 'mid')
                if not eop_merged.empty:
                    automate.trend_data(all_trend_records, eop_merged, cohort, 'eop')
            except Exception as e:
                st.warning(f"Trend error in cohort {cohort}: {e}")

        if all_trend_records:
            trend_df = pd.DataFrame(all_trend_records)
            automate.plot_trends(trend_df)
        else:
            st.info("No trend data available.")
else:
    st.info("Please upload the CSV files to view the dashboard.")