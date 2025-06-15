import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime

# Create data folder
DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# File paths
CONSENT_CSV = os.path.join(DATA_FOLDER, "consent_data.csv")
DEMOGRAPHIC_CSV = os.path.join(DATA_FOLDER, "demographic_data.csv")
TASK_CSV = os.path.join(DATA_FOLDER, "task_data.csv")
EXIT_CSV = os.path.join(DATA_FOLDER, "exit_data.csv")


def save_to_csv(data_dict, csv_file):
    df_new = pd.DataFrame([data_dict])
    if not os.path.isfile(csv_file):
        df_new.to_csv(csv_file, mode='w', header=True, index=False)
    else:
        df_new.to_csv(csv_file, mode='a', header=False, index=False)


def load_from_csv(csv_file):
    if os.path.isfile(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()


def main():
    st.title("Usability Testing Tool")

    home, consent, demographics, tasks, exit, collected_data, report = st.tabs(
        ["Home", "Consent", "Demographics", "Tasks", "Exit Questionnaire", "Collected Data", "Report"]
    )

    with home:
        st.header("Introduction")
        st.write("""
            Welcome to the Usability Testing Tool for the StockWatch application. 

            Stockwatch provides a complete stock trading and portfolio management interface with real-time data integration, user location mapping, and interactive visualization.

            In this test, you will:

            1. Provide consent.
            2. Fill out demographic information.
            3. Complete 3 usability tasks:
                - **Record a Transaction**
                - **View Live Market Updates**
                - **Find  Today's News**
            4. Provide final feedback.

            Your input helps us improve usability. Thank you!
        """)

    with consent:
        st.header("Consent Form")
        st.write("Please read the consent for below and confirm your agreement.")
        st.write("Content Agreement:")
        st.markdown("- I understand the purpose of this usability test.")
        st.markdown("- I am aware that my data will be collected solely for research and improvement purposes.")
        st.markdown("- I can withdraw at any time")

        consent_given = st.checkbox(
            "I agree to the terms above.", key="consent_checkbox"
        )

        if st.button("Submit Consent"):
            if not consent_given:
                st.warning("You must agree to the consent terms before proceeding.")
            else:
                # Save the consent acceptance time
                data_dict = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "consent_given": consent_given
                }
                save_to_csv(data_dict, CONSENT_CSV)
                st.success("Consent recorded successfully! Please proceed to the Demographics tab.")

    with demographics:
        st.header("Demographic Questionnaire")

        st.markdown(
            "Please provide some basic information about yourself. This helps us understand our participant pool.")

        with st.form("demographic_form"):

            name = st.text_input("Full Name", placeholder="Enter your full name")
            age = st.selectbox("Age Range", [
                "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
            ])
            occupation = st.text_input("Occupation", placeholder="Enter your occupation")
            familiarity = st.radio("Have you used StockWatch before?", ["Yes", "No"])

            submitted = st.form_submit_button("Submit Demographics")
            if submitted:
                if not name or not occupation or not age or not familiarity:
                    st.error("Please fill in all fields.")
                else:
                    data_dict = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "name": name,
                        "age": age,
                        "occupation": occupation,
                        "familiarity": familiarity
                    }
                    save_to_csv(data_dict, DEMOGRAPHIC_CSV)
                    st.success("Demographics saved successfully! Please proceed to the Tasks tab.")

    with tasks:
        st.header("Usability Tasks")

        tasks_list = [
            {
                "name": "Record a Transaction",
                "description": [
                    "Navigate to the Trade tab",
                    "Select a stock (preferably one that you do not hold)",
                    "Enter the quantity to buy (e.g. 100)",
                    "Complete the buy",
                    "Validate order in Transactions tab"
                ]
            },
            {
                "name": "View Live Market Updates",
                "description": [
                    "Navigate to the Market tab",
                    "Take note of the Market Overview values",
                    "Select a few stocks to view details",
                    "Validate data is updated in real-time",
                    "Refresh page and validate Market Overview data is updated in real-time"
                ]
            },
            {
                "name": "Find Today's News",
                "description": [
                    "Navigate to the News tab",
                    "Browse through News links",
                    "Use filters to find CNBC or Marketwatch news in the last 24-hours",
                    "Validate correct news links are displayed"
                ]
            }
        ]

        for task in tasks_list:
            st.subheader(task["name"])
            # st.markdown(f"**Description:** {task['description']}")
            for step in task["description"]:
                st.markdown(f" - {step}")

            if st.button(f"Start: {task['name']}"):
                st.session_state[f"{task['name']}_start"] = time.time()
                st.info("Task started. Complete the task in StockWatch, then return here.")

            if f"{task['name']}_start" in st.session_state:
                if st.button(f"Complete: {task['name']}"):
                    st.session_state[f"{task['name']}_end"] = time.time()
                    st.info("Task completed. Please fill out the evaluation form below.")

                if f"{task['name']}_end" in st.session_state:
                    with st.form(f"task_form_{task['name']}"):
                        duration = round(
                            st.session_state[f"{task['name']}_end"] - st.session_state[f"{task['name']}_start"], 2)
                        st.write(f"Task Duration: {duration} seconds")

                        success = st.radio(f"Did you complete the task: {task['name']}?", ["Yes", "No"],
                                           key=f"{task['name']}_success")
                        difficulty = st.slider(f"How difficult was this task (1=Easy, 5=Hard)?", 1, 5,
                                               key=f"{task['name']}_difficulty")
                        comments = st.text_area("Comments or issues?", key=f"{task['name']}_comments")

                        submitted = st.form_submit_button("Save Task Data")
                        if submitted:
                            save_to_csv({
                                "task_name": task["name"],
                                "start_time": datetime.fromtimestamp(st.session_state[f"{task['name']}_start"]),
                                "end_time": datetime.fromtimestamp(st.session_state[f"{task['name']}_end"]),
                                "duration_sec": duration,
                                "task_success": success,
                                "difficulty_rating": difficulty,
                                "user_comments": comments,
                                "timestamp": datetime.now()
                            }, TASK_CSV)
                            st.success(f"Task '{task['name']}' data saved successfully!")
                            # Clear the session state for this task
                            del st.session_state[f"{task['name']}_start"]
                            del st.session_state[f"{task['name']}_end"]

    with exit:
        st.header("Exit Questionnaire")
        satisfaction = st.slider("Overall, how satisfied were you with the app?", 1, 5)
        confusing = st.text_area("What was most confusing or frustrating?")
        suggestions = st.text_area("Any suggestions for improvement?")
        if st.button("Submit Exit Questionnaire"):
            save_to_csv({
                "satisfaction_rating": satisfaction,
                "confusing_part": confusing,
                "suggestions": suggestions,
                "timestamp": datetime.now()
            }, EXIT_CSV)
            st.success("Exit feedback recorded.")

    with collected_data:
        st.header("Collected Data Overview")
        if st.button("Show Data"):
            st.subheader("Consent Data")
            st.dataframe(load_from_csv(CONSENT_CSV))

            st.subheader("Demographic Data")
            st.dataframe(load_from_csv(DEMOGRAPHIC_CSV))

            st.subheader("Task Data")
            st.dataframe(load_from_csv(TASK_CSV))

            st.subheader("Exit Questionnaire Data")
            st.dataframe(load_from_csv(EXIT_CSV))

    with report:
        st.header("Usability Testing Report")

        if st.button("Generate Report"):
            # Load all data
            task_data = load_from_csv(TASK_CSV)
            demographic_data = load_from_csv(DEMOGRAPHIC_CSV)
            exit_data = load_from_csv(EXIT_CSV)

            if not task_data.empty:
                st.subheader("Task Success Rates")

                # Calculate success rates by task
                success_rates = task_data.groupby('task_name')['task_success'].apply(
                    lambda x: (x == 'Yes').sum() / len(x) * 100
                ).round(1)

                # Create columns for success rate cards
                cols = st.columns(len(success_rates))
                for i, (task_name, success_rate) in enumerate(success_rates.items()):
                    with cols[i]:
                        st.container()
                        st.metric(
                            label=task_name,
                            value=f"{success_rate}%"
                        )

                st.subheader("Average Time Taken for Each Task")

                # Calculate average times
                avg_times = task_data.groupby('task_name')['duration_sec'].mean().round(1)

                # Create columns for time cards
                cols = st.columns(len(avg_times))
                for i, (task_name, avg_time) in enumerate(avg_times.items()):
                    minutes = int(avg_time // 60)
                    seconds = int(avg_time % 60)
                    time_display = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                    with cols[i]:
                        st.container()
                        st.metric(
                            label=task_name,
                            value=time_display
                        )

                st.subheader("Key Insights from Participant Feedback")

                # Task difficulty analysis
                st.write("**Task Difficulty Ratings (1=Easy, 5=Hard):**")
                difficulty_avg = task_data.groupby('task_name')['difficulty_rating'].mean().round(1)

                for task_name, avg_difficulty in difficulty_avg.items():
                    difficulty_level = "Easy" if avg_difficulty <= 2 else "Moderate" if avg_difficulty <= 3.5 else "Hard"
                    st.write(f"- {task_name}: {avg_difficulty}/5 ({difficulty_level})")

                # Comments analysis
                st.write("\n**User Comments and Issues:**")
                comments = task_data[task_data['user_comments'].notna() & (task_data['user_comments'] != '')]

                if not comments.empty:
                    for _, row in comments.iterrows():
                        if row['user_comments'].strip():
                            st.write(f"- **{row['task_name']}**: {row['user_comments']}")
                else:
                    st.write("- No specific comments provided by participants")

                st.subheader("Observational Data")

                # Points of confusion or inefficiencies
                st.write("**Identified Issues:**")

                # Tasks with low success rates
                low_success_tasks = success_rates[success_rates < 80]
                if not low_success_tasks.empty:
                    st.write("- **Tasks with Low Success Rates (<80%):**")
                    for task_name, rate in low_success_tasks.items():
                        st.write(f"  - {task_name}: {rate}% success rate")

                # Tasks taking longer than average
                overall_avg_time = task_data['duration_sec'].mean()
                long_tasks = avg_times[avg_times > overall_avg_time * 1.5]
                if not long_tasks.empty:
                    st.write("- **Tasks Taking Significantly Longer:**")
                    for task_name, time_taken in long_tasks.items():
                        minutes = int(time_taken // 60)
                        seconds = int(time_taken % 60)
                        time_display = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                        st.write(f"  - {task_name}: {time_display} average")

                # Tasks with high difficulty ratings
                hard_tasks = difficulty_avg[difficulty_avg >= 4]
                if not hard_tasks.empty:
                    st.write("- **Tasks Rated as Difficult (4+/5):**")
                    for task_name, difficulty in hard_tasks.items():
                        st.write(f"  - {task_name}: {difficulty}/5 difficulty rating")

                # Exit questionnaire insights
                if not exit_data.empty:
                    st.subheader("Overall Satisfaction")

                    # Check if satisfaction_rating column exists
                    if 'satisfaction' in exit_data.columns:
                        avg_satisfaction = exit_data['satisfaction'].mean()

                        st.container()
                        st.metric(
                            label="Average Satisfaction",
                            value=f"{avg_satisfaction:.1f}/5"
                        )
                    else:
                        st.write("No satisfaction rating data available yet.")

                    # Most common issues
                    confusing_col = 'confusing' if 'confusing' in exit_data.columns else None
                    if confusing_col and not exit_data[confusing_col].isna().all():
                        confusing_parts = exit_data[exit_data[confusing_col].notna() & (exit_data[confusing_col] != '')]
                        if not confusing_parts.empty:
                            st.write("**Most Confusing/Frustrating Aspects:**")
                            for _, row in confusing_parts.iterrows():
                                if str(row[confusing_col]).strip():
                                    st.write(f"- {row[confusing_col]}")

                    # Suggestions
                    suggestions_col = 'suggestion' if 'suggestion' in exit_data.columns else None
                    if suggestions_col and not exit_data[suggestions_col].isna().all():
                        suggestions = exit_data[exit_data[suggestions_col].notna() & (exit_data[suggestions_col] != '')]
                        if not suggestions.empty:
                            st.write("**Suggestions for Improvement:**")
                            for _, row in suggestions.iterrows():
                                if str(row[suggestions_col]).strip():
                                    st.write(f"- {row[suggestions_col]}")

                else:
                    st.write("No exit questionnaire data available yet.")

            else:
                st.warning("No task data available. Please complete some tasks first to generate a report.")


if __name__ == "__main__":
    main()