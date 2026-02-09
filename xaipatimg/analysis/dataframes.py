import pandas as pd

def convert_participants_df_to_within_measures_df(df):
    id_participant = []
    score = []
    reliance = []
    overreliance = []
    underreliance = []
    trust = []
    cogload = []
    xai_condition = []
    time_pressure = []
    difficulty = []
    within_condition = []
    tasks_order = []
    task_index_order = []

    for index, row in df.iterrows():
        for diff in ["easy", "hard"]:
            for pressure in ["mild", "strong"]:

                within_cond = f"{diff}_{pressure}"

                id_participant.append(index)
                xai_condition.append(row["xai_condition"])
                time_pressure.append(pressure)
                difficulty.append(diff)
                within_condition.append(within_cond)

                score.append(row[f"score_{diff}_{pressure}"])
                reliance.append(row[f"reliance_{diff}_{pressure}"])
                overreliance.append(row[f"overreliance_{diff}_{pressure}"])
                underreliance.append(row[f"underreliance_{diff}_{pressure}"])
                trust.append(row[f"trust_{diff}_{pressure}"])
                cogload.append(row[f"cogload_{diff}_{pressure}"])
                tasks_order.append(row[f"tasks_order"])
                task_index_order.append(row["tasks_order"].index(within_cond))

    return pd.DataFrame({
        "participant_id": id_participant,
        "difficulty": difficulty,
        "pressure": time_pressure,
        "Difficulty/Time pressure": within_condition,
        "reliance": reliance,
        "overreliance": overreliance,
        "underreliance": underreliance,
        "score": score,
        "trust": trust,
        "cogload": cogload,
        "XAI condition": xai_condition,
        "tasks_order": tasks_order,
        "task_index_order": task_index_order
    })


def convert_participants_df_to_answers_times_df(df):

    id_participant = []
    answer_time = []

    xai_condition = []
    time_pressure = []
    difficulty = []
    within_condition = []
    tasks_order = []
    task_index_order = []

    for index, row in df.iterrows():
        for diff in ["easy", "hard"]:
            for pressure in ["mild", "strong"]:
                for time_value in row[f"answer_times_{diff}_{pressure}"]:

                    within_cond = f"{diff}_{pressure}"

                    id_participant.append(index)
                    xai_condition.append(row["xai_condition"])
                    time_pressure.append(pressure)
                    difficulty.append(diff)
                    within_condition.append(within_cond)
                    answer_time.append(time_value)
                    tasks_order.append(row[f"tasks_order"])
                    task_index_order.append(row["tasks_order"].index(within_cond))

    return pd.DataFrame({
        "participant_id": id_participant,
        "difficulty": difficulty,
        "pressure": time_pressure,
        "Difficulty/Time pressure": within_condition,
        "Answer time": answer_time,
        "XAI condition": xai_condition,
        "tasks_order": tasks_order,
        "task_index_order": task_index_order
    })


