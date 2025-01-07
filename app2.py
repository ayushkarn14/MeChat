from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import re
import pandas as pd
import emoji
import re
from datetime import timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
from collections import Counter
import datetime
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Serve the HTML file
@app.route("/")
def index():
    with open("index.html", "r") as file:
        content = file.read()
    return render_template_string(content)


def generate_output_summary(chat_data):
    # with open(file, "r", encoding="utf-8") as file:
    #     chat_data = file.readlines()
    message_pattern = r"(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2} [APap][Mm]) - (.*?): (.*)"

    dates, times, senders, messages = [], [], [], []

    for line in chat_data:
        match = re.match(message_pattern, line)
        if match:
            dates.append(match.group(1))
            times.append(match.group(2))
            senders.append(match.group(3))
            messages.append(match.group(4))
        else:
            if messages:
                messages[-1] += f" {line.strip()}"

    df = pd.DataFrame(
        {"Date": dates, "Time": times, "Sender": senders, "Message": messages}
    )

    # from here
    message_pattern = (
        r"(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm])\s-\s(.*?):\s(.*)"
    )

    dates, times, senders, messages = [], [], [], []

    for line in chat_data:
        match = re.match(message_pattern, line)
        if match:
            if "Messages and calls are end-to-end encrypted" in line:
                continue
            dates.append(match.group(1))
            times.append(match.group(2))
            senders.append(match.group(3))
            messages.append(match.group(4))
        elif "<Media omitted>" not in line:  # Skip media lines
            if messages:  # Handle continuation of a message
                messages[-1] += f" {line.strip()}"

    df = pd.DataFrame(
        {"Date": dates, "Time": times, "Sender": senders, "Message": messages}
    )
    print(df.head())
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%y %I:%M %p", errors="coerce"
    )

    df.drop(columns=["Date", "Time"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    print(df.head())
    df = df[df["Message"] != "<Media omitted>"]
    df.head()
    # Step 1: Convert the 'Timestamp' column to datetime if it is not already
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Step 2: Filter rows where the year is 2024
    df_2024 = df[df["Timestamp"].dt.year == 2024]
    df = df_2024
    # Step 3: Display the filtered dataframe
    print(df.head())
    df["Date"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%y, %I:%M %p")
    participant_stats = df["Sender"].value_counts()
    df["Hour"] = df["Date"].dt.hour
    hourly_activity = df["Hour"].value_counts()

    def extract_emojis(text):
        # Unicode range for emojis
        emoji_pattern = r"[\U00010000-\U0010ffff]"
        emojis = re.findall(emoji_pattern, text)
        return emojis

    df["Emojis"] = df["Message"].apply(extract_emojis)

    print(df[["Sender", "Message", "Emojis"]].head())
    df["YearMonth"] = df["Timestamp"].dt.to_period("M")

    emoji_counts_by_month = (
        df.explode("Emojis")
        .groupby(["YearMonth", "Emojis"])
        .size()
        .reset_index(name="Count")
    )

    print(emoji_counts_by_month.head())
    # Find the most used emoji for each month
    most_used_emoji_by_month = emoji_counts_by_month.loc[
        emoji_counts_by_month.groupby("YearMonth")["Count"].idxmax()
    ]

    # Display the result
    print(most_used_emoji_by_month[["YearMonth", "Emojis", "Count"]])

    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to the 'Message' column
    df["Sentiment"] = df["Message"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # Check the first few rows
    print(df[["Sender", "Message", "Sentiment"]].head())

    # Sample emoji-to-emotion mapping (expand as needed)
    emoji_to_emotion = {
        "😊": "Happiness",
        "😢": "Sadness",
        "😠": "Anger",
        "😍": "Love",
        "❤️": "Love",
        "😎": "Cool",
        "🎉": "Excitement",
        "😂": "Laughter",
        "😜": "Playfulness",
        "😭": "Sadness",
        "😡": "Anger",
        # Add more emojis and emotions as needed
    }

    # Function to extract emojis using regex
    def extract_emojis(text):
        # Using a regex pattern to find emojis
        return re.findall(r"[^\w\s,]", text)  # Matches all non-word characters (emojis)

    # Function to predict emotion based on emojis in the message
    def predict_emotion_from_emoji(text):
        emojis = extract_emojis(text)
        emotions = [emoji_to_emotion.get(e, "Neutral") for e in emojis]
        return emotions

    # Sample DataFrame (replace with actual data)
    # data = {
    #     'Sender': ['Psyduck', 'Ayush Karn', 'Psyduck', 'Ayush Karn', 'Psyduck', 'Ayush Karn'],
    #     'Message': [
    #         'Kiska kasam khate h Ayush Karn',
    #         'Yash ka? Apna? 😢',
    #         '😊 Mummy ka',
    #         '🎉 So happy today!',
    #         '😭 Feeling sad',
    #         '😢 It\'s over 😭'
    #     ],
    #     'Timestamp': ['2024-03-07 17:02:00', '2024-03-07 17:05:00', '2024-03-07 17:10:00', '2024-03-07 17:15:00', '2024-03-07 17:20:00', '2024-03-07 17:25:00']
    # }

    # Create a DataFrame
    # df = pd.DataFrame(data)

    # Convert Timestamp to datetime and extract month-year
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Month_Year"] = df["Timestamp"].dt.to_period("M")  # Month-Year

    # Apply the function to predict emotion from emojis
    df["Emotions"] = df["Message"].apply(predict_emotion_from_emoji)

    # Flatten the list of emotions and assign them to the DataFrame
    df_exploded = df.explode("Emotions")

    # Remove 'Neutral' emotions
    df_exploded = df_exploded[df_exploded["Emotions"] != "Neutral"]

    # Group by Month-Year and Emotions to get the count of each emotion
    emotion_counts = (
        df_exploded.groupby(["Month_Year", "Emotions"]).size().reset_index(name="Count")
    )

    # Sort emotions by count within each month and get the top 3 emotions
    top_3_emotions_per_month = (
        emotion_counts.groupby("Month_Year")
        .apply(lambda x: x.nlargest(3, "Count"))
        .reset_index(drop=True)
    )

    # Display the result
    print(top_3_emotions_per_month[["Month_Year", "Emotions", "Count"]])

    laughter_emotions = top_3_emotions_per_month[
        top_3_emotions_per_month["Emotions"] == "Laughter"
    ]

    # Scale down the happiness count for the laughing emoji (assuming "laughing" emoji corresponds to the 'Happiness' emotion)
    top_3_emotions_per_month.loc[
        top_3_emotions_per_month["Emotions"] == "Laughter", "Count"
    ] /= 2

    # Step 2: Verify the updated counts
    print(top_3_emotions_per_month.head())

    # Filter rows where Emojis list is not empty
    df_with_emojis = df[df["Emojis"].apply(lambda x: len(x) > 0)]
    df_with_emojis
    # Join all emojis into a single string
    all_emojis = "".join(
        [emoji for sublist in df_with_emojis["Emojis"] for emoji in sublist]
    )
    all_emojis

    # Count the frequency of each emoji
    emoji_counts = Counter(all_emojis)

    # Get the top 3 most common emojis
    top_3_emojis = emoji_counts.most_common(3)

    top_3_emojis

    # Assuming you have the DataFrame 'df' with columns 'YearMonth' and 'Emojis'
    # You can add sentiment scores for each emoji in a dictionary

    # Example of emoji sentiment scores (positive for happy, negative for sad)
    emoji_sentiments = {
        "😂": 1,  # Happy
        "🤣": 1,  # Happy
        "🥹": 0,  # Happy
        "🥰": 0.8,  # Happy
        "😭": -2,  # Sad
        "😔": -1,  # Sad
        "😢": -2,  # Sad
        "😞": -0.9,  # Sad
        "😩": -0.9,  # Sad
    }

    # Function to get sentiment for emojis in a row
    def get_sentiment(emojis):
        sentiment = 0
        for emoji in emojis:
            sentiment += emoji_sentiments.get(
                emoji, 0
            )  # Default to 0 if emoji is not in the dictionary
        return sentiment

    # Apply the function to calculate sentiment for each row
    df["Sentiment"] = df["Emojis"].apply(get_sentiment)

    # Group by 'YearMonth' and calculate the total sentiment for each month
    monthly_sentiment = df.groupby("YearMonth")["Sentiment"].sum().reset_index()

    # Find the happiest and saddest month
    happiest_month = monthly_sentiment.loc[monthly_sentiment["Sentiment"].idxmax()]
    saddest_month = monthly_sentiment.loc[monthly_sentiment["Sentiment"].idxmin()]

    # print("Happiest Month:", happiest_month)
    # print("Saddest Month:", saddest_month)
    # Get the happiest and saddest month names
    happiest_month_name = happiest_month["YearMonth"]
    saddest_month_name = saddest_month["YearMonth"]

    print("Happiest Month:", happiest_month_name)
    print("Saddest Month:", saddest_month_name)
    m_dict = {
        "2024-01": "January",
        "2024-02": "February",
        "2024-03": "March",
        "2024-04": "April",
        "2024-05": "May",
        "2024-06": "June",
        "2024-07": "July",
        "2024-08": "August",
        "2024-09": "September",
        "2024-10": "October",
        "2024-11": "November",
        "2024-12": "December",
    }
    print("Happiest month : ", m_dict[str(happiest_month_name)])
    print("Saddest month : ", m_dict[str(saddest_month_name)])
    # Summary
    e = ""
    for i in top_3_emojis:
        e += i[0]
    print("Top emojis of the conversation : ", e)

    # Assuming you have a DataFrame 'df' with columns 'Sender' and 'Timestamp'

    # Ensure 'Timestamp' is a datetime object
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by="Timestamp")

    # Calculate the time difference between consecutive messages
    df["Time_Diff"] = (
        df["Timestamp"].diff().dt.total_seconds() / 3600
    )  # Convert seconds to hours

    # Identify rows where the time difference is >= 4 hours
    df["Gap_Over_4hrs"] = df["Time_Diff"] >= 4

    # Shift the 'Gap_Over_4hrs' column to align with the sender of the message after the gap
    df["First_After_Gap"] = df["Gap_Over_4hrs"].shift(fill_value=False)

    # Filter for messages that are the first after a gap
    first_after_gap = df[df["First_After_Gap"]]

    # Count the frequency of who sends the first message after a gap
    frequency = first_after_gap["Sender"].value_counts()

    # Assuming 'df' has a 'Timestamp' column
    df["Date"] = df["Timestamp"].dt.date  # Extract the date from the Timestamp

    # Count the number of messages for each date
    daily_message_count = df.groupby("Date").size().reset_index(name="Message_Count")

    # Find the most active day
    most_active_day = daily_message_count.loc[
        daily_message_count["Message_Count"].idxmax()
    ]

    # Get the date and the total number of messages
    most_active_date = most_active_day["Date"]
    total_messages = most_active_day["Message_Count"]

    print(f"Most Active Day: {most_active_date}")
    print(f"Total Messages on that Day: {total_messages}")

    def format_date(date_str):
        # Parse the date string
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

        # Get the day, month, and year
        day = date_obj.day
        month = date_obj.strftime("%B")  # Full month name
        year = date_obj.year

        # Determine the ordinal suffix for the day
        if 10 <= day % 100 <= 20:  # Handles 11th, 12th, 13th, etc.
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        # Format the date
        formatted_date = f"{day}{suffix} {month}, {year}"
        return formatted_date

    def peak_chatting_hour(timestamps):
        """
        Calculate the peak chatting hour from a list of timestamps.

        Args:
            timestamps (list): List of timestamps in 'HH:MM:SS' format.

        Returns:
            int: The hour (0-23) with the highest frequency.
        """
        from collections import Counter

        # Extract hours from timestamps
        hours = [int(timestamp.split(":")[0]) for timestamp in timestamps]

        # Count occurrences of each hour
        hour_counts = Counter(hours)

        # Find the hour with the maximum count
        peak_hour = hour_counts.most_common(1)[0][0]

        return peak_hour

    # Example usage
    test_timestamps = ["14:23:45", "15:30:21", "14:15:05", "14:59:59", "15:01:01"]
    peak_hour = peak_chatting_hour(test_timestamps)
    print(f"Peak Chatting Hour: {peak_hour}:00")

    # Example usage
    date = "2024-01-12"
    print(format_date(date))  # Output: 12th January, 2024
    # print("Number of times ",frequency.index[0]," texted first",frequency[frequency.index[0]])
    # print("Number of times ",frequency.index[1]," texted first",frequency[frequency.index[1]])
    # # print(frequency)
    # print(f"Most Active Day: {format_date(str(most_active_date))}")
    # print("Happiest Month:", happiest_month_name)
    # print("Saddest Month:", saddest_month_name)
    # print(f"Total Messages on that Day: {total_messages}")
    # print('Top emojis of the conversation : ',e)
    # # output="Number of times ",frequency.index[0]," texted first",frequency[frequency.index[0]],"\n","Number of times ",frequency.index[1]," texted first",frequency[frequency.index[1]],"\n",f"Most Active Day: {format_date(str(most_active_date))}","\n","Happiest Month:", happiest_month_name,"\n","Saddest Month:", saddest_month_name,"\n",f"Total Messages on that Day: {total_messages}","\n",'Top emojis of the conversation : ',e,"\n"
    output_summary = f"""
Number of times {frequency.index[0]} texted first: {frequency[frequency.index[0]]}
Number of times {frequency.index[1]} texted first: {frequency[frequency.index[1]]}
Most Active Day: {format_date(str(most_active_date))}
Total Messages on that Day: {total_messages}
Happiest Month: {happiest_month_name}
Saddest Month: {saddest_month_name}
Top emojis of the conversation: {e}
"""
    print(output_summary)

    def longest_streak(data):
        data = data.sort_values(by="Timestamp").reset_index(drop=True)
        max_streak_duration = timedelta(0)
        current_streak_start = data["Timestamp"].iloc[0]

        for i in range(1, len(data)):
            time_diff = data["Timestamp"].iloc[i] - data["Timestamp"].iloc[i - 1]

            if time_diff < timedelta(minutes=10):
                continue
            else:
                # End the current streak and calculate its duration
                current_streak_duration = (
                    data["Timestamp"].iloc[i - 1] - current_streak_start
                )
                max_streak_duration = max(max_streak_duration, current_streak_duration)
                # Start a new streak
                current_streak_start = data["Timestamp"].iloc[i]

        # Handle the final streak
        final_streak_duration = data["Timestamp"].iloc[-1] - current_streak_start
        max_streak_duration = max(max_streak_duration, final_streak_duration)

        # Convert duration to a readable format
        hours, remainder = divmod(max_streak_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours)} hours, {int(minutes)} minutes"

    print(longest_streak(df))
    output_summary += f"Longest conversation: {longest_streak(df)}\n"

    def longest_day_streak(data):
        """
        Calculate the longest streak of consecutive days with at least one message.

        Args:
        data (pd.DataFrame): DataFrame with a column 'Timestamp' containing datetime objects.

        Returns:
        int: The length of the longest streak in days.
        """
        # Extract the dates from the timestamps
        data["Date"] = data["Timestamp"].dt.date

        # Get the unique days when messages were sent
        unique_days = sorted(data["Date"].unique())

        # Initialize variables
        longest_streak = 0
        current_streak = 1

        for i in range(1, len(unique_days)):
            if unique_days[i] == unique_days[i - 1] + timedelta(days=1):
                # Consecutive day, increase the streak
                current_streak += 1
            else:
                # Break in streak, update the longest streak
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1

        # Handle the final streak
        longest_streak = max(longest_streak, current_streak)

        return longest_streak

    # Example usage:
    # Assuming df is your DataFrame and 'Timestamp' column is of datetime type
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print(f"Longest streak of days talked: {longest_day_streak(df)} days")
    output_summary += f"Longest streak of days talked: {longest_day_streak(df)} days\n"
    # output_summary += f"Peak chatting hour {peak_chatting_hour(df['Time'])}"

    hour_counts = Counter(df["Hour"])
    peak_hour = hour_counts.most_common(1)[0][0]
    output_summary += f"Peak chatting hour: {peak_hour}"
    print(output_summary)
    return output_summary


# Endpoint for file processing
@app.route("/process_file", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    temp_path = os.path.join("/tmp", uploaded_file.filename)
    uploaded_file.save(temp_path)

    try:
        with open(temp_path, "r") as file:
            content = file.readlines()
        output_summary = generate_output_summary(content)
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
    finally:
        os.remove(temp_path)

    return jsonify({"output_summary": output_summary})


if __name__ == "__main__":
    app.run(debug=True)
