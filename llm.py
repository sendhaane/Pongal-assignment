from transformers import pipeline

# Load a pre-trained model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example accident report
accident_report = """
On January 15, 2025, a major accident occurred on Highway 17 due to dense fog and overspeeding. 
The accident involved 12 vehicles, leading to severe injuries to 15 people and minor injuries to 8. 
Emergency services were delayed due to poor visibility. The road remained closed for 6 hours.
"""

# Summarize the report
summary = summarizer(accident_report, max_length=50, min_length=20, do_sample=False)
print("Summary:", summary[0]['summary_text'])
