import os

print("\n RUN DAX DATA")
os.system("python financial_data.py")

print("\n=== Running download_gdelt.py ===")
os.system("python download_gdelt_gkg.py")

print("\n=== Running download_gdelt_gkg.py ===")
os.system("python process_gdelt_leads.py")

print("\n=== Running prepare_headlines_for_sentiments.py ===")
os.system("python prepare_headlines_for_sentiment.py")

print("\n=== Running final_sentiment_merge.py ===")
os.system("python preprocess.py")

print("\n=== Pipeline completed. Check your data folder for output parquet files ===")
