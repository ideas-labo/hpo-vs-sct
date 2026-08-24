import pandas as pd
import os

input_file = 'result.csv'

if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found")
else:
    try:
        df = pd.read_csv(input_file)
        
        required_columns = ['optimizer_category']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing required columns → {required_columns}")
        else:
            original_count = len(df)
            
            delete_condition = df['optimizer_category'] == 'general'
            df_filtered = df[~delete_condition]
            
            filtered_count = len(df_filtered)
            removed_count = original_count - filtered_count
            
            output_file = 'result.csv'
            df_filtered.to_csv(output_file, index=False)
            
            print(f"Processing completed!")
            print(f"Original rows: {original_count}")
            print(f"Removed rows: {removed_count} (optimizer_category='general')")
            print(f"Filtered data saved to: {output_file}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")