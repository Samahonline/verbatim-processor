# run_processing.py
from main_script import VerbatimProcessor
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_processing.py <input_file_path> [output_file_path]")
        print("Example: python run_processing.py data/raw_survey.xlsx")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    processor = VerbatimProcessor()
    result = processor.process_excel_file(input_file, output_file)
    
    if result['status'] == 'success':
        print(f"\n✅ Successfully processed {input_file}")
        print(f"📊 Output: {result['output_file']}")
        print(f"📋 Sheets created: {result['sheets_created']}")
        print(f"🔍 Found {len(result['verbatim_columns'])} verbatim columns")
    else:
        print(f"\n❌ Error: {result['error']}")

if __name__ == "__main__":
    main()