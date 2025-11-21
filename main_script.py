import pandas as pd
import os
import re
from typing import List, Dict, Tuple

class VerbatimProcessor:
    def __init__(self):
        self.verbatim_keywords = [
            'verbatim', 'text', 'comment', 'response', 'answer',
            'open', 'openend', 'qualitative', 'feedback', 'remark',
            'opinion', 'suggestion', 'describe', 'explain'
        ]
    
    def is_verbatim_column(self, column_name: str) -> bool:
        """
        Check if a column name indicates it's a verbatim/text column
        """
        if not isinstance(column_name, str):
            return False
        
        column_lower = column_name.lower()
        
        # Check for verbatim keywords in column name
        for keyword in self.verbatim_keywords:
            if keyword in column_lower:
                return True
        
        # Check for text-like patterns
        text_patterns = [r'.*text.*', r'.*comment.*', r'.*response.*', r'.*answer.*']
        for pattern in text_patterns:
            if re.match(pattern, column_lower):
                return True
        
        return False
    
    def find_intnr_column(self, df: pd.DataFrame) -> str:
        """
        Find the intnr column (case insensitive)
        """
        intnr_patterns = ['intnr', 'interview', 'respondent', 'id', 'caseid']
        
        for col in df.columns:
            if not isinstance(col, str):
                continue
            
            col_lower = col.lower()
            for pattern in intnr_patterns:
                if pattern in col_lower:
                    return col
        
        # If no specific intnr column found, look for the first column that seems like an ID
        for col in df.columns:
            if 'id' in str(col).lower() or 'num' in str(col).lower():
                return col
        
        # If still not found, return the first column
        return df.columns[0]
    
    def detect_verbatim_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that likely contain verbatim text data
        """
        verbatim_columns = []
        
        for column in df.columns:
            if self.is_verbatim_column(str(column)):
                verbatim_columns.append(column)
            else:
                # Additional check: if column contains mostly text data
                if self._is_text_data(df[column]):
                    verbatim_columns.append(column)
        
        return verbatim_columns
    
    def _is_text_data(self, series: pd.Series) -> bool:
        """
        Check if a series contains mostly text data
        """
        # Sample up to 100 non-null values for efficiency
        sample_size = min(100, series.dropna().shape[0])
        if sample_size == 0:
            return False
        
        sample = series.dropna().sample(n=sample_size, random_state=42)
        
        text_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
            
            total_count += 1
            # Check if value is string and has reasonable length for verbatim
            if isinstance(value, str) and len(value.strip()) > 10:
                text_count += 1
            elif isinstance(value, str) and any(char.isalpha() for char in value):
                text_count += 1
        
        # If more than 70% of sampled data appears to be text, consider it verbatim
        return (text_count / total_count) > 0.7 if total_count > 0 else False
    
    def process_excel_file(self, input_file_path: str, output_file_path: str = None) -> Dict:
        """
        Main processing function
        """
        try:
            # Read the Excel file
            print(f"Reading file: {input_file_path}")
            df = pd.read_excel(input_file_path)
            
            # Find intnr column
            intnr_column = self.find_intnr_column(df)
            print(f"Found intnr column: {intnr_column}")
            
            # Detect verbatim columns
            verbatim_columns = self.detect_verbatim_columns(df)
            print(f"Found {len(verbatim_columns)} verbatim columns: {verbatim_columns}")
            
            if not verbatim_columns:
                raise ValueError("No verbatim columns found in the dataset")
            
            # Generate output file path if not provided
            if output_file_path is None:
                base_name = os.path.splitext(input_file_path)[0]
                output_file_path = f"{base_name}_verbatim_analysis.xlsx"
            
            # Create Excel writer with xlsxwriter engine for better performance
            with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create summary sheet
                summary_data = {
                    'Verbatim Column': verbatim_columns,
                    'Number of Responses': [df[col].notna().sum() for col in verbatim_columns],
                    'Column Description': [f"Contains verbatim responses for question {col}" for col in verbatim_columns]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                summary_sheet.set_column('A:A', 30)
                summary_sheet.set_column('B:B', 20)
                summary_sheet.set_column('C:C', 50)
                
                # Create individual sheets for each verbatim column
                for i, verbatim_col in enumerate(verbatim_columns):
                    # Clean sheet name (Excel sheet names have limitations)
                    sheet_name = self._clean_sheet_name(verbatim_col, i)
                    
                    # Create dataframe with intnr and verbatim column
                    result_df = df[[intnr_column, verbatim_col]].copy()
                    result_df = result_df[result_df[verbatim_col].notna()]
                    result_df = result_df[result_df[verbatim_col] != '']
                    
                    # Write to Excel
                    result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:A', 15)  # intnr column
                    worksheet.set_column('B:B', 50)  # verbatim column
                
                # Add a key sheet with original column names
                key_data = {
                    'Sheet Name': [self._clean_sheet_name(col, i) for i, col in enumerate(verbatim_columns)],
                    'Original Column Name': verbatim_columns
                }
                key_df = pd.DataFrame(key_data)
                key_df.to_excel(writer, sheet_name='Key', index=False)
            
            print(f"Processing completed successfully!")
            print(f"Output file: {output_file_path}")
            print(f"Created {len(verbatim_columns)} verbatim sheets + Summary + Key sheets")
            
            return {
                'status': 'success',
                'output_file': output_file_path,
                'intnr_column': intnr_column,
                'verbatim_columns': verbatim_columns,
                'sheets_created': len(verbatim_columns) + 2
            }
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _clean_sheet_name(self, name: str, index: int) -> str:
        """
        Clean sheet name to comply with Excel limitations
        """
        # Excel sheet name limitations: max 31 characters, no special characters
        cleaned = re.sub(r'[\\/*?\[\]:]', '', str(name))
        cleaned = cleaned[:31]  # Truncate to 31 characters
        
        # If empty after cleaning, use generic name
        if not cleaned.strip():
            cleaned = f"Verbatim_{index + 1}"
        
        return cleaned

# Batch Processing Function
def process_multiple_files(input_folder: str, output_folder: str = None):
    """
    Process all Excel files in a folder
    """
    processor = VerbatimProcessor()
    
    if output_folder is None:
        output_folder = input_folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find Excel files
    excel_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.xlsx', '.xls')) and not f.startswith('~')]
    
    results = {}
    
    for file in excel_files:
        print(f"\n{'='*50}")
        print(f"Processing: {file}")
        print(f"{'='*50}")
        
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_verbatim_analysis.xlsx")
        
        result = processor.process_excel_file(input_path, output_path)
        results[file] = result
    
    return results

# Example Usage
if __name__ == "__main__":
    # Initialize processor
    processor = VerbatimProcessor()
    
    # Single file processing
    input_file = "raw_data.xlsx"  # Replace with your file path
    result = processor.process_excel_file(input_file)
    
    # For batch processing (uncomment to use)
    # folder_path = "path/to/your/excel/files"
    # batch_results = process_multiple_files(folder_path)