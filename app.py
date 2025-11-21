import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import List, Dict, Tuple
import tempfile
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Verbatim Data Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VerbatimProcessor:
    def __init__(self):
        self.verbatim_keywords = [
            'verbatim', 'text', 'comment', 'response', 'answer',
            'open', 'openend', 'qualitative', 'feedback', 'remark',
            'opinion', 'suggestion', 'describe', 'explain', 'why',
            'other', 'specify', 'additional', 'thought', 'idea'
        ]
        
        # Keywords that indicate NULL/empty columns (case insensitive)
        self.null_indicators = [
            'null', 'none', 'empty', 'n/a', 'na', 'missing', 
            'no data', 'no response', 'blank'
        ]
        
        # Date patterns to detect date-like columns
        self.date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD-MM-YY, etc.
            r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD, etc.
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # 15 Jan 2023
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # January 15, 2023
            r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)?',  # Time patterns
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Timestamps
        ]
        
        # Date-related column name keywords
        self.date_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'day', 'month', 'year', 'birth', 'dob', 'start', 'end',
            'submitted', 'completed', 'response_date'
        ]
        
        # Address-related column name keywords
        self.address_keywords = [
            'address', 'street', 'city', 'state', 'zip', 'postal', 'postcode',
            'country', 'location', 'county', 'province', 'region',
            'addr', 'st', 'road', 'ave', 'avenue', 'boulevard', 'blvd',
            'lane', 'ln', 'drive', 'dr', 'court', 'ct', 'place', 'pl'
        ]
        
        # Address patterns in content
        self.address_patterns = [
            r'\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln)',
            r'[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(-\d{4})?',  # City, STATE ZIP
            r'P\.?O\.?\s+Box\s+\d+',  # PO Box
            r'\b[A-Z]{2}\s*\d{5}\b',  # STATE ZIP
            r'\b\d{5}(-\d{4})?\b',  # ZIP code
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
    
    def is_address_column(self, series: pd.Series, column_name: str = "") -> bool:
        """
        Check if a column contains address data
        """
        if series.empty:
            return False
        
        # Check column name for address keywords
        if column_name and any(address_keyword in str(column_name).lower() 
                             for address_keyword in self.address_keywords):
            return True
        
        # Sample the data to check for address patterns
        non_null_series = series.dropna()
        if non_null_series.empty:
            return False
        
        sample_size = min(100, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        address_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            value_str = str(value).strip()
            
            # Check for address patterns
            if self._looks_like_address(value_str):
                address_count += 1
        
        # If more than 60% of the data appears to be addresses, exclude the column
        address_ratio = address_count / total_count if total_count > 0 else 0
        return address_ratio > 0.6
    
    def _looks_like_address(self, value: str) -> bool:
        """
        Check if a string looks like an address using regex patterns
        """
        value_clean = ' '.join(value.split())  # Normalize whitespace
        
        for pattern in self.address_patterns:
            if re.search(pattern, value_clean, re.IGNORECASE):
                return True
        
        # Additional checks for address-like content
        address_indicators = [
            # Contains street number and name
            re.search(r'^\d+\s+[A-Za-z]', value_clean),
            # Contains city, state pattern
            re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}', value_clean),
            # Contains ZIP code
            re.search(r'\b\d{5}(-\d{4})?\b', value_clean),
            # Contains common address abbreviations
            any(abbr in value_clean.lower() for abbr in [' st', ' ave', ' rd', ' dr', ' blvd', ' ln', ' ct'])
        ]
        
        return any(address_indicators)
    
    def is_date_column(self, series: pd.Series, column_name: str = "") -> bool:
        """
        Check if a column contains mostly date-like data
        """
        if series.empty:
            return False
        
        # Check column name for date keywords
        if column_name and any(date_keyword in str(column_name).lower() 
                             for date_keyword in self.date_keywords):
            return True
        
        # Check if the column is already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Sample the data to check for date patterns
        non_null_series = series.dropna()
        if non_null_series.empty:
            return False
        
        sample_size = min(100, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        date_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            value_str = str(value).strip()
            
            # Check for date patterns
            if self._looks_like_date(value_str):
                date_count += 1
            # Additional check: try to parse as date
            elif self._can_parse_as_date(value_str):
                date_count += 1
        
        # If more than 70% of the data appears to be dates, exclude the column
        date_ratio = date_count / total_count if total_count > 0 else 0
        return date_ratio > 0.7
    
    def _looks_like_date(self, value: str) -> bool:
        """
        Check if a string looks like a date using regex patterns
        """
        for pattern in self.date_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    def _can_parse_as_date(self, value: str) -> bool:
        """
        Try to parse the value as a date using pandas
        """
        try:
            # Try parsing with pandas
            parsed = pd.to_datetime(value, errors='coerce', infer_datetime_format=True)
            return not pd.isna(parsed)
        except:
            return False
    
    def is_null_column(self, series: pd.Series, column_name: str = "") -> bool:
        """
        Check if a column contains mostly NULL/empty data
        """
        if series.empty:
            return True
        
        # Convert to string and check for null indicators
        non_null_values = series.dropna()
        
        if len(non_null_values) == 0:
            return True
        
        # Check column name for null indicators
        if column_name and any(null_indicator in str(column_name).lower() 
                             for null_indicator in self.null_indicators):
            return True
        
        # Check if most values contain null indicators
        null_count = 0
        sample_size = min(100, len(non_null_values))
        
        if sample_size == 0:
            return True
        
        sample = non_null_values.sample(n=sample_size, random_state=42)
        
        for value in sample:
            if pd.isna(value):
                null_count += 1
                continue
                
            value_str = str(value).strip().lower()
            
            # Check for null indicators in the actual data
            if any(null_indicator in value_str for null_indicator in self.null_indicators):
                null_count += 1
            elif value_str == '':
                null_count += 1
            elif len(value_str) < 3:  # Very short responses might be placeholders
                null_count += 1
        
        # If more than 80% of the data appears to be NULL/empty, exclude the column
        null_ratio = null_count / sample_size
        return null_ratio > 0.8
    
    def find_intnr_column(self, df: pd.DataFrame) -> str:
        """
        Find the intnr column (case insensitive)
        """
        intnr_patterns = ['intnr', 'interview', 'respondent', 'id', 'caseid', 'resp_id']
        
        for col in df.columns:
            if not isinstance(col, str):
                continue
            
            col_lower = col.lower()
            for pattern in intnr_patterns:
                if pattern in col_lower:
                    return col
        
        # If no specific intnr column found, look for the first column that seems like an ID
        for col in df.columns:
            col_str = str(col).lower()
            if any(pattern in col_str for pattern in ['id', 'num', 'code', 'case']):
                return col
        
        # If still not found, return the first column
        return df.columns[0]
    
    def detect_verbatim_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that likely contain verbatim text data, excluding NULL, date, and address columns
        """
        verbatim_columns = []
        excluded_columns = []
        
        for column in df.columns:
            column_name = str(column)
            
            # First check if this is a NULL column
            if self.is_null_column(df[column], column_name):
                excluded_columns.append((column_name, "NULL column"))
                continue
            
            # Then check if it's a date column
            if self.is_date_column(df[column], column_name):
                excluded_columns.append((column_name, "Date column"))
                continue
            
            # Then check if it's an address column
            if self.is_address_column(df[column], column_name):
                excluded_columns.append((column_name, "Address column"))
                continue
            
            # Then check if it's a verbatim column
            if self.is_verbatim_column(column_name):
                verbatim_columns.append(column)
            else:
                # Additional check: if column contains mostly text data
                if self._is_text_data(df[column]):
                    verbatim_columns.append(column)
        
        # Log excluded columns for debugging
        if excluded_columns:
            st.sidebar.info(f"Excluded {len(excluded_columns)} columns")
            for col, reason in excluded_columns:
                st.sidebar.write(f"❌ {col} ({reason})")
        
        return verbatim_columns
    
    def _is_text_data(self, series: pd.Series) -> bool:
        """
        Check if a series contains mostly text data
        """
        # Sample up to 100 non-null values for efficiency
        non_null_series = series.dropna()
        if non_null_series.empty:
            return False
        
        sample_size = min(100, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        text_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
            
            total_count += 1
            value_str = str(value).strip()
            
            # Skip NULL indicator values
            if any(null_indicator in value_str.lower() for null_indicator in self.null_indicators):
                continue
            
            # Skip date-like values
            if self._looks_like_date(value_str) or self._can_parse_as_date(value_str):
                continue
            
            # Skip address-like values
            if self._looks_like_address(value_str):
                continue
            
            # Check if value is string and has reasonable length for verbatim
            if len(value_str) > 10:
                text_count += 1
            elif any(char.isalpha() for char in value_str):
                text_count += 1
        
        # If more than 70% of sampled data appears to be text, consider it verbatim
        return (text_count / total_count) > 0.7 if total_count > 0 else False
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[io.BytesIO, Dict]:
        """
        Process dataframe and return Excel file as BytesIO
        """
        try:
            # Find intnr column
            intnr_column = self.find_intnr_column(df)
            
            # Detect verbatim columns (automatically excludes NULL, date, and address columns)
            verbatim_columns = self.detect_verbatim_columns(df)
            
            if not verbatim_columns:
                raise ValueError("No valid verbatim columns found in the dataset")
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create summary sheet
                summary_data = {
                    'Question Column': verbatim_columns,
                    'Sheet Name': [self._clean_sheet_name(col, i) for i, col in enumerate(verbatim_columns)],
                    'Number of Responses': [df[col].notna().sum() for col in verbatim_columns],
                    'Description': [f"Verbatim responses for: {col}" for col in verbatim_columns]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                summary_sheet.set_column('A:A', 30)
                summary_sheet.set_column('B:B', 20)
                summary_sheet.set_column('C:C', 15)
                summary_sheet.set_column('D:D', 50)
                
                # Add header formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                
                # Create individual sheets for each verbatim column
                for i, verbatim_col in enumerate(verbatim_columns):
                    # Clean sheet name
                    sheet_name = self._clean_sheet_name(verbatim_col, i)
                    
                    # Create dataframe with intnr and verbatim column
                    result_df = df[[intnr_column, verbatim_col]].copy()
                    
                    # Remove empty responses, NULL indicator responses, date-like, and address responses
                    result_df = result_df.dropna(subset=[verbatim_col])
                    
                    # Create filter mask
                    mask = result_df[verbatim_col].astype(str).str.strip() != ''
                    
                    # Filter out NULL indicator values
                    mask &= ~result_df[verbatim_col].astype(str).str.lower().isin(
                        [null.lower() for null in self.null_indicators]
                    )
                    
                    # Additional check for values containing null indicators
                    for null_indicator in self.null_indicators:
                        mask &= ~result_df[verbatim_col].astype(str).str.lower().str.contains(
                            null_indicator, na=False
                        )
                    
                    # Filter out date-like values
                    mask &= ~result_df[verbatim_col].apply(
                        lambda x: self._looks_like_date(str(x)) or self._can_parse_as_date(str(x))
                    )
                    
                    # Filter out address-like values
                    mask &= ~result_df[verbatim_col].apply(
                        lambda x: self._looks_like_address(str(x))
                    )
                    
                    result_df = result_df[mask]
                    
                    # Write to Excel
                    result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Auto-adjust column widths and format
                    worksheet = writer.sheets[sheet_name]
                    worksheet.set_column('A:A', 15)  # intnr column
                    worksheet.set_column('B:B', 50)  # verbatim column
                    
                    # Add header formatting to question sheets
                    for col_num, value in enumerate(result_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                
                # Add processing information sheet
                info_data = {
                    'Processing Information': [
                        'Original file columns',
                        f'Intnr/ID column: {intnr_column}',
                        f'Verbatim columns found: {len(verbatim_columns)}',
                        f'Total rows processed: {len(df)}',
                        'NULL columns were automatically excluded',
                        'Date columns were automatically excluded',
                        'Address columns were automatically excluded',
                        'Generated by Verbatim Processor'
                    ]
                }
                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name='Info', index=False)
                info_sheet = writer.sheets['Info']
                info_sheet.set_column('A:A', 40)
            
            # Prepare output
            output.seek(0)
            
            return output, {
                'status': 'success',
                'intnr_column': intnr_column,
                'verbatim_columns': verbatim_columns,
                'sheets_created': len(verbatim_columns) + 2,
                'total_rows': len(df)
            }
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")
    
    def _clean_sheet_name(self, name: str, index: int) -> str:
        """
        Clean sheet name to comply with Excel limitations
        """
        # Excel sheet name limitations: max 31 characters, no special characters
        cleaned = re.sub(r'[\\/*?\[\]:]', '', str(name))
        cleaned = cleaned[:31]  # Truncate to 31 characters
        
        # If empty after cleaning, use generic name
        if not cleaned.strip():
            cleaned = f"Question_{index + 1}"
        
        return cleaned

def main():
    # Header
    st.title("📊 Verbatim Data Processor")
    st.markdown("""
    **Automatically extract and organize verbatim text responses from your survey data**
    
    Upload your Excel file containing survey data, and this tool will:
    - 🔍 Automatically detect verbatim/text columns
    - 🚫 **Exclude NULL/empty columns automatically**
    - 📅 **Exclude date/time columns automatically**
    - 🏠 **Exclude address columns automatically**
    - 📋 Identify respondent ID columns
    - 📄 Create separate sheets for each question
    - 💾 Generate a formatted Excel workbook for analysis
    """)
    
    # Initialize processor
    processor = VerbatimProcessor()
    
    # File upload section
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose Excel File",
        type=['xlsx', 'xls'],
        help="Upload your raw data Excel file"
    )
    
    # Processing options
    st.sidebar.header("Processing Options")
    sample_mode = st.sidebar.checkbox(
        "Process sample data (first 1000 rows)",
        value=False,
        help="Useful for testing with large files"
    )
    
    # Advanced options
    st.sidebar.header("Advanced Options")
    show_excluded = st.sidebar.checkbox(
        "Show excluded columns",
        value=True,
        help="Display columns that were excluded due to NULL, date, or address content"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                with st.spinner("Reading Excel file..."):
                    if sample_mode:
                        df = pd.read_excel(uploaded_file, nrows=1000)
                        st.info("📝 Sample mode: Processing first 1000 rows only")
                    else:
                        df = pd.read_excel(uploaded_file)
                
                # Display file info
                st.success(f"✅ File loaded successfully: {uploaded_file.name}")
                
                # Show data preview
                st.subheader("Data Preview")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show first few rows
                preview_expander = st.expander("View First 5 Rows")
                with preview_expander:
                    st.dataframe(df.head())
                
                # Show column information
                st.subheader("Column Analysis")
                intnr_column = processor.find_intnr_column(df)
                verbatim_columns = processor.detect_verbatim_columns(df)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("ID Column Found", intnr_column)
                    st.metric("Valid Verbatim Columns", len(verbatim_columns))
                
                with col_b:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                
                # Show detected columns
                if verbatim_columns:
                    st.write("**✅ Valid Verbatim Columns Found:**")
                    for i, col in enumerate(verbatim_columns, 1):
                        non_null_count = df[col].notna().sum()
                        st.write(f"{i}. **{col}** ({non_null_count} responses)")
                else:
                    st.warning("No valid verbatim columns detected after filtering.")
                    st.info("""
                    **Tips for better detection:**
                    - Ensure column names contain words like: verbatim, text, comment, response, answer
                    - Columns should contain actual text responses (not NULL/empty values, dates, or addresses)
                    - Avoid columns that are mostly placeholder data
                    - Columns should have meaningful text content
                    """)
                
                # Process data when button is clicked
                if st.button("🚀 Process Data", type="primary"):
                    with st.spinner("Processing data and generating Excel workbook..."):
                        try:
                            output_bytes, result_info = processor.process_dataframe(df)
                            
                            # Success message
                            st.success("✅ Processing completed successfully!")
                            
                            # Results summary
                            st.subheader("Processing Results")
                            results_col1, results_col2, results_col3 = st.columns(3)
                            
                            with results_col1:
                                st.metric("Sheets Created", result_info['sheets_created'])
                            with results_col2:
                                st.metric("Questions Processed", len(result_info['verbatim_columns']))
                            with results_col3:
                                st.metric("Total Rows", result_info['total_rows'])
                            
                            # Download section
                            st.subheader("Download Results")
                            
                            # Generate download filename
                            original_name = uploaded_file.name
                            base_name = original_name.replace('.xlsx', '').replace('.xls', '')
                            download_name = f"{base_name}_verbatim_analysis.xlsx"
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Processed Excel File",
                                data=output_bytes,
                                file_name=download_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Click to download the processed Excel workbook"
                            )
                            
                            # Preview of what's included
                            st.info("""
                            **Output includes:**
                            - 📋 **Summary Sheet**: Overview of all questions and response counts
                            - ❓ **Individual Question Sheets**: One sheet per verbatim question with ID column
                            - ℹ️ **Info Sheet**: Processing information and metadata
                            - 🚫 **NULL columns automatically excluded**
                            - 📅 **Date columns automatically excluded**
                            - 🏠 **Address columns automatically excluded**
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please make sure you've uploaded a valid Excel file.")
        
        else:
            # Show instructions when no file is uploaded
            st.info("👆 Please upload an Excel file to get started")
            
            # Example of expected format
            st.subheader("Expected Input Format")
            st.markdown("""
            Your Excel file should contain:
            - One row per respondent
            - One column for respondent ID (automatically detected)
            - Multiple columns for verbatim/text responses (automatically detected)
            
            **The following columns will be automatically excluded:**
            - **NULL columns**: null, none, empty, n/a, na, missing, no data
            - **Date columns**: date, time, timestamp, created, updated, etc.
            - **Address columns**: address, street, city, state, zip, country, location
            - **Columns with >70% non-verbatim content**
            """)
    
    with col2:
        # Sidebar content
        st.sidebar.header("How It Works")
        st.sidebar.markdown("""
        1. **Upload** your Excel file with survey data
        2. **Automatic filtering** of NULL, date, and address columns
        3. **Detection** of valid verbatim columns
        4. **Review** the detected columns
        5. **Process** and download results
        
        **Exclusion Criteria:**
        - NULL/empty columns (>80% NULL content)
        - Date columns (>70% date-like content)
        - Address columns (>60% address-like content)
        - Columns with non-verbatim names
        """)
        
        # Exclusion indicators info
        st.sidebar.header("🚫 Excluded Columns")
        st.sidebar.markdown("""
        **NULL Indicators:**
        - null, none, empty
        - n/a, na, missing
        - no data, no response
        
        **Date Indicators:**
        - date, time, timestamp
        - created, updated, modified
        - MM/DD/YYYY formats
        
        **Address Indicators:**
        - address, street, city
        - state, zip, country
        - 123 Main St patterns
        - City, STATE ZIP patterns
        """)

if __name__ == "__main__":
    main()