import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import List, Dict, Tuple
import base64

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
        
        self.null_indicators = [
            'null', 'none', 'empty', 'n/a', 'na', 'missing', 
            'no data', 'no response', 'blank'
        ]
        
        self.date_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'day', 'month', 'year', 'birth', 'dob', 'start', 'end',
            'submitted', 'completed', 'response_date'
        ]
        
        self.address_keywords = [
            'address', 'street', 'city', 'state', 'zip', 'postal', 'postcode',
            'country', 'location', 'county', 'province', 'region',
            'addr', 'st', 'road', 'ave', 'avenue', 'boulevard', 'blvd'
        ]

    def is_verbatim_column(self, column_name: str) -> bool:
        if not isinstance(column_name, str):
            return False
        
        column_lower = column_name.lower()
        for keyword in self.verbatim_keywords:
            if keyword in column_lower:
                return True
        
        text_patterns = [r'.*text.*', r'.*comment.*', r'.*response.*', r'.*answer.*']
        for pattern in text_patterns:
            if re.match(pattern, column_lower):
                return True
        
        return False

    def is_excluded_column(self, column_name: str) -> bool:
        """Check if column should be excluded based on name"""
        if not isinstance(column_name, str):
            return False
            
        col_lower = column_name.lower()
        
        # Check for exclusion keywords
        if any(null_indicator in col_lower for null_indicator in self.null_indicators):
            return True
        if any(date_keyword in col_lower for date_keyword in self.date_keywords):
            return True
        if any(address_keyword in col_lower for address_keyword in self.address_keywords):
            return True
            
        return False

    def find_intnr_column(self, df: pd.DataFrame) -> str:
        intnr_patterns = ['intnr', 'interview', 'respondent', 'id', 'caseid', 'resp_id']
        for col in df.columns:
            if not isinstance(col, str):
                continue
            col_lower = col.lower()
            for pattern in intnr_patterns:
                if pattern in col_lower:
                    return col
        
        for col in df.columns:
            col_str = str(col).lower()
            if any(pattern in col_str for pattern in ['id', 'num', 'code', 'case']):
                return col
        
        return df.columns[0]

    def detect_verbatim_columns(self, df: pd.DataFrame) -> List[str]:
        verbatim_columns = []
        excluded_columns = []
        
        for column in df.columns:
            column_name = str(column)
            
            # Skip excluded columns by name
            if self.is_excluded_column(column_name):
                excluded_columns.append((column_name, "Excluded by name"))
                continue
            
            # Skip if mostly empty (but keep if at least some responses)
            non_null_count = df[column].notna().sum()
            if non_null_count == 0:
                excluded_columns.append((column_name, "All NULL"))
                continue
                
            # Check if verbatim column
            if self.is_verbatim_column(column_name):
                verbatim_columns.append(column)
            else:
                # Sample to check if it's text data
                sample = df[column].dropna().head(10)
                if len(sample) > 0:
                    text_like = sum(1 for val in sample if isinstance(val, str) and len(str(val).strip()) > 10)
                    if text_like / len(sample) > 0.5:  # If >50% seems like text
                        verbatim_columns.append(column)
                    else:
                        excluded_columns.append((column_name, "Not text-like"))
                else:
                    excluded_columns.append((column_name, "No valid data"))
        
        # Show excluded columns in sidebar
        if excluded_columns:
            with st.sidebar.expander("Excluded Columns", expanded=False):
                st.write(f"Excluded {len(excluded_columns)} columns:")
                for col, reason in excluded_columns:
                    st.write(f"❌ {col} ({reason})")
        
        return verbatim_columns

    def clean_verbatim_value(self, value):
        """Clean individual verbatim values while preserving missing data structure"""
        if pd.isna(value):
            return ""  # Keep as empty string for missing responses
        
        value_str = str(value).strip()
        
        # Check for NULL indicators
        if any(null_indicator in value_str.lower() for null_indicator in self.null_indicators):
            return ""
            
        # Check for very short meaningless responses
        if len(value_str) < 3:
            return ""
            
        return value_str

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[io.BytesIO, Dict]:
        try:
            intnr_column = self.find_intnr_column(df)
            verbatim_columns = self.detect_verbatim_columns(df)
            
            if not verbatim_columns:
                raise ValueError("No valid verbatim columns found. Please check your data format.")
            
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create a format for empty/missing responses
                empty_format = workbook.add_format({'font_color': '#999999'})
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Summary sheet with response statistics
                summary_data = {
                    'Question Column': verbatim_columns,
                    'Total Sample Size': [len(df)] * len(verbatim_columns),
                    'Responses Received': [df[col].notna().sum() for col in verbatim_columns],
                    'Missing Responses': [df[col].isna().sum() for col in verbatim_columns],
                    'Response Rate': [f"{(df[col].notna().sum() / len(df) * 100):.1f}%" for col in verbatim_columns]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                summary_sheet.set_column('A:A', 30)
                summary_sheet.set_column('B:E', 15)
                
                # Individual question sheets - ALL SAMPLES INCLUDED
                for i, verbatim_col in enumerate(verbatim_columns):
                    sheet_name = f"Q{i+1}_{verbatim_col}"[:31]  # Excel sheet name limit
                    
                    # Create dataframe with ALL samples (including missing responses)
                    result_df = df[[intnr_column, verbatim_col]].copy()
                    
                    # Clean the verbatim responses but keep all rows
                    result_df['Cleaned_Response'] = result_df[verbatim_col].apply(self.clean_verbatim_value)
                    
                    # Create final output with original ID and cleaned response
                    final_df = result_df[[intnr_column, 'Cleaned_Response']].copy()
                    final_df.columns = [intnr_column, f'Response_{verbatim_col}']
                    
                    # Write to Excel
                    final_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format the worksheet
                    worksheet = writer.sheets[sheet_name]
                    
                    # Write headers with formatting
                    for col_num, value in enumerate(final_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Apply formatting to empty cells
                    for row_num in range(1, len(final_df) + 1):
                        response_value = final_df.iloc[row_num-1, 1]
                        if response_value == "":
                            worksheet.write(row_num, 1, "", empty_format)
                    
                    worksheet.set_column('A:A', 15)  # ID column
                    worksheet.set_column('B:B', 50)  # Response column
                    
                    # Add auto-filter
                    worksheet.autofilter(0, 0, len(final_df), 1)
                
                # Sample Tracking Sheet - Master list of all respondents
                tracking_data = {
                    intnr_column: df[intnr_column],
                    'In_Final_Analysis': ['Yes'] * len(df)  # All samples included
                }
                tracking_df = pd.DataFrame(tracking_data)
                tracking_df.to_excel(writer, sheet_name='Sample_Tracking', index=False)
                
                # Format tracking sheet
                tracking_sheet = writer.sheets['Sample_Tracking']
                for col_num, value in enumerate(tracking_df.columns.values):
                    tracking_sheet.write(0, col_num, value, header_format)
                tracking_sheet.set_column('A:A', 15)
                tracking_sheet.set_column('B:B', 15)
            
            output.seek(0)
            
            return output, {
                'status': 'success',
                'intnr_column': intnr_column,
                'verbatim_columns': verbatim_columns,
                'sheets_created': len(verbatim_columns) + 2,  # +2 for Summary and Tracking
                'total_samples': len(df),
                'total_questions': len(verbatim_columns)
            }
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

def main():
    st.title("📊 Verbatim Data Processor")
    st.markdown("""
    **Upload your Excel file to automatically extract and organize verbatim text responses**
    
    *New Feature: All samples are included in every sheet for consistent analysis*
    
    *Features:*
    - 🔍 Automatically detects verbatim columns
    - 🚫 Excludes NULL, date, and address columns
    - 📄 Creates separate sheets for each question
    - 👥 **Includes ALL samples (even missing responses)**
    - 💾 Downloads formatted Excel workbook
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel File", 
        type=['xlsx', 'xls'],
        help="Upload your survey data Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Read file with progress
            with st.spinner("Reading Excel file..."):
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully: {uploaded_file.name}")
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
            
            # Show preview
            with st.expander("📋 Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))
            
            # Initialize processor
            processor = VerbatimProcessor()
            
            # Detect columns
            st.subheader("🔍 Column Analysis")
            intnr_column = processor.find_intnr_column(df)
            verbatim_columns = processor.detect_verbatim_columns(df)
            
            if verbatim_columns:
                st.success(f"Found {len(verbatim_columns)} verbatim columns")
                
                # Display verbatim columns with statistics
                for i, col in enumerate(verbatim_columns, 1):
                    response_count = df[col].notna().sum()
                    missing_count = df[col].isna().sum()
                    response_rate = (response_count / len(df)) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{i}. {col}**")
                    with col2:
                        st.write(f"Responses: {response_count}")
                    with col3:
                        st.write(f"Missing: {missing_count}")
                    with col4:
                        st.write(f"Rate: {response_rate:.1f}%")
                
                # Process button
                if st.button("🚀 Process Data & Generate Excel", type="primary", use_container_width=True):
                    with st.spinner("Processing data and generating Excel file..."):
                        try:
                            output_bytes, result_info = processor.process_dataframe(df)
                            
                            st.success("✅ Processing completed!")
                            
                            # Results summary
                            st.subheader("📊 Processing Results")
                            
                            results_col1, results_col2, results_col3, results_col4 = st.columns(4)
                            with results_col1:
                                st.metric("Total Samples", result_info['total_samples'])
                            with results_col2:
                                st.metric("Questions", result_info['total_questions'])
                            with results_col3:
                                st.metric("Excel Sheets", result_info['sheets_created'])
                            with results_col4:
                                st.metric("ID Column", result_info['intnr_column'])
                            
                            # Download section
                            st.subheader("💾 Download Results")
                            
                            # Generate filename
                            original_name = uploaded_file.name
                            base_name = original_name.replace('.xlsx', '').replace('.xls', '')
                            download_name = f"{base_name}_verbatim_analysis.xlsx"
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Excel Workbook",
                                data=output_bytes,
                                file_name=download_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                help="Click to download the processed Excel workbook with all samples included"
                            )
                            
                            # Features explanation
                            with st.expander("📁 What's included in the download?"):
                                st.markdown("""
                                **Excel Workbook Contents:**
                                - 📊 **Summary Sheet**: Overview of all questions with response rates
                                - ❓ **Question Sheets**: One sheet per verbatim question (ALL samples included)
                                - 👥 **Sample Tracking**: Master list of all respondents
                                
                                **Key Features:**
                                - ✅ All samples included in every question sheet
                                - ✅ Consistent sample size across all analysis
                                - ✅ Missing responses shown as blank cells
                                - ✅ Response rates calculated for each question
                                - ✅ Auto-filters applied for easy analysis
                                """)
                            
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
                            st.info("Please check your data format and try again.")
            else:
                st.error("No verbatim columns detected. Please check your data format.")
                st.info("""
                **Tips for better detection:**
                - Ensure you have columns with text responses
                - Column names with words like 'verbatim', 'comment', 'text', 'response' work best
                - Avoid columns that are entirely empty or contain only dates/addresses
                - Make sure your data has some text responses
                """)
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure you've uploaded a valid Excel file.")
    
    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload an Excel file to get started")
        
        # Sample data section
        st.subheader("🎯 How it works with sample consistency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Before (Traditional Approach):**
            - Question 1: 850 samples (only respondents who answered)
            - Question 2: 720 samples (only respondents who answered)
            - Question 3: 910 samples (only respondents who answered)
            *→ Inconsistent sample sizes!*
            """)
        
        with col2:
            st.markdown("""
            **Now (Improved Approach):**
            - Question 1: 1,000 samples (ALL respondents)
            - Question 2: 1,000 samples (ALL respondents) 
            - Question 3: 1,000 samples (ALL respondents)
            *→ Consistent sample sizes!*
            """)
        
        # Sample data download
        st.subheader("📥 Need a sample file?")
        
        # Create sample data with missing responses to demonstrate
        sample_data = {
            'respondent_id': [1001, 1002, 1003, 1004, 1005, 1006],
            'verbatim_feedback': [
                'I really like the product design and user interface',
                'The customer service could be improved',
                None,  # Missing response
                'Overall satisfied but would like more features',
                'Excellent quality and fast delivery',
                ''  # Empty response
            ],
            'open_comments': [
                'The mobile app works very smoothly',
                None,  # Missing response
                'Better documentation would be helpful',
                'Price is reasonable for the value',
                '',  # Empty response
                'Great product overall'
            ],
            'additional_thoughts': [
                None,  # Missing
                'Would recommend to friends',
                'Good value for money',
                None,  # Missing
                'Easy to use interface',
                'Fast shipping'
            ],
            'response_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'],
            'customer_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Ave', '987 Cedar Ln']
        }
        sample_df = pd.DataFrame(sample_data)
        
        sample_output = io.BytesIO()
        with pd.ExcelWriter(sample_output, engine='xlsxwriter') as writer:
            sample_df.to_excel(writer, sheet_name='SurveyData', index=False)
        sample_output.seek(0)
        
        st.download_button(
            label="Download Sample Template with Missing Data",
            data=sample_output,
            file_name="sample_survey_data_with_missing.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="This sample file contains missing responses to demonstrate the new feature"
        )

if __name__ == "__main__":
    main()
