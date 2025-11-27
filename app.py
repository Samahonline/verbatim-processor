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
            'other', 'specify', 'additional', 'thought', 'idea',
            'q8', 'q9', 'ta', 'tb', 'tc', 'tf', 'ts', 's8', 's9'  # Added common question patterns
        ]
        
        # Only exclude columns with explicit NULL names, not content
        self.null_column_names = [
            'null', 'none', 'empty', 'missing', 'blank'
        ]
        
        # Keep date and address exclusion for column names only
        self.date_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'day', 'month', 'year', 'birth', 'dob', 'start', 'end',
            'submitted', 'completed', 'response_date'
        ]
        
        self.address_keywords = [
            'address', 'street', 'city', 'state', 'zip', 'postal', 'postcode',
            'country', 'location', 'county', 'province', 'region'
        ]

    def is_verbatim_column(self, column_name: str) -> bool:
        """
        More inclusive verbatim column detection
        """
        if not isinstance(column_name, str):
            return False
        
        column_lower = column_name.lower()
        
        # Check for verbatim keywords in column name
        for keyword in self.verbatim_keywords:
            if keyword in column_lower:
                return True
        
        # Check for common question patterns (Q, TA, TB, etc.)
        question_patterns = [
            r'^q\d', r'^ta\d', r'^tb\d', r'^tc\d', r'^tf\d', r'^ts\d',
            r'^s\d', r'^t\d', r'^[a-z]{1,2}\d+[a-z]*\d*'
        ]
        
        for pattern in question_patterns:
            if re.match(pattern, column_lower):
                return True
        
        # Check for text-like patterns
        text_patterns = [r'.*text.*', r'.*comment.*', r'.*response.*', r'.*answer.*']
        for pattern in text_patterns:
            if re.match(pattern, column_lower):
                return True
        
        return False

    def should_exclude_column_by_name(self, column_name: str) -> bool:
        """
        Only exclude columns based on name, not content
        """
        if not isinstance(column_name, str):
            return False
            
        col_lower = column_name.lower()
        
        # Only exclude if column name explicitly indicates NULL/date/address
        if any(null_name in col_lower for null_name in self.null_column_names):
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
        """
        More inclusive verbatim column detection
        Focus on column names and basic content checks
        """
        verbatim_columns = []
        excluded_columns = []
        
        for column in df.columns:
            column_name = str(column)
            
            # Skip only columns with explicit NULL/date/address names
            if self.should_exclude_column_by_name(column_name):
                excluded_columns.append((column_name, "Excluded by name"))
                continue
            
            # Skip if completely empty (no data at all)
            non_null_count = df[column].notna().sum()
            if non_null_count == 0:
                excluded_columns.append((column_name, "Completely empty"))
                continue
            
            # Check if it's a verbatim column by name patterns
            if self.is_verbatim_column(column_name):
                verbatim_columns.append(column)
                continue
            
            # For columns not matching verbatim patterns, check if they contain any text data
            if self._has_any_text_content(df[column]):
                verbatim_columns.append(column)
            else:
                excluded_columns.append((column_name, "No text content"))
        
        # Show excluded columns in sidebar
        if excluded_columns:
            with st.sidebar.expander("Excluded Columns", expanded=False):
                st.write(f"Excluded {len(excluded_columns)} columns:")
                for col, reason in excluded_columns:
                    st.write(f"❌ {col} ({reason})")
        
        return verbatim_columns

    def _has_any_text_content(self, series: pd.Series) -> bool:
        """
        Check if series has ANY text content at all
        Very inclusive - just check for non-empty strings
        """
        non_null_series = series.dropna()
        if non_null_series.empty:
            return False
        
        # Check if any value is a non-empty string
        for value in non_null_series.head(20):  # Check first 20 non-null values
            if isinstance(value, str) and value.strip():
                return True
        
        return False

    def clean_verbatim_value(self, value):
        """
        Preserve ALL text content, only convert actual missing values to empty strings
        """
        if pd.isna(value):
            return ""  # Convert actual missing values to empty string
        
        value_str = str(value).strip()
        
        # Return the original text as-is, preserving all characters
        return value_str

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[io.BytesIO, Dict]:
        try:
            intnr_column = self.find_intnr_column(df)
            verbatim_columns = self.detect_verbatim_columns(df)
            
            if not verbatim_columns:
                raise ValueError("No verbatim columns found. Please check your data format.")
            
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create formats
                empty_format = workbook.add_format({'font_color': '#999999'})
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Enhanced Summary sheet
                summary_data = {
                    'Sheet Name': [],
                    'Original Column': [],
                    'Total Sample Size': [],
                    'Responses Received': [],
                    'Missing Responses': [],
                    'Response Rate': []
                }
                
                for i, col in enumerate(verbatim_columns):
                    sheet_name = f"Q{i+1}_{col}"[:31]
                    total_samples = len(df)
                    responses_received = df[col].notna().sum()
                    missing_responses = df[col].isna().sum()
                    response_rate = (responses_received / total_samples) * 100
                    
                    summary_data['Sheet Name'].append(sheet_name)
                    summary_data['Original Column'].append(col)
                    summary_data['Total Sample Size'].append(total_samples)
                    summary_data['Responses Received'].append(responses_received)
                    summary_data['Missing Responses'].append(missing_responses)
                    summary_data['Response Rate'].append(f"{response_rate:.1f}%")
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                summary_sheet.set_column('A:A', 20)
                summary_sheet.set_column('B:B', 30)
                summary_sheet.set_column('C:F', 15)
                
                # Individual question sheets - ALL SAMPLES INCLUDED
                for i, verbatim_col in enumerate(verbatim_columns):
                    sheet_name = f"Q{i+1}_{verbatim_col}"[:31]
                    
                    # Create dataframe with ALL samples
                    result_df = df[[intnr_column, verbatim_col]].copy()
                    
                    # Clean responses but preserve all content
                    result_df['Cleaned_Response'] = result_df[verbatim_col].apply(self.clean_verbatim_value)
                    
                    # Create final output
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
                    
                    worksheet.set_column('A:A', 15)
                    worksheet.set_column('B:B', 50)
                    
                    # Add auto-filter
                    worksheet.autofilter(0, 0, len(final_df), 1)
                
                # Sample Tracking Sheet
                tracking_data = {
                    intnr_column: df[intnr_column],
                    'Total_Samples': len(df)
                }
                tracking_df = pd.DataFrame(tracking_data)
                tracking_df.to_excel(writer, sheet_name='Sample_Tracking', index=False)
                
                tracking_sheet = writer.sheets['Sample_Tracking']
                for col_num, value in enumerate(tracking_df.columns.values):
                    tracking_sheet.write(0, col_num, value, header_format)
                tracking_sheet.set_column('A:B', 15)
            
            output.seek(0)
            
            return output, {
                'status': 'success',
                'intnr_column': intnr_column,
                'verbatim_columns': verbatim_columns,
                'sheets_created': len(verbatim_columns) + 2,
                'total_samples': len(df),
                'total_questions': len(verbatim_columns)
            }
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

def main():
    st.title("📊 Verbatim Data Processor")
    st.markdown("""
    **Upload your Excel file to automatically extract and organize verbatim text responses**
    
    *Improved Detection: Now more inclusive of all question patterns*
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel File", 
        type=['xlsx', 'xls'],
        help="Upload your survey data Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
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
            with st.expander("📋 Data Preview (First 5 rows)"):
                st.dataframe(df.head())
            
            # Initialize processor
            processor = VerbatimProcessor()
            
            # Detect columns
            st.subheader("🔍 Column Analysis")
            intnr_column = processor.find_intnr_column(df)
            verbatim_columns = processor.detect_verbatim_columns(df)
            
            st.metric("ID Column", intnr_column)
            st.metric("Verbatim Columns Found", len(verbatim_columns))
            
            if verbatim_columns:
                st.success("✅ Verbatim columns detected:")
                
                # Display columns in a structured way
                for i, col in enumerate(verbatim_columns, 1):
                    response_count = df[col].notna().sum()
                    missing_count = df[col].isna().sum()
                    response_rate = (response_count / len(df)) * 100
                    
                    with st.expander(f"{i}. {col} ({response_count} responses, {response_rate:.1f}%)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Responses:** {response_count}")
                        with col2:
                            st.write(f"**Missing:** {missing_count}")
                        with col3:
                            st.write(f"**Rate:** {response_rate:.1f}%")
                        
                        # Show sample responses
                        sample_responses = df[col].dropna().head(3)
                        if len(sample_responses) > 0:
                            st.write("**Sample responses:**")
                            for resp in sample_responses:
                                st.write(f"- `{resp}`")
                
                # Process button
                if st.button("🚀 Process Data & Generate Excel", type="primary", use_container_width=True):
                    with st.spinner("Processing data and generating Excel file..."):
                        try:
                            output_bytes, result_info = processor.process_dataframe(df)
                            
                            st.success("✅ Processing completed!")
                            
                            # Results summary
                            st.subheader("📊 Processing Results")
                            
                            results_col1, results_col2, results_col3 = st.columns(3)
                            with results_col1:
                                st.metric("Total Samples", result_info['total_samples'])
                            with results_col2:
                                st.metric("Questions", result_info['total_questions'])
                            with results_col3:
                                st.metric("Excel Sheets", result_info['sheets_created'])
                            
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
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
            else:
                st.error("No verbatim columns detected.")
                st.info("""
                **If columns are missing that should be included:**
                - The app now uses more inclusive detection
                - Common patterns like Q8e_1, TA9C5, S8, etc. should be detected
                - Only explicitly named NULL/date/address columns are excluded
                """)
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload an Excel file to get started")

if __name__ == "__main__":
    main()
