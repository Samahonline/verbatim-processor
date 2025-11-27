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
            'q8', 'q9', 'ta', 'tb', 'tc', 'tf', 'ts', 's8', 's9'
        ]
        
        # Enhanced NULL indicators including Excel error values
        self.null_indicators = [
            'null', 'none', 'empty', 'n/a', 'na', 'missing', 
            'no data', 'no response', 'blank', '#null!', '#null',
            '#n/a', '#value!', '#ref!', '#div/0!', '#name?',
            'system missing', 'sysmiss', '.', '-', '--'
        ]
        
        self.date_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated', 'modified',
            'day', 'month', 'year', 'birth', 'dob', 'start', 'end',
            'submitted', 'completed', 'response_date'
        ]
        
        self.address_keywords = [
            'address', 'street', 'city', 'state', 'zip', 'postal', 'postcode',
            'country', 'location', 'county', 'province', 'region'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        ]

    def is_verbatim_column(self, column_name: str) -> bool:
        """
        Check if column name indicates verbatim content
        """
        if not isinstance(column_name, str):
            return False
        
        column_lower = column_name.lower()
        
        # Check for verbatim keywords in column name
        for keyword in self.verbatim_keywords:
            if keyword in column_lower:
                return True
        
        # Check for common question patterns
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

    def is_numeric_column(self, series: pd.Series) -> bool:
        """
        Ultra-aggressive numeric detection for all types of numeric data
        """
        if series.empty:
            return False
        
        # Check if pandas numeric type
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return False
        
        # Sample more data for better detection
        sample_size = min(500, len(non_null_series))
        sample = non_null_series.head(sample_size)
        
        numeric_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            
            # Check if value is numeric type
            if isinstance(value, (int, float, np.number)):
                numeric_count += 1
            elif isinstance(value, str):
                value_str = str(value).strip()
                
                # Skip system NULL values
                if self._is_system_null(value_str):
                    continue
                
                # Ultra-aggressive numeric detection
                if self._is_definitely_numeric(value_str):
                    numeric_count += 1
        
        # If more than 50% is numeric, exclude the column (very aggressive threshold)
        numeric_ratio = numeric_count / total_count if total_count > 0 else 0
        return numeric_ratio > 0.5

    def _is_definitely_numeric(self, value_str: str) -> bool:
        """
        Ultra-aggressive numeric detection for all numeric formats
        """
        # Remove any whitespace
        value_str = value_str.strip()
        
        # Handle empty strings
        if not value_str:
            return False
        
        # Remove any surrounding parentheses (common in some data formats)
        value_str = value_str.strip('()')
        
        # Check for pure numeric patterns first (most common)
        if value_str.isdigit():
            return True
            
        # Check for simple integers with signs
        if value_str in ['0', '+0', '-0']:
            return True
            
        # Enhanced numeric patterns
        patterns = [
            r'^[+-]?\d+$',                          # Integers with optional sign
            r'^[+-]?\d*\.\d+$',                     # Decimals: 1.5, .5, 0.5
            r'^[+-]?\d+\.\d*$',                     # Decimals: 1., 1.5
            r'^[+-]?\d+\.\d+e[+-]?\d+$',            # Scientific notation
            r'^[+-]?\.\d+$',                        # Decimals without leading zero
            r'^[+-]?\d+\.$',                        # Decimals without trailing digits
            r'^[+-]?\d{1,3}(,\d{3})*(\.\d*)?$',     # Numbers with commas
            r'^[+-]?\d*\.?\d+%$',                   # Percentages
            r'^\d+\.\d+\.\d+$',                     # Version numbers (treat as numeric)
            r'^[+-]?\d+/\d+$',                      # Fractions: 1/2, 3/4
        ]
        
        # Remove commas for pattern matching
        clean_value = value_str.replace(',', '')
        
        for pattern in patterns:
            if re.match(pattern, clean_value, re.IGNORECASE):
                return True
        
        # Try direct conversion (most reliable method)
        try:
            # Remove percentage signs, currency symbols, etc.
            test_value = clean_value.replace('%', '').replace('$', '').replace('€', '').replace('£', '').replace('¥', '')
            # Handle fractions
            if '/' in test_value:
                parts = test_value.split('/')
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    return True
            # Try float conversion
            float(test_value)
            return True
        except (ValueError, TypeError):
            pass
        
        # Additional checks for edge cases
        # Check if it's a numeric string with some non-numeric characters at the ends
        if len(value_str) > 1:
            # Remove non-numeric characters from start and end
            stripped = value_str.strip(' \t\n\r$€£¥%()[]{}')
            if stripped and (stripped.isdigit() or self._is_decimal_number(stripped)):
                return True
        
        return False

    def _is_decimal_number(self, value_str: str) -> bool:
        """Check if string is a decimal number"""
        try:
            # Count decimal points - should be exactly 1
            if value_str.count('.') == 1:
                parts = value_str.split('.')
                # Both parts should be digits or empty (like .5 or 1.)
                if (not parts[0] or parts[0].lstrip('-+').isdigit()) and (not parts[1] or parts[1].isdigit()):
                    return True
        except:
            pass
        return False

    def is_date_column(self, series: pd.Series, column_name: str = "") -> bool:
        """
        Check if column contains date data
        """
        if series.empty:
            return False
        
        # Check column name first
        if column_name and any(date_keyword in str(column_name).lower() 
                             for date_keyword in self.date_keywords):
            return True
        
        # Check if pandas datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return False
        
        sample_size = min(50, len(non_null_series))
        sample = non_null_series.head(sample_size)
        
        date_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            value_str = str(value).strip()
            
            # Skip numeric values that might be mistaken for dates
            if self._is_definitely_numeric(value_str):
                continue
            
            # Check for date patterns
            if self._looks_like_date(value_str):
                date_count += 1
            # Try pandas date parsing
            elif self._can_parse_as_date(value_str):
                date_count += 1
        
        return (date_count / total_count) > 0.7 if total_count > 0 else False

    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date"""
        for pattern in self.date_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    def _can_parse_as_date(self, value: str) -> bool:
        """Try to parse as date using pandas"""
        try:
            parsed = pd.to_datetime(value, errors='coerce')
            return not pd.isna(parsed)
        except:
            return False

    def is_null_column(self, series: pd.Series, column_name: str = "") -> bool:
        """
        Check if column is mostly NULL values including #NULL!
        """
        if series.empty:
            return True
        
        # Check if all values are NaN
        if series.isna().all():
            return True
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return True
        
        # Sample the non-null values to check for #NULL! and other system missing
        sample_size = min(100, len(non_null_series))
        sample = non_null_series.head(sample_size)
        
        null_like_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            value_str = str(value).strip()
            
            # Check for NULL indicators including Excel errors
            if self._is_system_null(value_str):
                null_like_count += 1
        
        # If more than 90% of non-null values are NULL-like, exclude the column
        null_ratio = null_like_count / total_count if total_count > 0 else 0
        return null_ratio > 0.9

    def should_exclude_column(self, df: pd.DataFrame, column_name: str) -> Tuple[bool, str]:
        """
        Check if column should be excluded with reason
        """
        series = df[column_name]
        
        # Check column name exclusions
        col_lower = str(column_name).lower()
        if any(null_indicator in col_lower for null_indicator in self.null_indicators):
            return True, "NULL column name"
        if any(date_keyword in col_lower for date_keyword in self.date_keywords):
            return True, "Date column name"
        if any(address_keyword in col_lower for address_keyword in self.address_keywords):
            return True, "Address column name"
        
        # Check content exclusions
        if self.is_null_column(series, column_name):
            return True, "NULL content (#NULL! values)"
        
        if self.is_numeric_column(series):
            return True, "Numeric column (including decimals)"
        
        if self.is_date_column(series, column_name):
            return True, "Date content"
        
        return False, ""

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
        Smart verbatim column detection with proper filtering
        """
        verbatim_columns = []
        excluded_columns = []
        
        for column in df.columns:
            column_name = str(column)
            
            # Check if column should be excluded
            should_exclude, reason = self.should_exclude_column(df, column_name)
            if should_exclude:
                excluded_columns.append((column_name, reason))
                continue
            
            # Check if it matches verbatim patterns
            if self.is_verbatim_column(column_name):
                verbatim_columns.append(column)
                continue
            
            # For other columns, check if they have meaningful text content
            if self._has_meaningful_text_content(df[column]):
                verbatim_columns.append(column)
            else:
                excluded_columns.append((column_name, "No meaningful text"))
        
        # Show excluded columns
        if excluded_columns:
            with st.sidebar.expander("🚫 Excluded Columns", expanded=True):
                st.write(f"Excluded {len(excluded_columns)} columns:")
                for col, reason in excluded_columns:
                    st.write(f"❌ {col} ({reason})")
        
        return verbatim_columns

    def _has_meaningful_text_content(self, series: pd.Series) -> bool:
        """
        Check if series has meaningful text content (not just numbers or system NULLs)
        """
        non_null_series = series.dropna()
        if non_null_series.empty:
            return False
        
        # Check first 30 non-null values
        sample = non_null_series.head(30)
        
        text_count = 0
        total_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            total_count += 1
            value_str = str(value).strip()
            
            # Skip system NULL values and Excel errors
            if self._is_system_null(value_str):
                continue
            
            # Skip numeric values
            if self._is_definitely_numeric(value_str):
                continue
            
            # Check if it's text (contains letters or meaningful text patterns)
            if any(char.isalpha() for char in value_str):
                text_count += 1
            elif len(value_str) > 3 and any(char.isalnum() for char in value_str):
                # Longer strings with alphanumeric content
                text_count += 1
        
        return (text_count / total_count) > 0.3 if total_count > 0 else False

    def _is_system_null(self, value_str: str) -> bool:
        """
        Check if a value is a system NULL value including #NULL!
        """
        value_lower = value_str.lower()
        
        # Exact matches
        if value_lower in [null.lower() for null in self.null_indicators]:
            return True
        
        # Excel error values
        if value_str.startswith('#') and any(error in value_lower for error in ['null', 'n/a', 'value', 'ref', 'div', 'name']):
            return True
        
        # Empty or very short meaningless values
        if value_str == '' or (len(value_str) <= 2 and not any(c.isalpha() for c in value_str)):
            return True
        
        # System missing values
        if value_str in ['.', '-', '--']:
            return True
        
        return False

    def clean_verbatim_value(self, value):
        """
        Clean verbatim values - preserve real text, filter system NULLs including #NULL!
        """
        if pd.isna(value):
            return ""
        
        value_str = str(value).strip()
        
        # Filter out system NULL values including #NULL!
        if self._is_system_null(value_str):
            return ""
        
        # Filter out numeric values
        if self._is_definitely_numeric(value_str):
            return ""
        
        return value_str

    def process_dataframe(self, df: pd.DataFrame) -> Tuple[io.BytesIO, Dict]:
        try:
            intnr_column = self.find_intnr_column(df)
            verbatim_columns = self.detect_verbatim_columns(df)
            
            if not verbatim_columns:
                raise ValueError("No valid verbatim columns found.")
            
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
                
                # Summary sheet
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
                    
                    # Calculate actual responses (excluding system NULLs and numeric values)
                    response_count = 0
                    for value in df[col]:
                        if pd.isna(value):
                            continue
                        value_str = str(value).strip()
                        if (not self._is_system_null(value_str) and 
                            not self._is_definitely_numeric(value_str)):
                            response_count += 1
                    
                    missing_count = total_samples - response_count
                    response_rate = (response_count / total_samples) * 100
                    
                    summary_data['Sheet Name'].append(sheet_name)
                    summary_data['Original Column'].append(col)
                    summary_data['Total Sample Size'].append(total_samples)
                    summary_data['Responses Received'].append(response_count)
                    summary_data['Missing Responses'].append(missing_count)
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
                
                # Individual question sheets
                for i, verbatim_col in enumerate(verbatim_columns):
                    sheet_name = f"Q{i+1}_{verbatim_col}"[:31]
                    
                    # Create dataframe with ALL samples
                    result_df = df[[intnr_column, verbatim_col]].copy()
                    
                    # Clean responses - filter out #NULL!, system NULLs, and numeric values
                    result_df['Cleaned_Response'] = result_df[verbatim_col].apply(self.clean_verbatim_value)
                    
                    # Create final output
                    final_df = result_df[[intnr_column, 'Cleaned_Response']].copy()
                    final_df.columns = [intnr_column, f'Response_{verbatim_col}']
                    
                    # Write to Excel
                    final_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format worksheet
                    worksheet = writer.sheets[sheet_name]
                    for col_num, value in enumerate(final_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Format empty cells
                    for row_num in range(1, len(final_df) + 1):
                        response_value = final_df.iloc[row_num-1, 1]
                        if response_value == "":
                            worksheet.write(row_num, 1, "", empty_format)
                    
                    worksheet.set_column('A:A', 15)
                    worksheet.set_column('B:B', 50)
                    worksheet.autofilter(0, 0, len(final_df), 1)
            
            output.seek(0)
            
            return output, {
                'status': 'success',
                'intnr_column': intnr_column,
                'verbatim_columns': verbatim_columns,
                'sheets_created': len(verbatim_columns) + 1,
                'total_samples': len(df),
                'total_questions': len(verbatim_columns)
            }
            
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

def main():
    st.title("📊 Verbatim Data Processor")
    st.markdown("""
    **Upload your Excel file to automatically extract verbatim text responses**
    
    *Enhanced filtering: Now properly excludes numeric columns (including decimals), dates, and #NULL! values*
    """)
    
    uploaded_file = st.file_uploader(
        "Choose Excel File", 
        type=['xlsx', 'xls'],
        help="Upload your survey data Excel file"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Reading Excel file..."):
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
            
            with st.expander("📋 Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))
            
            processor = VerbatimProcessor()
            
            st.subheader("🔍 Column Analysis")
            intnr_column = processor.find_intnr_column(df)
            verbatim_columns = processor.detect_verbatim_columns(df)
            
            st.metric("ID Column", intnr_column)
            st.metric("Verbatim Columns Found", len(verbatim_columns))
            
            if verbatim_columns:
                st.success("✅ Verbatim columns detected:")
                
                for i, col in enumerate(verbatim_columns, 1):
                    # Calculate actual responses excluding system NULLs and numeric values
                    response_count = 0
                    for value in df[col]:
                        if pd.isna(value):
                            continue
                        value_str = str(value).strip()
                        if (not processor._is_system_null(value_str) and 
                            not processor._is_definitely_numeric(value_str)):
                            response_count += 1
                    
                    missing_count = len(df) - response_count
                    response_rate = (response_count / len(df)) * 100
                    
                    with st.expander(f"{i}. {col} ({response_count} responses, {response_rate:.1f}%)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Responses:** {response_count}")
                        with col2:
                            st.write(f"**Missing:** {missing_count}")
                        with col3:
                            st.write(f"**Rate:** {response_rate:.1f}%")
                        
                        # Show sample non-NULL, non-numeric responses
                        sample_responses = []
                        for value in df[col].dropna().head(5):
                            value_str = str(value).strip()
                            if (not processor._is_system_null(value_str) and 
                                not processor._is_definitely_numeric(value_str)):
                                sample_responses.append(value_str)
                        
                        if sample_responses:
                            st.write("**Sample responses:**")
                            for resp in sample_responses:
                                st.write(f"- `{resp}`")
                        else:
                            st.write("*No valid text responses found*")
                
                if st.button("🚀 Process Data & Generate Excel", type="primary", use_container_width=True):
                    with st.spinner("Processing data and generating Excel file..."):
                        try:
                            output_bytes, result_info = processor.process_dataframe(df)
                            
                            st.success("✅ Processing completed!")
                            
                            st.subheader("📊 Processing Results")
                            
                            results_col1, results_col2, results_col3 = st.columns(3)
                            with results_col1:
                                st.metric("Total Samples", result_info['total_samples'])
                            with results_col2:
                                st.metric("Questions", result_info['total_questions'])
                            with results_col3:
                                st.metric("Excel Sheets", result_info['sheets_created'])
                            
                            st.subheader("💾 Download Results")
                            
                            original_name = uploaded_file.name
                            base_name = original_name.replace('.xlsx', '').replace('.xls', '')
                            download_name = f"{base_name}_verbatim_analysis.xlsx"
                            
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
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload an Excel file to get started")

if __name__ == "__main__":
    main()
