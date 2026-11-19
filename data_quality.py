
import pandas as pd
import streamlit as st


def show_data_quality(df):

    st.divider()
    st.header("🔍 Data Quality Inspection")

    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Overview
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", total_rows)
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", missing_values)
    col4.metric("Duplicate Rows", duplicate_rows)

    # Column-level inspection
    st.subheader("Column Quality Report")

    report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2),
        "Unique Values": df.nunique(dropna=True).values
    })

    st.dataframe(report, use_container_width=True)

    # Detect constant columns
    constant_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1
    ]

    # Detect leading/trailing whitespace
    whitespace_cols = []

    for col in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        values = df[col].dropna()

        has_whitespace = values.map(
            lambda x: isinstance(x, str) and x != x.strip()
        ).any()

        if has_whitespace:
            whitespace_cols.append(col)

    # Display detected issues
    st.subheader("Detected Data Quality Issues")

    issues_found = False

    if missing_values > 0:
        st.warning(
            f"{missing_values} missing values detected."
        )
        issues_found = True

    if duplicate_rows > 0:
        st.warning(
            f"{duplicate_rows} duplicate rows detected."
        )
        issues_found = True

    if constant_cols:
        st.info(
            "Constant columns detected: "
            + ", ".join(map(str, constant_cols))
        )
        issues_found = True

    if whitespace_cols:
        st.warning(
            "Leading/trailing whitespace detected in: "
            + ", ".join(map(str, whitespace_cols))
        )
        issues_found = True

    if not issues_found:
        st.success(
            "No issues detected by the current quality checks."
        )

    st.caption(
        "Note: These checks do not guarantee that every value "
        "in the dataset is correct."
    )
