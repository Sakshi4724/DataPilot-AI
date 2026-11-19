import hashlib

import json

import re

import pandas as pd

import streamlit as st

from ai_provider import generate_ai_text

# =========================================================

# 1. DATASET INSPECTION HELPERS

# =========================================================

def is_identifier(column_name):

    name = str(column_name).lower().replace("_", "").replace(" ", "")

    known_names = {

        "id", "customerid", "studentid", "userid", "productid",

        "orderid", "transactionid", "invoice", "invoiceno",

        "stockcode", "student", "customer", "product", "email"

    }

    return name in known_names or name.endswith("id") or "email" in name

def detect_datetime_values(series):

    """Return True when most non-empty sampled values look like dates."""

    values = series.dropna().astype(str).str.strip()

    if len(values) < 2:

        return False

    sample = values.head(100)

    pattern = (

        r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"

        r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"

    )

    matches = sample.str.fullmatch(pattern, case=False, na=False)

    return bool(matches.mean() >= 0.90)

def detect_identifier_columns(df):

    protected = []

    for col in df.columns:

        if detect_datetime_values(df[col]):

            continue

        if is_identifier(col):

            protected.append(col)

            continue

        if (

            pd.api.types.is_object_dtype(df[col])

            or pd.api.types.is_string_dtype(df[col])

        ):

            values = df[col].dropna().astype(str).str.strip()

            if len(values) >= 2 and values.nunique() / len(values) >= 0.95:

                protected.append(col)

    return protected

def detect_outliers_iqr(series):

    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) < 4:

        return None

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:

        return None

    lower = float(q1 - 1.5 * iqr)

    upper = float(q3 + 1.5 * iqr)

    count = int(((values < lower) | (values > upper)).sum())

    if count == 0:

        return None

    return {"count": count, "lower": lower, "upper": upper}

def detect_invalid_numeric_values(df):

    issues = {}

    for col in df.select_dtypes(include="number").columns:

        name = str(col).lower().replace("_", "").replace(" ", "")

        values = df[col].dropna()

        if values.empty:

            continue

        mask = pd.Series(False, index=values.index)

        reason = None

        if "age" in name:

            mask = values < 0

            reason = "Age values should not be negative."

        elif any(word in name for word in ("mark", "score", "attendance")):

            mask = values < 0

            reason = "This field is not expected to be negative."

        if any(word in name for word in ("percentage", "percent", "attendance")):

            mask = (values < 0) | (values > 100)

            reason = "Percentage values are normally between 0 and 100."

        invalid = values[mask]

        if not invalid.empty:

            issues[col] = {

                "count": int(len(invalid)),

                "examples": invalid.head(5).tolist(),

                "reason": reason

            }

    return issues

def detect_constant_columns(df):

    result = {}

    for col in df.columns:

        values = df[col].dropna()

        if not values.empty and values.nunique() == 1:

            result[col] = {

                "value": str(values.iloc[0]),

                "count": int(len(values))

            }

    return result

def inspect_dataset(df):

    return {

        "rows": int(len(df)),

        "columns": int(len(df.columns)),

        "missing_values": int(df.isna().sum().sum()),

        "duplicate_rows": int(df.duplicated().sum()),

        "invalid_numeric_values": detect_invalid_numeric_values(df),

        "constant_columns": detect_constant_columns(df),

        "column_details": [

            {

                "column": str(col),

                "data_type": str(df[col].dtype),

                "missing": int(df[col].isna().sum()),

                "unique": int(df[col].nunique(dropna=True))

            }

            for col in df.columns

        ]

    }

# =========================================================

# 2. PREPROCESSING PLAN

# =========================================================

def create_preprocessing_plan(df, observations):

    plan = []

    protected = detect_identifier_columns(df)

    # Safe text cleanup

    text_cols = df.select_dtypes(include=["object", "string"]).columns

    columns_with_spaces = []

    for col in text_cols:

        sample = df[col].dropna().astype(str)

        if not sample.empty and sample.map(lambda x: x != x.strip()).any():

            columns_with_spaces.append(col)

    if columns_with_spaces:

        plan.append({

            "type": "safe",

            "action": "trim_whitespace",

            "description": "Trim extra spaces in: " + ", ".join(map(str, columns_with_spaces))

        })

    if observations["duplicate_rows"] > 0:

        plan.append({

            "type": "approval",

            "action": "remove_duplicates",

            "description": f"Remove {observations['duplicate_rows']:,} completely duplicated rows."

        })

    for col, info in observations["invalid_numeric_values"].items():

        plan.append({

            "type": "approval",

            "action": "review_invalid_numeric",

            "column": col,

            "description": (

                f"{info['count']:,} potentially invalid values in '{col}'. "

                f"Examples: {info['examples']}. {info['reason']} "

                "Review is recommended; values will not be changed automatically."

            )

        })

    for col, info in observations["constant_columns"].items():

        plan.append({

            "type": "approval",

            "action": "review_constant_column",

            "column": col,

            "description": (

                f"'{col}' contains one non-missing value ({info['value']}). "

                "Review whether this column is useful. It will not be deleted automatically."

            )

        })

    numeric_cols = [

        col for col in df.select_dtypes(include="number").columns

        if col not in protected

    ]

    for col in numeric_cols:

        missing = int(df[col].isna().sum())

        if missing:

            values = df[col].dropna()

            strategy = "median"

            if len(values) >= 2 and abs(values.skew()) <= 1:

                strategy = "mean"

            plan.append({

                "type": "approval",

                "action": f"fill_numeric_{strategy}",

                "column": col,

                "description": f"Fill {missing:,} missing values in '{col}' using the {strategy}."

            })

        outlier_info = detect_outliers_iqr(df[col])

        if outlier_info:

            plan.append({

                "type": "approval",

                "action": "cap_numeric_outliers",

                "column": col,

                "lower": outlier_info["lower"],

                "upper": outlier_info["upper"],

                "description": (

                    f"Found {outlier_info['count']:,} potential outliers in '{col}'. "

                    f"Cap values outside {outlier_info['lower']:.2f} to {outlier_info['upper']:.2f}."

                )

            })

    # Text date columns

    for col in text_cols:

        if detect_datetime_values(df[col]):

            plan.append({

                "type": "approval",

                "action": "convert_datetime",

                "column": col,

                "description": f"Convert '{col}' from text to datetime format."

            })

    # Categorical missing values and case differences

    categorical_cols = [

        col for col in text_cols

        if col not in protected and not detect_datetime_values(df[col])

    ]

    for col in categorical_cols:

        series = df[col]

        missing = int(series.isna().sum())

        if missing:

            ratio = missing / max(len(series), 1)

            if ratio > 0.50:

                plan.append({

                    "type": "warning",

                    "description": f"'{col}' has {ratio:.1%} missing values. Review manually instead of automatic filling."

                })

            else:

                values = series.dropna()

                mode = values.mode()

                if not mode.empty and (values == mode.iloc[0]).mean() >= 0.60:

                    action = "fill_categorical_mode"

                    description = f"Fill {missing:,} missing values in '{col}' with its most frequent category ({mode.iloc[0]})."

                    fill_value = mode.iloc[0]

                else:

                    action = "fill_categorical_unknown"

                    description = f"Fill {missing:,} missing values in '{col}' with 'Unknown' because no category dominates."

                    fill_value = None

                plan.append({

                    "type": "approval",

                    "action": action,

                    "column": col,

                    "value": fill_value,

                    "description": description

                })

        # Detect variants that differ only by case/outer spaces

        normalized = series.dropna().astype(str).str.strip().str.casefold()

        original = series.dropna().astype(str).str.strip()

        variants = pd.DataFrame({"normalized": normalized, "original": original})

        inconsistent_groups = variants.groupby("normalized")["original"].nunique()

        inconsistent_count = int((inconsistent_groups > 1).sum())

        if inconsistent_count:

            plan.append({

                "type": "approval",

                "action": "standardize_category_case",

                "column": col,

                "description": f"Found {inconsistent_count} category groups in '{col}' that differ only by case or outer spaces. Standardize equivalent labels."

            })

    return plan

# =========================================================

# 3. QUICK LOCAL QWEN PLAN REVIEW

# =========================================================

def generate_plan_reasoning(plan, df):
    """Generate and validate one AI review per approval-required operation."""

    approval_items = [
        item for item in plan
        if item["type"] == "approval"
    ]

    if not approval_items:
        return {}, "No approval-required operations were found."

    
    # Prepare a compact dataset profile for AI review
    dataset_context = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_profiles": {}
    }

    for column in df.columns:
        series = df[column]
        profile = {
            "data_type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "sample_values": [
                str(value)[:80]
                for value in series.dropna().head(5).tolist()
            ]
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()

            if not numeric.empty:
                profile.update({
                    "minimum": round(float(numeric.min()), 3),
                    "maximum": round(float(numeric.max()), 3),
                    "mean": round(float(numeric.mean()), 3),
                    "median": round(float(numeric.median()), 3)
                })

        dataset_context["column_profiles"][str(column)] = profile

    reviews = {}
    raw_responses = []

    for index, item in enumerate(approval_items):


        
        
        column_name = str(item.get("column", "Dataset"))

        column_profile = dataset_context["column_profiles"].get(
            column_name, {}
        )

        prompt = f"""
Review this ONE data preprocessing operation.

Dataset size: {dataset_context["rows"]} rows
Column: {column_name}
Column information:
{json.dumps(column_profile, ensure_ascii=False, default=str)}

Operation:
{item["description"]}

Write ONE dataset-specific review sentence.

Explain a realistic risk or reason for approving this operation.
Use the actual column statistics and sample values.
Use simple professional English.
Maximum 35 words.

IMPORTANT:
Return ONLY one plain-text review sentence.
Do not return JSON.
Do not repeat the dataset information or operation details.
"""



        system_prompt = (
            "Return only one plain-text review sentence. "
            "Never return JSON."
        )

        try:
            raw_text = generate_ai_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=150
            )
        except Exception as e:
            print("AI review generation failed:", str(e))
            raw_text = ""
        print("\n========== DATAPILOT AI DEBUG ==========")
        print("COLUMN BEING REVIEWED:", column_name)
        print("PROMPT SENT TO QWEN:\n", prompt)
        print("RAW RESPONSE FROM QWEN:\n", raw_text)
        print("========================================\n")
        raw_responses.append(
            f"OPERATION {index}:\n{raw_text}"
        )

        
        review = re.sub(
            r"<think>.*?</think>",
            "",
            raw_text,
            flags=re.DOTALL
        ).strip()

        review = re.sub(
            r"^```(?:text)?\s*|\s*```$",
            "",
            review,
            flags=re.IGNORECASE
        ).strip()

        # Reject JSON or copied operation details
        if review.startswith("{") or review.startswith("["):
            continue

        if (
            review
            and len(review.split()) <= 40
            and "\n" not in review
        ):
            reviews[index] = review


    raw_text = "\n\n".join(raw_responses)

    if len(reviews) != len(approval_items):
        raise ValueError(
            f"Qwen returned {len(reviews)} valid review(s) "
            f"for {len(approval_items)} operation(s).\n\n"
            "RAW QWEN RESPONSE:\n" + raw_text[:5000]
        )

    return reviews, raw_text


# 4. EXECUTE ONLY APPROVED OPERATIONS

# =========================================================

def execute_preprocessing(df, plan, approved_actions):

    result = df.copy()

    log = []

    # Safe operation: trim whitespace

    for item in plan:

        if item["type"] == "safe" and item["action"] == "trim_whitespace":

            changed = 0

            for col in result.select_dtypes(include=["object", "string"]).columns:

                before = result[col].copy()

                result[col] = result[col].map(

                    lambda value: value.strip() if isinstance(value, str) else value

                )

                changed += int((before != result[col]).fillna(False).sum())

            log.append(f"Automatically trimmed whitespace in {changed:,} text values.")

    for item in plan:

        if item["type"] != "approval":

            continue

        action = item["action"]

        col = item.get("column")

        if (action, col) not in approved_actions:

            continue

        if action == "remove_duplicates":

            before = len(result)

            result = result.drop_duplicates()

            log.append(f"Removed {before - len(result):,} duplicate rows.")

        elif action in ("fill_numeric_mean", "fill_numeric_median"):

            if col in result.columns:

                missing = int(result[col].isna().sum())

                fill_value = result[col].mean() if action.endswith("mean") else result[col].median()

                if missing and pd.notna(fill_value):

                    result[col] = result[col].fillna(fill_value)

                    log.append(f"Filled {missing:,} missing values in '{col}' using {action.replace('fill_numeric_', '')}.")

        elif action in ("fill_categorical_mode", "fill_categorical_unknown"):

            if col in result.columns:

                missing = int(result[col].isna().sum())

                fill_value = item.get("value") if action.endswith("mode") else "Unknown"

                if missing and fill_value is not None:

                    result[col] = result[col].fillna(fill_value)

                    log.append(f"Filled {missing:,} missing values in '{col}'.")

        elif action == "cap_numeric_outliers" and col in result.columns:

            before = result[col].copy()

            result[col] = result[col].clip(lower=item["lower"], upper=item["upper"])

            changed = int((before != result[col]).fillna(False).sum())

            log.append(f"Capped {changed:,} potential outlier values in '{col}'.")

        elif action == "convert_datetime" and col in result.columns:

            converted = pd.to_datetime(result[col], errors="coerce", dayfirst=True)

            original_count = int(result[col].notna().sum())

            if int(converted.notna().sum()) == original_count:

                result[col] = converted

                log.append(f"Converted '{col}' to datetime.")

            else:

                log.append(f"Skipped '{col}' datetime conversion because some values could not be parsed safely.")

        elif action == "standardize_category_case" and col in result.columns:

            values = result[col].dropna().astype(str)

            groups = {}

            for value in values:

                groups.setdefault(value.strip().casefold(), [])

                if value.strip() not in groups[value.strip().casefold()]:

                    groups[value.strip().casefold()].append(value.strip())

            mapping = {}

            for variants in groups.values():

                if len(variants) > 1:

                    counts = values.value_counts()

                    canonical = max(variants, key=lambda value: (counts.get(value, 0), value.istitle()))

                    for variant in variants:

                        mapping[variant] = canonical

            before = result[col].copy()

            result[col] = result[col].map(

                lambda value: mapping.get(value.strip(), value) if isinstance(value, str) else value

            )

            changed = int((before != result[col]).fillna(False).sum())

            log.append(f"Standardized {changed:,} category values in '{col}'.")

        elif action == "review_invalid_numeric":

            log.append(f"Reviewed potentially invalid numeric values in '{col}'. No values were changed.")

        elif action == "review_constant_column":

            log.append(f"Reviewed constant column '{col}'. No column was deleted.")

    return result, log

# =========================================================

# 5. BEFORE / AFTER COMPARISON

# =========================================================

def show_comparison(original, processed):

    comparison = pd.DataFrame({

        "Metric": ["Total Rows", "Total Columns", "Missing Values", "Duplicate Rows"],

        "Before": [

            len(original),

            len(original.columns),

            int(original.isna().sum().sum()),

            int(original.duplicated().sum())

        ],

        "After": [

            len(processed),

            len(processed.columns),

            int(processed.isna().sum().sum()),

            int(processed.duplicated().sum())

        ]

    })

    st.dataframe(comparison, width="stretch", hide_index=True)

# =========================================================

# 6. STREAMLIT AGENT INTERFACE

# =========================================================

def show_preprocessing_agent(df, dataset_key=None):

    st.divider()

    st.header("🤖 Data Preprocessing Agent")

    st.write(

        "The agent inspects your dataset and prepares a preprocessing plan. "

        "Safe cleanup is automatic; other changes require your approval."

    )

    if df is None or df.empty:

        st.warning("Please upload a dataset containing at least one row.")

        return df

    observations = inspect_dataset(df)

    st.subheader("1. Dataset Observations")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{observations['rows']:,}")

    c2.metric("Columns", f"{observations['columns']:,}")

    c3.metric("Missing Values", f"{observations['missing_values']:,}")

    c4.metric("Duplicate Rows", f"{observations['duplicate_rows']:,}")

    with st.expander("View column-level observations"):

        st.dataframe(pd.DataFrame(observations["column_details"]), width="stretch", hide_index=True)

    st.subheader("2. Data Quality Checks")

    invalid = observations["invalid_numeric_values"]

    if invalid:

        for col, info in invalid.items():

            st.warning(f"**{col}:** {info['count']:,} potentially invalid values. {info['reason']} Examples: {info['examples']}")

    else:

        st.success("No invalid numeric values detected by the current rules.")

    constants = observations["constant_columns"]

    if constants:

        for col, info in constants.items():

            st.info(f"**{col}:** one unique non-missing value ({info['value']}).")

    else:

        st.success("No constant columns detected.")

    protected = detect_identifier_columns(df)

    if protected:

        st.caption("Protected identifier columns (not automatically imputed): " + ", ".join(map(str, protected)))

    plan = create_preprocessing_plan(df, observations)

    if not plan:

        st.success("No preprocessing issues were detected by the current rules.")

        return df

    st.subheader("3. Agent Preprocessing Plan")

    approval_items = [item for item in plan if item["type"] == "approval"]

    plan_signature = hashlib.md5(

        json.dumps(plan, sort_keys=True, default=str).encode("utf-8")

    ).hexdigest()[:10]

    review_key = f"ai_plan_review_{dataset_key or 'default'}_{plan_signature}"

    review_attempted_key = f"ai_plan_review_attempted_{dataset_key or 'default'}_{plan_signature}"

    # Only show review status after the user explicitly clicks the AI Review button.

    # This prevents an old empty result in Streamlit session state from showing a

    # warning automatically on later page reruns.

    review_clicked = approval_items and st.button(

        "🧠 Get AI Review of This Plan",

        key=f"ai_review_button_{dataset_key or 'default'}_{plan_signature}"

    )

    raw_review_key = f"{review_key}_raw"
    if review_clicked:
        st.session_state[review_attempted_key] = True
        st.session_state.pop(review_key, None)
        st.session_state.pop(f"{review_key}_error", None)
        st.session_state.pop(raw_review_key, None)
        try:
            with st.spinner("Qwen is reviewing the plan..."):
                review_map, raw_response = generate_plan_reasoning(plan, df)
            st.session_state[review_key] = review_map
            st.session_state[raw_review_key] = raw_response
        except Exception as error:
            st.session_state[review_key] = {}
            st.session_state[f"{review_key}_error"] = str(error)
            st.session_state[raw_review_key] = str(error)

    reasoning_map = st.session_state.get(review_key, {})
    if st.session_state.get(review_attempted_key, False):
        if reasoning_map and len(reasoning_map) == len(approval_items):
            st.success(f"AI review ready: {len(reasoning_map)} operation(s) reviewed.")
        else:
            st.warning(
                "AI review could not be validated. Your dataset and preprocessing plan are unchanged. "
                "Open the details below and share the displayed text with me."
            )
            with st.expander("AI Review Debug Details"):
                st.code(st.session_state.get(raw_review_key, "No response details available."))

    approval_index = 0

    approved_actions = set()

    # Each approval checkbox is displayed immediately below its own plan item.

    with st.form(f"preprocessing_form_{dataset_key or 'default'}_{plan_signature}"):

        for index, item in enumerate(plan):

            if item["type"] == "safe":

                st.success("AUTOMATIC — " + item["description"])

            elif item["type"] == "warning":

                st.warning("REVIEW RECOMMENDED — " + item["description"])

            else:

                st.info("APPROVAL REQUIRED — " + item["description"])

                reasoning = reasoning_map.get(approval_index)

                if reasoning:

                    st.caption("🧠 AI review: " + reasoning)

                approval_index += 1

                approved = st.checkbox(

                    "Approve this operation",

                    key=f"approve_{dataset_key or 'default'}_{plan_signature}_{index}"

                )

                if approved:

                    approved_actions.add((item["action"], item.get("column")))

                st.divider()

        run_agent = st.form_submit_button(

            "🚀 Run Preprocessing Agent",

            type="primary"

        )

    result_key = f"preprocessing_result_{dataset_key or 'default'}_{plan_signature}"

    log_key = f"preprocessing_log_{dataset_key or 'default'}_{plan_signature}"

    if run_agent:

        with st.spinner("Applying approved preprocessing operations..."):

            processed, log = execute_preprocessing(df, plan, approved_actions)

        st.session_state[result_key] = processed

        st.session_state[log_key] = log

    if result_key not in st.session_state:

        st.info("Review the plan, select the operations you approve, and run the agent.")

        return df

    processed = st.session_state[result_key]

    st.subheader("4. Preprocessing Execution Log")

    logs = st.session_state.get(log_key, [])

    if logs:

        for message in logs:

            st.write("✔️", message)

    else:

        st.info("No changes were necessary or approved.")

    st.subheader("5. Before-and-After Verification")

    show_comparison(df, processed)

    st.subheader("6. Processed Dataset Preview")

    st.dataframe(processed.head(10), width="stretch")

    st.download_button(

        "⬇️ Download Preprocessed Dataset",

        data=processed.to_csv(index=False).encode("utf-8-sig"),

        file_name="preprocessed_dataset.csv",

        mime="text/csv",

        key=f"download_preprocessed_{dataset_key or 'default'}_{plan_signature}"

    )

    return processed
