# pages/2_Faculty.py
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from db import col
from utils.auth import require_role, current_user


# ──────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────
def _user_header(u: dict | None):
    if not u:
        return
    st.markdown(
        f"""
        <div style="margin-top:-8px;margin-bottom:10px;padding:10px 12px;
             border:1px solid rgba(0,0,0,.06); border-radius:10px;
             background:linear-gradient(180deg,#0b1220 0%,#0e1729 100%);
             color:#e6edff;">
          <div style="font-size:14px;opacity:.85">Signed in as</div>
          <div style="font-size:16px;font-weight:700;">{u.get('name','')}</div>
          <div style="font-size:13px;opacity:.75;">{u.get('email','')}</div>
          <div style="margin-top:6px;font-size:12px;display:inline-block;
               padding:2px 6px;border:1px solid rgba(255,255,255,.12);
               border-radius:6px;letter-spacing:.4px;">
            {(u.get('role','') or '').upper()}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────
def _term_label(sy: str | None, sem: int | None) -> str:
    if not sy:
        return "—"
    try:
        s = int(sem or 0)
    except Exception:
        s = 0
    return f"{sy} S{s}" if s else sy


def _term_sort_key(label: str) -> tuple[int, int]:
    """Sort '2023-2024 S1', '2023-2024 S2', '2023-2024 S3' correctly."""
    if not isinstance(label, str) or " S" not in label:
        return (0, 0)
    sy, s = label.split(" S", 1)
    try:
        start_year = int(sy.split("-")[0])
    except Exception:
        start_year = 0
    try:
        sem = int(s)
    except Exception:
        sem = 0
    return (start_year, sem)


def _to_num_grade(x) -> float | None:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _nemail(e: str | None) -> str:
    return (e or "").strip().lower()


@st.cache_data(show_spinner=False)
def load_term_catalog() -> list[tuple[str, str, int]]:
    """Read all school-years/semesters from `semesters`."""
    rows = list(col("semesters").find({}, {"school_year": 1, "semester": 1}))
    seen = set()
    out: list[tuple[str, str, int]] = []
    for r in rows:
        sy = r.get("school_year")
        sem = r.get("semester")
        if sy and sem is not None:
            label = _term_label(sy, sem)
            key = (label, sy, int(sem))
            if key not in seen:
                seen.add(key)
                out.append(key)
    out.sort(key=lambda t: _term_sort_key(t[0]))
    return out


def list_teacher_emails() -> List[Tuple[str, str]]:
    """Returns [(name, email), ...] from enrollments.teacher.*."""
    pipe = [
        {"$match": {"teacher.email": {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": "$teacher.email", "name": {"$first": "$teacher.name"}}},
        {"$sort": {"_id": 1}},
    ]
    out: list[tuple[str, str]] = []
    for r in col("enrollments").aggregate(pipe):
        em = _nemail(r.get("_id"))
        nm = r.get("name") or ""
        if em:
            out.append((nm or em, em))
    return out


@st.cache_data(show_spinner=False)
def load_enrollments_df(teacher_email: Optional[str]) -> pd.DataFrame:
    """
    Load enrollments as a flattened DataFrame.
    If `teacher_email` is provided, filter to that teacher; otherwise return all rows.
    """
    q = {}
    if teacher_email:
        q = {"teacher.email": teacher_email}

    proj = {
        "term.grade": 1,
        "term.remark": 1,
        "term.school_year": 1,
        "term.semester": 1,
        "student.name": 1,
        "student.student_no": 1,
        "subject.code": 1,
        "subject.title": 1,
        "teacher.email": 1,
        "teacher.name": 1,
        "program.program_code": 1,
        "term.section": 1,
    }
    rows = list(col("enrollments").find(q, proj))

    def flatten(e: dict) -> dict:
        term = e.get("term") or {}
        stu = e.get("student") or {}
        sub = e.get("subject") or {}
        tch = e.get("teacher") or {}
        prog = e.get("program") or {}
        return {
            "student_no": stu.get("student_no"),
            "student_name": stu.get("name"),
            "subject_code": sub.get("code"),
            "subject_title": sub.get("title"),
            "grade": _to_num_grade(term.get("grade")),
            "remark": term.get("remark"),
            "term_label": _term_label(term.get("school_year"), term.get("semester")),
            "teacher_email": _nemail(tch.get("email")),
            "teacher_name": tch.get("name"),
            "program_code": prog.get("program_code"),
            "section": term.get("section"),
        }

    df = pd.DataFrame([flatten(r) for r in rows]) if rows else pd.DataFrame(
        columns=[
            "student_no", "student_name", "subject_code", "subject_title",
            "grade", "remark", "term_label", "teacher_email", "teacher_name",
            "program_code", "section",
        ]
    )
    return df


# ──────────────────────────────────────────
# Page
# ──────────────────────────────────────────
def main():
    # Auth — faculty, registrar, or admin
    user = require_role("faculty", "teacher", "registrar", "admin")

    st.title("🏫 Faculty Dashboard")
    st.caption("Teacher scope applies automatically for faculty. Registrars/Admins can filter by teacher.")
    _user_header(user)

    teacher_email: Optional[str] = None
    teachers = list_teacher_emails()

    # Scope by role
    role = (user.get("role") or "").lower()
    if role in ("faculty", "teacher"):
        teacher_email = _nemail(user.get("email"))
    else:
        # Registrar/Admin can pick a teacher if present, else ALL
        if teachers:
            label_options = [f"{nm} ({em})" for nm, em in teachers]
            picked = st.selectbox("Filter by teacher (Registrar/Admin)", options=label_options)
            idx = label_options.index(picked)
            teacher_email = teachers[idx][1]
        else:
            st.info("No teacher emails found in enrollments; showing **all enrollments** instead.")

    # Base data (already scoped by teacher)
    df = load_enrollments_df(teacher_email)

    # Global term catalog for the picker (always show all terms)
    term_catalog = load_term_catalog()
    all_term_labels = [lbl for (lbl, _, _) in term_catalog]

    # Quick filters (terms/subjects/programs)
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_terms = st.multiselect("Term(s)", options=all_term_labels, default=all_term_labels)
    with c2:
        subjects = sorted(df["subject_code"].dropna().unique())
        sel_subjects = st.multiselect("Subject(s)", options=subjects, default=subjects)
    with c3:
        progs = sorted(df["program_code"].dropna().unique())
        sel_progs = st.multiselect("Program(s)", options=progs, default=progs)

    if sel_terms:
        df = df[df["term_label"].isin(sel_terms)]
    if sel_subjects:
        df = df[df["subject_code"].isin(sel_subjects)]
    if sel_progs:
        df = df[df["program_code"].isin(sel_progs)]

    # ─────────────────────────────────────
    # 1) Class Grade Distribution
    # ─────────────────────────────────────
    st.subheader("1) Class Grade Distribution")

    # Faculty Name — show selected teacher if registrar/admin
    faculty_display_name = user.get("name") or ""
    if role in ("registrar", "admin") and teacher_email:
        match = next((nm for nm, em in teachers if em == teacher_email), None)
        if match:
            faculty_display_name = match

    hc1, hc2 = st.columns([1, 1.4])
    with hc1:
        st.text_input("Faculty Name:", value=faculty_display_name, key="faculty_name_display")
    with hc2:
        st.text_input(
            "Semester and School Year:",
            value=", ".join(sel_terms) if sel_terms else "",
            key="term_sy_display",
        )

    graded = df.dropna(subset=["grade"])
    if graded.empty:
        st.info("No graded entries found for this scope.")
    else:
        # Build subject-by-bin distribution table (percentages)
        bins = [0, 75, 80, 85, 90, 95, 100]
        labels = ["Below 75 (%)", "75–79 (%)", "80–84 (%)", "85–89 (%)", "90–94 (%)", "95–100 (%)"]

        tmp = graded.copy()
        tmp["Course Code"] = tmp["subject_code"].fillna("")
        tmp["Course Name"] = tmp["subject_title"].fillna("")
        tmp["bin"] = pd.cut(tmp["grade"], bins=bins, labels=labels, right=True, include_lowest=True)

        counts = (
            tmp.groupby(["Course Code", "Course Name", "bin"], observed=False)  # avoid FutureWarning
               .size()
               .unstack(fill_value=0)
        )
        for col in labels:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[labels]
        totals = counts.sum(axis=1)
        pct = (counts.div(totals.replace(0, 1), axis=0) * 100).round(0).astype("Int64").astype(str) + "%"
        pct["Total"] = totals.values
        pct = pct.reset_index()

        st.dataframe(
            pct[["Course Code", "Course Name"] + labels + ["Total"]],
            width='stretch',
            hide_index=True,
        )

        st.markdown("**Followed by: histogram**")
        hist_bins = list(range(60, 101, 5))
        hist_counts = pd.cut(graded["grade"], bins=hist_bins, right=True, include_lowest=True)\
                        .value_counts().sort_index()
        chart_df = pd.DataFrame({"Range": hist_counts.index.astype(str), "Count": hist_counts.values})\
                     .set_index("Range")
        st.bar_chart(chart_df)

    # ─────────────────────────────────────
    # 2) Student Progress Tracker (table → line chart)
    # ─────────────────────────────────────
    st.subheader("2) Student Progress Tracker")
    st.caption("Shows longitudinal performance for individual students. Filtered by Subject or Course or YearLevel.")

    # Use the page-scope dataset (already limited by teacher/term/section) as the base.
    try:
        base_df = df_scope.copy()  # prefer page-level filtered dataset if available
    except NameError:
        try:
            base_df = df.copy()    # fall back to raw df in this file
        except NameError:
            base_df = pd.DataFrame()  # ultimate fallback to avoid NameError during static analysis / runtime

    if base_df.empty:
        st.info("No data available for student progress in the current page filters.")
    else:
        # Normalize common fields
        if "subject_code" in base_df.columns:
            base_df["subject_code"] = base_df["subject_code"].astype(str).str.strip().str.upper()
        if "program_code" in base_df.columns:
            base_df["program_code"] = base_df["program_code"].astype(str).str.strip()

        # Local, section-specific filters (optional). If left empty, we inherit the page's top filters.
        lf1, lf2, lf3 = st.columns(3)
        with lf1:
            sec2_subjects = st.multiselect(
                "Filter (Subject / Course Code)",
                options=sorted(base_df["subject_code"].dropna().unique().tolist())
                        if "subject_code" in base_df.columns else [],
                help="Optional; leave blank to keep the current page filters."
            )
        with lf2:
            sec2_programs = st.multiselect(
                "Filter (Program / Course)",
                options=sorted(base_df["program_code"].dropna().unique().tolist())
                        if "program_code" in base_df.columns else [],
                help="Optional; leave blank to keep the current page filters."
            )
        ycol = "year_level" if "year_level" in base_df.columns else ("yearlevel" if "yearlevel" in base_df.columns else None)
        with lf3:
            sec2_yearlevels = st.multiselect(
                "Filter (Year Level)",
                options=sorted(base_df[ycol].dropna().unique().tolist()) if ycol else [],
                help="Optional; ignored if the dataset has no year level field."
            )

        # Apply OPTIONAL local filters on top of the page filters.
        g = base_df.copy()
        before = len(g)

        if sec2_subjects and "subject_code" in g.columns:
            sel_upper = [str(s).strip().upper() for s in sec2_subjects]
            g = g[g["subject_code"].isin(sel_upper)]
        after_subj = len(g)

        if sec2_programs and "program_code" in g.columns:
            g = g[g["program_code"].isin(sec2_programs)]
        after_prog = len(g)

        if sec2_yearlevels and ycol and ycol in g.columns:
            g = g[g[ycol].isin(sec2_yearlevels)]
        after_year = len(g)

        # Diagnostics
        st.caption(
            f"Rows after page filters: **{before}** · "
            f"after Subject filter: **{after_subj}** · "
            f"after Program filter: **{after_prog}** · "
            f"after Year Level filter: **{after_year}**"
        )

        g["grade"] = pd.to_numeric(g.get("grade"), errors="coerce")
        g = g.dropna(subset=["grade"])
        if g.empty:
            st.info("No graded rows match the chosen filters. Try removing one of the local filters.")
        else:
            # Derive GPA 0–4 from numeric grade
            g["gpa"] = (g["grade"] / 100.0 * 4.0).clip(0, 4).round(2)

            # Prefer page-selected terms if available; else all (latest 3)
            terms_from_page = sel_terms if "sel_terms" in globals() or "sel_terms" in locals() else None
            terms_order = terms_from_page if terms_from_page else \
                sorted(g["term_label"].dropna().unique().tolist(), key=_term_sort_key)
            if len(terms_order) > 3:
                terms_order = terms_order[-3:]

            # Wide table: rows = students; columns = terms; values = mean GPA
            pivot = (
                g[g["term_label"].isin(terms_order)]
                .groupby(["student_no", "student_name", "term_label"], observed=False)["gpa"]
                .mean()
                .reset_index()
                .pivot_table(index=["student_no", "student_name"], columns="term_label", values="gpa", aggfunc="mean")
                .reindex(columns=terms_order)
            )

            # If empty (e.g., no grades in those terms), fall back to all terms
            if pivot.empty:
                all_terms_sorted = sorted(g["term_label"].dropna().unique().tolist(), key=_term_sort_key)
                pivot = (
                    g.groupby(["student_no", "student_name", "term_label"], observed=False)["gpa"]
                    .mean().reset_index()
                    .pivot_table(index=["student_no", "student_name"], columns="term_label", values="gpa", aggfunc="mean")
                    .reindex(columns=all_terms_sorted)
                )
                terms_order = all_terms_sorted

            # Trend descriptor
            def trend_text(row):
                vals = [v for v in row.tolist() if pd.notnull(v)]
                if len(vals) < 2:
                    return "—"
                delta = vals[-1] - vals[0]
                if delta >= 0.10:
                    return "↑ Improving"
                if delta <= -0.10:
                    return "↓ Needs Attention"
                return "→ Stable High"

            trend = pivot.apply(trend_text, axis=1)
            pivot_display = pivot.copy().round(2)
            pivot_display.insert(0, "Student ID", [i[0] for i in pivot_display.index])
            pivot_display.insert(1, "Name", [i[1] for i in pivot_display.index])
            pivot_display["Overall Trend"] = trend.values
            pivot_display = pivot_display.reset_index(drop=True)

            st.dataframe(
                pivot_display[["Student ID", "Name"] + terms_order + ["Overall Trend"]],
                width='stretch',
                hide_index=True,
            )

            st.markdown("**Followed by: line graph / scatter chart.**")

            # Multi-series line (one series per student)
            long_df = (
                pivot.reset_index()
                    .melt(id_vars=["student_no", "student_name"], value_vars=terms_order,
                        var_name="Term", value_name="GPA")
                    .dropna(subset=["GPA"])
                    .sort_values(["student_no", "Term"])
            )
            if not long_df.empty:
                chart_wide = (
                    long_df.pivot_table(index="Term", columns="student_name", values="GPA", aggfunc="mean")
                        .reindex(terms_order)
                )
                st.line_chart(chart_wide)
            else:
                st.caption("No GPA points to chart for the current filters.")


    # ─────────────────────────────────────
    # 3. Subject Difficulty Heatmap
    # ─────────────────────────────────────
    st.subheader("3. Subject Difficulty Heatmap")
    st.markdown(
        "- Visualizes subjects with high failure or dropouts\n"
        "- Subjects handled by the faculty"
    )

    # Default passing threshold if not defined elsewhere
    try:
        PASSING_GRADE  # noqa: F823
    except NameError:
        PASSING_GRADE = 75

    # Base: use page-scope (already filtered by teacher/term/section)
    try:
        base = df_scope.copy()
    except NameError:
        base = df.copy()

    if base.empty:
        st.info("No enrollments available in the current page filters.")
    else:
        # Normalize columns defensively
        if "subject_code" in base.columns:
            base["subject_code"] = base["subject_code"].astype(str).str.strip().str.upper()
        else:
            st.info("No subject codes found in the current data.")
            st.stop()

        if "subject_title" in base.columns:
            base["subject_title"] = base["subject_title"].astype(str).str.strip()
        else:
            base["subject_title"] = ""  # keep pipeline working

        # Grades
        grades = pd.to_numeric(base["grade"], errors="coerce") if "grade" in base.columns else pd.Series(dtype=float)
        is_graded = grades.notna()
        is_failed = is_graded & (grades < PASSING_GRADE)

        # Dropout heuristic
        if "remark" in base.columns:
            remarks = base["remark"].astype(str).str.lower().fillna("")
            dropped_by_remark = remarks.str.contains("drop") | remarks.str.contains("withdr")
        else:
            dropped_by_remark = pd.Series(False, index=base.index)

        # Ungraded counts as potential dropout; refine to your policy if needed
        is_dropout = (~is_graded) | dropped_by_remark

        work = base.assign(
            _graded=is_graded,
            _failed=is_failed,
            _drop=is_dropout,
        )

        # Aggregate per subject
        grp = (
            work.groupby(["subject_code", "subject_title"], dropna=False)
                .agg(
                    total=("subject_code", "size"),
                    graded_cnt=("_graded", "sum"),
                    failed_cnt=("_failed", "sum"),
                    dropout_cnt=("_drop", "sum"),
                )
                .reset_index()
        )

        # Rates (%)
        grp["Fail Rate (%)"] = np.where(
            grp["graded_cnt"] > 0,
            (grp["failed_cnt"] / grp["graded_cnt"] * 100).round(0),
            0,
        ).astype(int)

        grp["Dropout Rate (%)"] = np.where(
            grp["total"] > 0,
            (grp["dropout_cnt"] / grp["total"] * 100).round(0),
            0,
        ).astype(int)

        # Difficulty rule (tweak thresholds as needed)
        def _difficulty(row):
            fail = row["Fail Rate (%)"]
            drp = row["Dropout Rate (%)"]
            if fail >= 15 or drp >= 5:
                return "High"
            if fail >= 8 or drp >= 3:
                return "Medium"
            return "Low"

        grp["Difficulty Level"] = grp.apply(_difficulty, axis=1)

        # Final display
        show = grp.rename(columns={
            "subject_code": "Course Code",
            "subject_title": "Course Name",
        })[["Course Code", "Course Name", "Fail Rate (%)", "Dropout Rate (%)", "Difficulty Level"]]

        # Sort: High → Medium → Low, then Fail Rate desc
        lvl_order = pd.CategoricalDtype(categories=["High", "Medium", "Low"], ordered=True)
        show["Difficulty Level"] = show["Difficulty Level"].astype(lvl_order)
        show = show.sort_values(["Difficulty Level", "Fail Rate (%)"], ascending=[True, False]).reset_index(drop=True)

        st.dataframe(show, width='stretch')


    # ─────────────────────────────────────
    # 4. Intervention Candidates List
    # ─────────────────────────────────────
    st.subheader("4. Intervention Candidates List")
    st.markdown(
        "- Lists students at academic risk based on current semester data (e.g. low grades, missing grades)"
    )

    # Default passing threshold if not defined globally
    try:
        PASSING_GRADE
    except NameError:
        PASSING_GRADE = 75

    # Use page-scope dataset (already filtered by teacher / term / section)
    try:
        base4 = df_scope.copy()
    except NameError:
        base4 = df.copy()

    if base4.empty:
        st.info("No enrollments available in the current page filters.")
    else:
        # Normalize fields
        if "subject_code" in base4.columns:
            base4["subject_code"] = base4["subject_code"].astype(str).str.strip().str.upper()
        if "subject_title" in base4.columns:
            base4["subject_title"] = base4["subject_title"].astype(str).str.strip()

        # Determine the current term scope
        try:
            terms_selected = sel_terms if sel_terms else None
        except NameError:
            terms_selected = None

        if terms_selected:
            current_scope = base4[base4["term_label"].isin(terms_selected)].copy()
        else:
            if base4["term_label"].dropna().empty:
                current_scope = base4.copy()
            else:
                latest_term = sorted(base4["term_label"].dropna().unique().tolist(), key=_term_sort_key)[-1]
                current_scope = base4[base4["term_label"] == latest_term].copy()

        if current_scope.empty:
            st.info("No enrollments found for the selected/most recent term.")
        else:
            # Grade + remark processing
            grades = pd.to_numeric(current_scope.get("grade"), errors="coerce")
            remarks = current_scope.get("remark")
            if remarks is None:
                remarks = pd.Series([""] * len(current_scope), index=current_scope.index)
            remarks = remarks.astype(str).str.lower()

            # Missing grade = INC in remarks or NaN grade
            missing_mask = grades.isna() | remarks.str.contains(r"\binc|\bincomplete")

            # Failed = below passing grade (numeric)
            failed_mask = grades.notna() & (grades < PASSING_GRADE)

            risk_rows = current_scope[missing_mask | failed_mask].copy()
            if risk_rows.empty:
                st.success("No intervention candidates in the current term scope.")
            else:
                def risk_flag(row):
                    g = pd.to_numeric(row.get("grade"), errors="coerce")
                    r = str(row.get("remark") or "").lower()
                    if pd.isna(g) or ("inc" in r or "incomplete" in r):
                        return "Missing Grade"
                    if g < PASSING_GRADE:
                        return f"At Risk (<{PASSING_GRADE})"
                    return "—"

                risk_rows["Risk Flag"] = risk_rows.apply(risk_flag, axis=1)

                # Display table
                show = risk_rows.rename(columns={
                    "student_no": "Student ID",
                    "student_name": "Name",
                    "subject_code": "Course Code",
                    "subject_title": "Course Name",
                    "grade": "Current Grade",
                })[
                    ["Student ID", "Name", "Course Code", "Course Name", "Current Grade", "Risk Flag"]
                ].sort_values(["Risk Flag", "Name", "Course Code"]).reset_index(drop=True)

                # Faculty name line
                fac_name = ""
                if "teacher_name" in current_scope.columns and current_scope["teacher_name"].notna().any():
                    fac_name = current_scope["teacher_name"].dropna().iloc[0]
                elif "teacher_email" in current_scope.columns and current_scope["teacher_email"].notna().any():
                    fac_name = current_scope["teacher_email"].dropna().iloc[0]

                st.markdown(f"**Faculty Name:** {fac_name if fac_name else '—'}")
                st.dataframe(show, width='stretch')

    # ─────────────────────────────────────
    # 5. Grade Submission Status
    # ─────────────────────────────────────
    st.subheader("5. Grade Submission Status")
    st.markdown(
        "- Tracks the status of grade submissions by faculty for each class. (e.g., complete grades, with blank grades)"
    )

    # Use page-scope dataset (already filtered by teacher / term / section)
    try:
        base5 = df_scope.copy()
    except NameError:
        base5 = df.copy()

    if base5.empty:
        st.info("No enrollments available in the current page filters.")
    else:
        # Normalize fields we rely on
        base5["subject_code"] = base5.get("subject_code", "").astype(str).str.strip().str.upper()
        base5["subject_title"] = base5.get("subject_title", "").astype(str).str.strip()

        # Faculty line
        fac_name = ""
        if "teacher_name" in base5.columns and base5["teacher_name"].notna().any():
            fac_name = base5["teacher_name"].dropna().iloc[0]
        elif "teacher_email" in base5.columns and base5["teacher_email"].notna().any():
            fac_name = base5["teacher_email"].dropna().iloc[0]
        st.markdown(f"**Grade Submission Status — {fac_name if fac_name else '—'}**")

        # Per-subject submission stats
        grades = pd.to_numeric(base5.get("grade"), errors="coerce")
        base5 = base5.assign(_submitted=grades.notna())

        status = (
            base5.groupby(["subject_code", "subject_title"], dropna=False)
                .agg(
                    Submitted_Grades=("_submitted", "sum"),
                    Total_Students=("subject_code", "size"),
                )
                .reset_index()
        )
        status["Submission Rate"] = np.where(
            status["Total_Students"] > 0,
            (status["Submitted_Grades"] / status["Total_Students"] * 100).round(0).astype(int),
            0
        )

        show = status.rename(columns={
            "subject_code": "Course Code",
            "subject_title": "Course Title",
            "Submitted_Grades": "Submitted Grades",
            "Total_Students": "Total Students",
        })[["Course Code", "Course Title", "Submitted Grades", "Total Students", "Submission Rate"]]

        # Sort by lowest submission rate first (to highlight gaps)
        show = show.sort_values(["Submission Rate", "Course Code"]).reset_index(drop=True)

        # Render (add % sign for readability)
        show_display = show.copy()
        show_display["Submission Rate"] = show_display["Submission Rate"].astype(int).astype(str) + "%"

        st.dataframe(show_display, width='stretch')


    # ─────────────────────────────────────
    # 6) Custom Query Builder
    # ─────────────────────────────────────
    st.subheader("6) Custom Query Builder")
    st.caption("Allows users to build filtered queries (e.g., “Show all students with < 75 in CS101”).")

    df_safe = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if df_safe.empty and 'teacher_email' not in locals():
        st.info("No enrollments to query.")
    else:
        # Use FULL (unfiltered) scope for option lists so all terms/subjects appear
        try:
            df_all = load_enrollments_df(teacher_email)
        except Exception:
            df_all = df_safe.copy()

        subj_opts = sorted(df_all.get("subject_code", pd.Series(dtype=str)).dropna().unique().tolist())
        term_opts = sorted(df_all.get("term_label", pd.Series(dtype=str)).dropna().unique().tolist(), key=_term_sort_key)
        prog_opts = sorted(df_all.get("program_code", pd.Series(dtype=str)).dropna().unique().tolist())

        cqb1, cqb2, cqb3, cqb4 = st.columns([1, 1, 1, 1])
        with cqb1:
            cq_subject = st.selectbox("Course Code", options=["(any)"] + subj_opts, key="cq_subject")
        with cqb2:
            cq_op = st.selectbox("Grade Operator", ["<", "<=", "==", ">=", ">", "between"], key="cq_op")
        with cqb3:
            cq_val1 = st.number_input("Value", min_value=0.0, max_value=100.0, value=75.0, step=1.0, key="cq_val1")
        with cqb4:
            cq_val2 = None
            if cq_op == "between":
                cq_val2 = st.number_input("and", min_value=0.0, max_value=100.0, value=85.0, step=1.0, key="cq_val2")

        cqb5, cqb6 = st.columns(2)
        with cqb5:
            cq_terms = st.multiselect("Term(s) (optional)", options=term_opts, key="cq_terms")
        with cqb6:
            cq_progs = st.multiselect("Program(s) (optional)", options=prog_opts, key="cq_progs")

        run_query = st.button("Run Query", type="primary", key="cq_run")

        # Example line
        if cq_subject != "(any)":
            if cq_op == "between" and cq_val2 is not None:
                example_txt = f"Show all students with {cq_val1:.0f} ≤ grade ≤ {cq_val2:.0f} in {cq_subject}"
            else:
                example_txt = f"Show all students with {cq_op} {cq_val1:.0f} in {cq_subject}"
        else:
            if cq_op == "between" and cq_val2 is not None:
                example_txt = f"Show all students with {cq_val1:.0f} ≤ grade ≤ {cq_val2:.0f}"
            else:
                example_txt = f"Show all students with grade {cq_op} {cq_val1:.0f}"
        st.markdown(f"*Query Example:* _{example_txt}_")

        if run_query:
            q = df_safe.copy()
            if "grade" not in q.columns:
                st.info("No grade column to query.")
            else:
                q = q.dropna(subset=["grade"])

                if cq_subject != "(any)" and "subject_code" in q.columns:
                    q = q[q["subject_code"] == cq_subject]
                if cq_terms and "term_label" in q.columns:
                    q = q[q["term_label"].isin(cq_terms)]
                if cq_progs and "program_code" in q.columns:
                    q = q[q["program_code"].isin(cq_progs)]

                if cq_op == "<":
                    q = q[q["grade"] < cq_val1]
                elif cq_op == "<=":
                    q = q[q["grade"] <= cq_val1]
                elif cq_op == "==":
                    q = q[q["grade"] == cq_val1]
                elif cq_op == ">=":
                    q = q[q["grade"] >= cq_val1]
                elif cq_op == ">":
                    q = q[q["grade"] > cq_val1]
                elif cq_op == "between" and cq_val2 is not None:
                    lo, hi = (cq_val1, cq_val2) if cq_val1 <= cq_val2 else (cq_val2, cq_val1)
                    q = q[(q["grade"] >= lo) & (q["grade"] <= hi)]

                cols_out = ["student_no", "student_name", "subject_code", "subject_title", "grade"]
                cols_out = [c for c in cols_out if c in q.columns]
                res = (
                    q[cols_out]
                    .rename(columns={
                        "student_no": "Student ID",
                        "student_name": "Name",
                        "subject_code": "Course Code",
                        "subject_title": "Course Name",
                        "grade": "Grade",
                    })
                    .sort_values(["Course Code", "Name"], na_position="last")
                )

                st.dataframe(res, width='stretch', hide_index=True)
                st.caption(f"{len(res)} result(s)")

    # ─────────────────────────────────────
    # 7) Students Grade Analytics (Per Teacher)
    # ─────────────────────────────────────
    st.subheader("7) Students Grade Analytics (Per Teacher)")

    _all_teachers = teachers or list_teacher_emails()
    if not _all_teachers:
        st.info("No teachers found.")
    else:
        labels = [f"{nm}" for nm, _em in _all_teachers]
        emails = [em for _nm, em in _all_teachers]
        default_idx = 0
        if teacher_email in emails:
            default_idx = emails.index(teacher_email)

        sel_teacher_name = st.selectbox("Select Teacher", options=labels, index=default_idx, key="gpt_tch_sel")
        sel_teacher_email = emails[labels.index(sel_teacher_name)]

        df_t = load_enrollments_df(sel_teacher_email)

        if df_t.empty:
            st.info("No enrollments for the selected teacher.")
        else:
            subj_labels = (
                df_t.assign(_label=lambda d: d["subject_code"].fillna("").astype(str) + " - " +
                                      d["subject_title"].fillna("").astype(str))
                   .drop_duplicates(subset=["subject_code"])
                   .sort_values("subject_code")
            )
            subj_options = subj_labels["_label"].tolist()
            label_to_code = dict(zip(subj_labels["_label"], subj_labels["subject_code"]))

            sel_subj_label = st.selectbox("Select Subject:", options=subj_options, key="gpt_subj_sel")
            sel_subj_code = label_to_code.get(sel_subj_label)

            df_s = df_t[df_t["subject_code"] == sel_subj_code].copy()
            df_s = df_s.dropna(subset=["grade"])
            st.markdown(f"**Grades Summary of Faculty: {sel_teacher_name}**")

            if df_s.empty:
                st.info("No graded entries for this subject.")
            else:
                g = df_s["grade"].astype(float)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{g.mean():.2f}")
                c2.metric("Median", f"{g.median():.2f}")
                c3.metric("Highest", f"{g.max():.2f}")
                c4.metric("Lowest", f"{g.min():.2f}")

                st.markdown(f"**Grade Distribution – {sel_subj_label}**")
                bins = list(range(60, 101, 5))
                hist = pd.cut(g, bins=bins, right=True, include_lowest=True).value_counts().sort_index()
                hist_df = pd.DataFrame({"Range": hist.index.astype(str), "Frequency": hist.values}).set_index("Range")
                st.bar_chart(hist_df)

                st.markdown("**Pass vs Fail**")
                pass_threshold = 75.0
                pass_cnt = int((g >= pass_threshold).sum())
                fail_cnt = int((g < pass_threshold).sum())
                pvf_df = pd.DataFrame({"Outcome": ["Pass", "Fail"], "Count": [pass_cnt, fail_cnt]}).set_index("Outcome")
                st.bar_chart(pvf_df)

                table_cols = {
                    "student_no": "Student ID",
                    "student_name": "Student Name",
                    "program_code": "Course",
                    "grade": "Grade",
                }
                if "year_level" in df_s.columns:
                    df_s["_YearLevel"] = df_s["year_level"]
                elif "yearlevel" in df_s.columns:
                    df_s["_YearLevel"] = df_s["yearlevel"]
                else:
                    df_s["_YearLevel"] = ""

                out = (
                    df_s.assign(_pass_fail=lambda d: np.where(d["grade"].astype(float) >= pass_threshold, "Pass", "Fail"))
                        .rename(columns=table_cols)
                )
                cols_to_show = ["Student ID", "Student Name", "Course", "_YearLevel", "Grade", "_pass_fail"]
                cols_to_show = [c for c in cols_to_show if c in out.columns]
                out = out[cols_to_show].rename(columns={"_YearLevel": "YearLevel", "_pass_fail": "Pass/Fail"})
                out = out.sort_values(["Pass/Fail", "Student Name"], ascending=[True, True], na_position="last")

                st.markdown("**Student Grades**")
                st.dataframe(out, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
