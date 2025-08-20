# --- numeric_input_form.py (updated with unique form key) ---
import streamlit as st
from datetime import datetime, date
from pinecone_search import embed_and_store
import json
from pathlib import Path
from uuid import uuid4
from pinecone_search import _append_history


CONFIG_PATH = Path("config/descriptors.json")
ENTRIES_PATH = Path("config/entries.json")

DEFAULT_DIMENSIONS = {
    "weight": ["kg"],
    "height": ["cm", "m"],
    "glucose": ["mg/dL", "mmol/L"],
    "cholesterol": ["mg/dL", "mmol/L"],
    "hdl": ["mg/dL", "mmol/L"],
    "ldl": ["mg/dL", "mmol/L"],
    "triglycerides": ["mg/dL", "mmol/L"],
    "blood pressure": ["mmHg"],
    "bp": ["mmHg"],
}

def load_descriptor_config():
    user_cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            user_cfg = json.load(f)
    cfg = {**DEFAULT_DIMENSIONS}
    for k, v in user_cfg.items():
        base = {u for u in cfg.get(k, [])}
        merged = list({*base, *(u for u in v)})
        cfg[k] = merged
    return cfg

def save_descriptor_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def save_entries(entries):
    ENTRIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTRIES_PATH, "w") as f:
        json.dump(entries, f, indent=2)

def load_saved_entries():
    if ENTRIES_PATH.exists():
        with open(ENTRIES_PATH, "r") as f:
            return json.load(f)
    return []

def _entry_to_text(entry):
    return (
        f"On {entry['date']}, the user's {entry['descriptor']} was "
        f"{entry['value']} {entry['dimension']}."
    ).strip()

def _reset_value_fields():
    for k in ("value_num", "value_text", "bp_sys", "bp_dia"):
        if k in st.session_state:
            del st.session_state[k]

def render_numeric_input_form():
    st.markdown("Enter structured numeric health records. Each entry will be embedded and stored in Pinecone.")

    descriptor_config = load_descriptor_config()

    if "entries" not in st.session_state:
        st.session_state.entries = load_saved_entries()

    colA, colB = st.columns([2, 2])

    with colA:
        descriptors = sorted(list(descriptor_config.keys())) + ["Other"]
        picked_descriptor = st.selectbox(
            "Descriptor",
            options=descriptors,
            index=0,
            key="ui_descriptor",
            on_change=_reset_value_fields,
        )
        if picked_descriptor == "Other":
            custom_desc = st.text_input("Enter custom descriptor", key="ui_custom_desc")
            descriptor = (custom_desc or "").strip()
            if descriptor and descriptor not in descriptor_config:
                descriptor_config[descriptor] = []
                save_descriptor_config(descriptor_config)
        else:
            descriptor = picked_descriptor

    with colB:
        allowed_dims = descriptor_config.get(descriptor, [])
        if allowed_dims:
            picked_dim = st.selectbox(
                "Dimension",
                options=allowed_dims + ["Other"],
                index=0,
                key="ui_dimension",
                on_change=_reset_value_fields,
            )
        else:
            picked_dim = "Other"

        if picked_dim == "Other":
            custom_dim = st.text_input("Enter custom dimension", key="ui_custom_dim")
            dimension = (custom_dim or "").strip()
            if dimension and dimension not in descriptor_config.get(descriptor, []):
                descriptor_config.setdefault(descriptor, []).append(dimension)
                save_descriptor_config(descriptor_config)
        else:
            dimension = picked_dim

    bp_alias = descriptor.lower() in {"blood pressure", "bp"} if descriptor else False
    if bp_alias and not dimension:
        dimension = "mmHg"

    # UNIQUE FORM KEY (prevents duplicate-key error if the form appears twice on the page)
    with st.form(key="pa_numeric_entry_form"):
        if bp_alias:
            sys = st.number_input("Systolic", min_value=50, max_value=250, step=1, key="bp_sys")
            dia = st.number_input("Diastolic", min_value=30, max_value=150, step=1, key="bp_dia")
            value = f"{int(sys)}/{int(dia)}"
        else:
            value_input_type = st.radio("Input Type", ["Numeric", "Textual (e.g. 115/77)"], key="value_kind")
            if value_input_type == "Numeric":
                value = st.number_input("Value", min_value=0.0, step=0.1, key="value_num")
            else:
                value = st.text_input("Enter value as text", key="value_text")

        today = date.today()
        d = st.date_input("Date", value=today)
        # give time a deterministic default so it doesn't vary across reruns
        t = st.time_input("Time", value=datetime.now().time())
        date_str = datetime.combine(d, t).strftime("%Y-%m-%d %H:%M")

        submitted = st.form_submit_button("Add Entry")
        if submitted:
            if not descriptor:
                st.error("Please provide a descriptor.")
            elif not dimension:
                st.error("Please provide a dimension.")
            elif (not bp_alias) and isinstance(value, float) and value <= 0:
                st.error("Numeric value must be greater than 0.")
            else:
                entry = {
                    "id": str(uuid4()),
                    "date": date_str,
                    "descriptor": descriptor,
                    "dimension": dimension,
                    "value": value,
                }
                if bp_alias:
                    entry.update({"systolic": int(st.session_state.get("bp_sys", 0)),
                                  "diastolic": int(st.session_state.get("bp_dia", 0))})
                st.session_state.entries.append(entry)
                save_entries(st.session_state.entries)
                st.success(f"Added entry for {descriptor} at {date_str}")

    if st.session_state.entries:
        st.markdown("### Entries Ready for Upload")

        labels = [
            f"{i+1}: {e['descriptor']} {e['value']} {e['dimension']} ({e['date']})"
            for i, e in enumerate(st.session_state.entries)
        ]
        delete_index = st.radio("Select entry to delete:", options=labels, index=0, key="del_select")
        if st.button("Delete Selected Entry"):
            idx = int(delete_index.split(":")[0]) - 1
            deleted = st.session_state.entries.pop(idx)
            save_entries(st.session_state.entries)
            st.warning(f"Deleted: {deleted['descriptor']} {deleted['value']} {deleted['dimension']}")

        for i, entry in enumerate(st.session_state.entries):
            st.markdown(f"**{i+1}.** `{entry['date']}` — **{entry['descriptor']}**: {entry['value']} {entry['dimension']}")

        if st.button("Upload All to Pinecone"):
            for entry in st.session_state.entries:
                summary = _entry_to_text(entry)
                embed_and_store(summary, source=f"form-entry-{entry['id']}", type_="health_metric")
                _append_history(st.session_state.entries)
            st.success("✅ All entries uploaded to Pinecone.")
            st.session_state.entries.clear()
            save_entries([])
            st.rerun()
    else:
        st.info("No entries added yet.")
