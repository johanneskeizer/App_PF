#(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
#diff --git a/modules/config_loader.py b/modules/config_loader.py
#index e00b1cf9a331dda7f2cc24f4435d5b031cc71348..426f79e426772e6e3ea4a2352905b22a235f0afe 100644
#--- a/modules/config_loader.py
#+++ b/modules/config_loader.py
#@@ -1,27 +1,26 @@
 # === modules/config_loader.py ===
# === modules/config_loader.py ===
from dotenv import dotenv_values 
import streamlit as st
import yaml
import os
 
 
def load_config():
     env_config = dotenv_values(".env.pa")
     config_file = "config.yaml"
     yaml_config = {}
     if os.path.exists(config_file):
         with open(config_file, "r") as f:
             yaml_config = yaml.safe_load(f)
     merged = {**yaml_config, **env_config}
     st.session_state["config"] = merged
     return merged
 
def authenticate():
     cfg = st.session_state.get("config", {})
     required_pw = cfg.get("APP_PASSWORD")
     if not required_pw:
         return True
 
     if "authenticated" not in st.session_state:
         pw = st.text_input("Enter app password", type="password")
         if pw == required_pw:
             st.session_state["authenticated"] = True
             st.success("Authenticated successfully!")
         else:
             st.error("Authentication failed. Please try again.")
             return False          

#EOF
