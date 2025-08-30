# -*- coding: utf-8 -*-
# Harmonia · v2.1
# - Sidebar: neon glass style
# - Single input: "Reference Nearest Matches (Top-3)" naming
# - Batch: robust invalid_type extraction, includes 'advertisement'

import ast, re, math, hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Harmonia", page_icon="✨", layout="wide")
st.markdown("""
<style>
/* 在这里加 CSS 样式 */
/* 控制 radio 文字变亮 */
.stSidebar .stRadio label, .sidebar .stRadio label {
    color: #E0E6FF !important;
    font-weight: 500;
}

/* 选中态高亮 */
.stSidebar .stRadio div[role='radiogroup'] label[data-baseweb="radio"]:has(input:checked),
.sidebar .stRadio div[role='radiogroup'] label:has(input:checked) {
    color: #66B2FF !important;
    font-weight: 600;
/* ===== 让 sidebar 里的 radio（ver1/ver2/ver3）文字常亮 ===== */
div[data-testid="stSidebar"] .stRadio label,
div[data-testid="stSidebar"] .stRadio label span,
div[data-testid="stSidebar"] .stRadio p{
  color: #E0E6FF !important;         /* 亮蓝白 */
  -webkit-text-fill-color:#E0E6FF !important;
  opacity: 1 !important;             /* 取消灰度 */
  filter: none !important;
}

/* 选中项再稍微更亮一点（两种写法同时给，兼容不同浏览器/版本） */
div[data-testid="stSidebar"] .stRadio label:has(input:checked) span{
  color:#9CC7FF !important;
  -webkit-text-fill-color:#9CC7FF !important;
  font-weight:700 !important;
}
div[data-testid="stSidebar"] .stRadio label input:checked + div span{
  color:#9CC7FF !important;
  -webkit-text-fill-color:#9CC7FF !important;
  font-weight:700 !important;
}

}
# 优先使用 segmented_control，回退到 radio（保证兼容）
/* 让 ver1 / ver2 / ver3 的文字始终显示为白色 */
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] span {
  color: #FFFFFF !important;   /* 白色 */
  -webkit-text-fill-color: #FFFFFF !important;
  opacity: 1 !important;       /* 取消灰色淡化 */
  filter: none !important;
}


</style>
""", unsafe_allow_html=True)

# ---------- Styles (sidebar + hero upgraded) ----------
st.markdown("""
<style>
:root{
  --bg:#0B0F19; --panel:#0F1629; --glass:#0F1A2DAA; --border:#1F2A44;
  --text:#E9EDF5; --muted:#A6B0BE; --primary:#6E80FF; --accent:#22D3EE;
  --success:#22C55E; --warn:#FB923C; --danger:#F43F5E;
}
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 600px at 12% 12%, rgba(110,128,255,0.22), transparent),
    radial-gradient(900px 600px at 88% 0%,  rgba(124,58,237,0.20), transparent),
    linear-gradient(180deg, var(--bg), #070A12 50%, var(--bg));
  color: var(--text);
}
/* Hero brighter */
.hero{
  position:relative; padding:22px 24px; border:1px solid var(--border); border-radius:20px;
  background: linear-gradient(140deg, rgba(110,128,255,.18), rgba(124,58,237,.14) 55%, rgba(34,211,238,.12));
  box-shadow: 0 16px 46px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
}
.h1{
  font-size: 34px; font-weight: 900; letter-spacing:.2px;
  background: linear-gradient(90deg,#FFFFFF,#E8E9FF 25%, #BFD1FF 55%, #8DB6FF 80%);
  -webkit-background-clip:text; color:transparent;
  text-shadow: 0 0 22px rgba(164,179,255,.55), 0 0 8px rgba(164,179,255,.45);
}
.sub{ color:var(--text); opacity:.84; font-size:.98rem; margin-top:4px }

/* Sidebar neon glass */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0a1223, #0c1220);
  border-right: 1px solid #162243;
  box-shadow: inset -1px 0 0 rgba(255,255,255,.03);
}
.sb-card{
  background: #0e1a33aa;
  border: 1px solid #1f2a44;
  border-radius: 14px;
  padding: 12px 12px 2px 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
  margin-bottom: 14px;
}
.sb-title{
  font-weight: 900; letter-spacing:.2px; margin-bottom: 8px;
  background: linear-gradient(90deg,#e8ecff,#b7c6ff 60%, #8db6ff 90%);
  -webkit-background-clip:text; color:transparent;
  text-shadow: 0 0 14px rgba(144,160,255,.45);
}
label, .stTextInput label, .stTextArea label { color:#E9EDF5 !important; opacity:.96 !important; }

/* Controls */
button[kind="primary"], button[kind="secondary"], .stButton>button, .stDownloadButton>button{
  border-radius:12px !important; border:1px solid rgba(255,255,255,.16);
  color:#fff !important; background: linear-gradient(180deg, #6E80FF, #394BFF) !important;
  box-shadow: 0 12px 28px rgba(110,128,255,.45), inset 0 1px 0 rgba(255,255,255,.08);
}
button[kind="primary"]:hover, .stButton>button:hover, .stDownloadButton>button:hover{
  filter:brightness(1.06); transform: translateY(-1px);
}
[data-baseweb="slider"] div[role="slider"]{
  background: radial-gradient(8px 8px at 50% 50%, #fff, #cbd5e1);
  box-shadow: 0 0 0 4px rgba(110,128,255,.28);
}
[data-baseweb="slider"] div[data-testid="stThumbValue"]{ color:#fff; background:#6E80FF }

.card{
  background: #0F1A2DAA; border:1px solid var(--border); border-radius:16px; padding:14px 16px;
  box-shadow: 0 10px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.05);
  animation: rise .6s ease both;
}
@keyframes rise{ from{opacity:0; transform: translateY(10px)} to{opacity:1; transform: translateY(0)} }

.pill{ display:inline-block; padding:.28rem .65rem; border-radius:999px; font-weight:800; font-size:.92rem }
.pill-valid{   background:#0C2C19; color:#86EFAC; border:1px solid rgba(34,197,94,.55) }
.pill-invalid{ background:#3B0A14; color:#FECACA; border:1px solid rgba(244,63,94,.55) }

.small{ color:#B3BDD0; font-size:.92rem }
.kpi{ font-size:26px; font-weight:900; letter-spacing:.3px }
</style>
""", unsafe_allow_html=True)

# ---------- LLM choices ----------
LLM_CHOICES = ["Qwen3-8B", "Qwen2.5-14B", "Gemma3-12B", "GPT-4o-mini", "GPT-4o"]
PROMPTS = {
    "Zero Shot": "Classify review as valid/invalid (advertisement/irrelevant/rant_no_visit). Be strict with links and unrelated topics.",
    "Few Shot": "Focus on experiential evidence. If the user says they haven't visited or cites hearsay, mark as rant_no_visit.",
    "Few Shot + CoT": "High-precision mode. Only mark valid if the comment clearly describes an on-site experience.",
    "Few Shot + CoT + RAG": "Advanced mode combining retrieval, reasoning, and contextual analysis for optimal accuracy.",
}
DEFAULT_PROMPT_KEY = "Zero Shot"

# ---------- Stubs (replace with real model calls) ----------
def has_link(text:str)->bool:
    if not isinstance(text,str): return False
    return bool(re.search(r"(https?://|www\\.|\\.[a-z]{2,4}\\b)", text, flags=re.I))

def hash_float_0_1(s:str)->float:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16)/0xFFFFFFFF

def model1_valid_proba(text:str)->float:
    return hash_float_0_1("m1:"+text)

def model2_llm_label(text:str, model_choice:str, prompt_key:str)->tuple[str,str]:
    if has_link(text): return "invalid","advertisement"
    s = hash_float_0_1(f"m2:{model_choice}:{prompt_key}:{text}")
    if s < .25: return "invalid","irrelevant"
    if s < .35: return "invalid","rant_no_visit"
    return "valid",""

def extract_first_image_url(pics)->str|None:
    try:
        if isinstance(pics,list) and pics:
            u = pics[0].get("url")
            if isinstance(u,list) and u: return u[0]
            if isinstance(u,str): return u
    except Exception: pass
    return None

def model3_clip_similarity(text:str, pics)->float|None:
    url = extract_first_image_url(pics)
    if not url: return None
    return hash_float_0_1(f"m3:{text}:{url}")

def haversine_km(lat1,lon1,lat2,lon2):
    try:
        R=6371; phi1,phi2=map(math.radians,[float(lat1),float(lat2)])
        dphi=math.radians(float(lat2)-float(lat1)); dl=math.radians(float(lon2)-float(lon1))
        a=math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))
    except: return None

def model4_metadata_flags(df:pd.DataFrame)->pd.DataFrame:
    out=pd.DataFrame(index=df.index); out["meta_reason"]=""; out["meta_invalid"]=False
    if {"rating","business_avg_rating"}.issubset(df.columns):
        diff=(df["rating"]-df["business_avg_rating"]).abs(); m=diff>=3
        out.loc[m,"meta_invalid"]=True; out.loc[m,"meta_reason"]=(out["meta_reason"]+"; metadata_extreme_rating").str.strip("; ")
    if {"user_id","time"}.issubset(df.columns):
        dd=df.copy(); dd["time"]=pd.to_datetime(dd["time"], errors="coerce", unit="ms")
        for _,g in dd.groupby("user_id"):
            g=g.sort_values("time")
            for i in range(len(g)):
                t0=g.iloc[i]["time"]; 
                if pd.isna(t0): continue
                within=g[(g["time"]>=t0)&(g["time"]<=t0+timedelta(hours=24))]
                if len(within)>=10:
                    idx=within.index; out.loc[idx,"meta_invalid"]=True
                    out.loc[idx,"meta_reason"]=(out.loc[idx,"meta_reason"].replace("","metadata_burst")+"; metadata_burst").str.strip("; ")
    if {"user_id","time","lat","lon"}.issubset(df.columns):
        dd=df.copy(); dd["time"]=pd.to_datetime(dd["time"], errors="coerce", unit="ms")
        for _,g in dd.groupby("user_id"):
            g=g.sort_values("time")
            for i in range(len(g)-1):
                t1,t2=g.iloc[i]["time"],g.iloc[i+1]["time"]
                if pd.isna(t1) or pd.isna(t2): continue
                dt = (t2 - t1).total_seconds() / 3600
                d  = haversine_km(g.iloc[i]["lat"], g.iloc[i]["lon"], g.iloc[i+1]["lat"], g.iloc[i+1]["lon"])

                if dt and d and dt>0 and d/dt>500:
                    idx=[g.index[i],g.index[i+1]]
                    out.loc[idx,"meta_invalid"]=True
                    out.loc[idx,"meta_reason"]=(out.loc[idx,"meta_reason"].replace("","metadata_teleport")+"; metadata_teleport").str.strip("; ")
    return out

# ---------- Screenshot Analysis Models ----------
def model5_screenshot_analysis(screenshot_image)->tuple[str,str]:
    """Analyze Google review screenshot and classify as valid/invalid"""
    if screenshot_image is None:
        return "invalid", "no_screenshot"
    
    # Mock analysis based on image hash
    image_bytes = screenshot_image.getvalue() if hasattr(screenshot_image, 'getvalue') else b""
    s = hash_float_0_1(f"screenshot:{len(image_bytes)}")
    
    if s < 0.15: return "invalid", "fake_review"
    if s < 0.25: return "invalid", "spam_content"  
    if s < 0.35: return "invalid", "irrelevant_content"
    if s < 0.45: return "invalid", "advertisement"
    return "valid", ""

def model6_business_image_similarity(review_text:str, business_image, similarity_threshold:float=0.6)->tuple[str,float|None,str]:
    """Compare review text with business image for consistency"""
    if business_image is None:
        return "valid", None, ""
    
    # Mock similarity calculation
    image_bytes = business_image.getvalue() if hasattr(business_image, 'getvalue') else b""
    text_hash = hash_float_0_1(f"text:{review_text}")
    image_hash = hash_float_0_1(f"image:{len(image_bytes)}")
    similarity = abs(text_hash - image_hash)  # Mock similarity score
    
    if similarity >= similarity_threshold:
        return "valid", similarity, ""
    else:
        return "invalid", similarity, "image_text_mismatch"

# ---------- Pipeline ----------
def stage0_rules(text:str)->tuple[str,str]:
    if not str(text or "").strip(): return "invalid","missing_comment"
    if has_link(text): return "invalid","advertisement"
    return "pass",""

def stage1_model1(text:str, low:float, high:float)->tuple[str,float,str]:
    p=model1_valid_proba(text)
    if p<low:
        rsn="advertisement" if has_link(text) else ("irrelevant" if hash_float_0_1("m1r:"+text)<.5 else "rant_no_visit")
        return "invalid",p,rsn
    if p>high: return "valid",p,""
    return "unsure",p,""

def stage2_model2(text:str, model_choice:str, prompt_key:str)->tuple[str,str]:
    return model2_llm_label(text, model_choice, prompt_key)

def stage3_model3(text:str, pics, thr:float)->tuple[str,float|None,str]:
    sim=model3_clip_similarity(text, pics)
    if sim is None: return "valid",None,""
    if float(sim)>=thr: return "valid",sim,""
    return "invalid",sim,"image_text_mismatch"

def run_pipeline_single(row:dict, low:float, high:float, clip_thr:float, model_choice:str, prompt_key:str)->dict:
    text=str(row.get("text") or row.get("user_comment") or "")
    s0,rs0=stage0_rules(text)
    if s0=="invalid": return {"final":{"label":"invalid","reason":rs0}, "s0":(s0,rs0)}
    s1,p1,rs1=stage1_model1(text, low, high)
    if s1=="invalid": return {"final":{"label":"invalid","reason":rs1}, "s0":("pass",""), "s1":(s1,p1,rs1)}
    if s1=="valid": after2=("valid","")
    else:
        s2,rs2=stage2_model2(text, model_choice, prompt_key)
        if s2=="invalid": return {"final":{"label":"invalid","reason":rs2}, "s0":("pass",""), "s1":(s1,p1,""), "s2":(s2,rs2)}
        after2=("valid","")
    s3,sim3,rs3=stage3_model3(text, row.get("pics"), clip_thr)
    if s3=="invalid": return {"final":{"label":"invalid","reason":rs3}, "s0":("pass",""), "s1":(s1,p1,""), "s3":(s3,sim3,rs3)}
    df=pd.DataFrame([row]); flags=model4_metadata_flags(df)
    if len(flags)>0 and bool(flags.iloc[0]["meta_invalid"]):
        r=str(flags.iloc[0]["meta_reason"]); return {"final":{"label":"invalid","reason":r}, "s0":("pass",""), "s1":(s1,p1,""), "s3":("valid",sim3,""), "s4":(True,r)}
    return {"final":{"label":"valid","reason":""}, "s0":("pass",""), "s1":(s1,p1,""), "s3":("valid",sim3,""), "s4":(False,"")}

def run_pipeline_batch(df:pd.DataFrame, low:float, high:float, clip_thr:float, model_choice:str, prompt_key:str)->pd.DataFrame:
    rows=[]
    for _,row in df.iterrows():
        d=row.to_dict()
        r=run_pipeline_single(d, low, high, clip_thr, model_choice, prompt_key)
        final=r["final"]; s1=r.get("s1"); s3=r.get("s3"); s4=r.get("s4")
        rows.append({**d,
            "s0_label": r["s0"][0] if "s0" in r else "", "s0_reason": r["s0"][1] if "s0" in r else "",
            "s1_label": s1[0] if s1 else "", "s1_p_valid": s1[1] if s1 else np.nan, "s1_reason": s1[2] if s1 else "",
            "s3_label": s3[0] if s3 else "", "s3_similarity": s3[1] if s3 else np.nan, "s3_reason": s3[2] if s3 else "",
            "s4_meta_invalid": s4[0] if s4 else False, "s4_meta_reason": s4[1] if s4 else "",
            "final_label": final["label"], "final_reason": final["reason"]})
    out=pd.DataFrame(rows)
    # ---- Normalize invalid type for charts
    def pick_invalid_type(row):
        # priority by stage
        for key in ["s0_reason","s1_reason","s3_reason","s4_meta_reason","final_reason"]:
            v=str(row.get(key) or "").strip()
            if not v: continue
            # if metadata reason has multiple tokens, keep first tag
            v=v.split(";")[0].strip()
            return v
        return "unknown"
    out["invalid_type"]=np.where(out["final_label"]=="invalid", out.apply(pick_invalid_type, axis=1), "valid")
    return out

# ---------- Simple reference nearest (model-agnostic) ----------
def tokenize(s:str)->set[str]:
    s=(s or "").lower(); s=re.sub(r"[^a-z0-9\u4e00-\u9fff]+"," ",s); 
    return set([w for w in s.split() if w])

def jaccard(a:set[str], b:set[str])->float:
    if not a or not b: return 0.0
    return len(a & b)/len(a | b)

def top3_similar(query_text:str, ref_df:pd.DataFrame, text_col:str="text")->list[tuple[str,float]]:
    tq=tokenize(query_text); sims=[]
    for _,r in ref_df.iterrows():
        t=str(r.get(text_col,"") or ""); sims.append((t, jaccard(tq, tokenize(t))))
    sims.sort(key=lambda x:x[1], reverse=True)
    return sims[:3]

# ---------- Natural Language Field Extraction ----------
def extract_review_fields(natural_input: str) -> dict:
    """Extract review fields from natural language input"""
    import re
    from datetime import datetime
    
    fields = {
        "user_id": "NA",
        "name": "NA", 
        "time": "NA",
        "rating": "NA",
        "text": "NA",
        "gmap_id": "NA"
    }
    
    if not natural_input.strip():
        return fields
    
    # Extract user ID (numbers, emails, or @mentions)
    user_id_match = re.search(r'(?:user[_\s]*id[:\s]*|id[:\s]*|@)([a-zA-Z0-9_@.-]+)', natural_input, re.IGNORECASE)
    if user_id_match:
        fields["user_id"] = user_id_match.group(1)
    
    # Extract name (after "name:", "by", "from", or quoted)
    name_match = re.search(r'(?:name[:\s]*|by[:\s]+|from[:\s]+|reviewer[:\s]*)["\']?([A-Za-z\s]{2,30})["\']?', natural_input, re.IGNORECASE)
    if name_match:
        fields["name"] = name_match.group(1).strip()
    
    # Extract rating (X/5, X stars, X out of 5)
    rating_match = re.search(r'(?:rating[:\s]*|rated[:\s]*|stars?[:\s]*|score[:\s]*)(\d)[/\s]*(?:out\s*of\s*)?[5\s]*(?:stars?)?', natural_input, re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
        if 1 <= rating <= 5:
            fields["rating"] = rating
    
    # Extract time/date
    time_patterns = [
        r'(?:time[:\s]*|date[:\s]*|on[:\s]*)(\d{10,13})',  # epoch timestamp
        r'(?:time[:\s]*|date[:\s]*|on[:\s]*)(\d{4}[-/]\d{1,2}[-/]\d{1,2})',  # date format
        r'(?:time[:\s]*|date[:\s]*|on[:\s]*)(\d{1,2}[-/]\d{1,2}[-/]\d{4})',  # US date format
    ]
    for pattern in time_patterns:
        time_match = re.search(pattern, natural_input, re.IGNORECASE)
        if time_match:
            fields["time"] = time_match.group(1)
            break
    
    # Extract Google Maps ID
    gmap_match = re.search(r'(?:gmap[_\s]*id[:\s]*|google[_\s]*maps?[_\s]*id[:\s]*|maps?[_\s]*id[:\s]*)([0-9a-fx:]+)', natural_input, re.IGNORECASE)
    if gmap_match:
        fields["gmap_id"] = gmap_match.group(1)
    
    # Extract review text (everything else, or after "review:", "comment:", "text:")
    text_match = re.search(r'(?:review[:\s]*|comment[:\s]*|text[:\s]*|says?[:\s]*)["\']?([^"\']+)["\']?', natural_input, re.IGNORECASE)
    if text_match:
        fields["text"] = text_match.group(1).strip()
    else:
        # If no explicit markers, treat the longest sentence as review text
        sentences = re.split(r'[.!?]+', natural_input)
        longest_sentence = max(sentences, key=len, default="").strip()
        if len(longest_sentence) > 20:  # Reasonable review length
            fields["text"] = longest_sentence
    
    return fields

# ---------- Header ----------
st.markdown('<div class="hero"><div class="h1">Harmonia</div><div class="sub">Four-stage pipeline · thresholds & model selection</div></div>', unsafe_allow_html=True)
st.write("")

# ---------- Sidebar (neon groups) ----------
st.sidebar.markdown('<div class="sb-card"><div class="sb-title">Stage2</div>', unsafe_allow_html=True)
low_cut  = st.sidebar.slider("Fast BERT lower invalid threshold", 0.0, 0.5, 0.20, 0.01)
high_cut = st.sidebar.slider("Fast BERT upper valid threshold",   0.5, 1.0, 0.80, 0.01)
clip_thr = st.sidebar.slider("Clip similarity threshold", 0.0, 1.0, 0.20, 0.01)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sb-card"><div class="sb-title">Stage3</div>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Model", LLM_CHOICES, index=0)
prompt_key   = st.sidebar.radio("Prompt", list(PROMPTS.keys()), index=list(PROMPTS).index(DEFAULT_PROMPT_KEY), horizontal=True)
st.sidebar.markdown(f"<p style='color:#E0E6FF'>{PROMPTS[prompt_key]}</p>", unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Default reference dataset
ref_df = pd.DataFrame({"text":[
    "Great staff and the clinic is super clean. Highly recommend.",
    "Terrible service. I waited 2 hours and nobody came to help.",
    "Visit my shop at www.bestdeal.com and get promo now!",
    "I haven't been there but my friend said it's awful.",
    "The latte art was beautiful and prices were fair.",
    "Amazing dentist, painless procedure, friendly nurses.",
]})

# ---------- Tabs ----------
tab_single, tab_batch, tab_screenshot = st.tabs(["Single Input", "CSV Batch", "Screenshot Analysis"])

with tab_single:
    st.markdown("##### Natural Language Review Input")
    st.caption("Enter review information in natural language. The system will automatically extract fields like user ID, name, rating, etc.")
    
    # Natural language input
    default_input = """User ID: 106533466896145407182, Name: Amy VG, Time: 1568748357166, Rating: 5 stars, Review: I can't say I've ever been excited about a dentist visit before but this place changed my mind, Google Maps ID: 0x87ec2394c2cd9d2d:0xd1119cfbee0da6f3"""
    
    natural_input = st.text_area(
        "Review Information", 
        value=default_input,
        height=150,
        placeholder="Example: User John Smith rated 4 stars on 2023-01-15, review: Great service and friendly staff, user ID: 12345"
    )
    
    # Extract and display fields
    if natural_input.strip():
        extracted_fields = extract_review_fields(natural_input)
        
        st.markdown("#### Extracted Fields")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**User ID:** {extracted_fields['user_id']}")
            st.write(f"**Name:** {extracted_fields['name']}")
            st.write(f"**Time:** {extracted_fields['time']}")
        with col2:
            st.write(f"**Rating:** {extracted_fields['rating']}")
            st.write(f"**Google Maps ID:** {extracted_fields['gmap_id']}")
        
        st.write(f"**Review Text:** {extracted_fields['text']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Business image upload for similarity comparison
    st.markdown("##### Business Image Upload")
    st.caption("Upload business image for similarity comparison with review content")
    business_image = st.file_uploader("Upload Business Image", type=['png', 'jpg', 'jpeg'], key="business_img")
    
    if business_image:
        st.image(business_image, caption="Uploaded Business Image", width=200)
    
    run_one = st.button("Run Analysis", use_container_width=True)

    if run_one:
        # Extract fields from natural language input
        if not natural_input.strip():
            st.error("Please enter review information to analyze.")
        else:
            # Show processing message and wait
            with st.spinner("Processing analysis..."):
                import time
                time.sleep(3.0)  # 3 second delay as requested
            extracted_fields = extract_review_fields(natural_input)
            
            # Build review dict from extracted fields
            row = {}
            row["user_id"] = extracted_fields["user_id"]
            row["name"] = extracted_fields["name"]
            row["text"] = extracted_fields["text"]
            row["gmap_id"] = extracted_fields["gmap_id"]
            
            # Handle rating
            if extracted_fields["rating"] != "NA":
                row["rating"] = extracted_fields["rating"]
            else:
                row["rating"] = "NA"
            
            # Handle time field
            if extracted_fields["time"] != "NA":
                try:
                    # Try to convert to int if it looks like epoch timestamp
                    if extracted_fields["time"].isdigit() and len(extracted_fields["time"]) >= 10:
                        row["time"] = int(extracted_fields["time"])
                    else:
                        row["time"] = extracted_fields["time"]
                except ValueError:
                    row["time"] = "NA"
            else:
                row["time"] = "NA"
            
            # Set default values for pipeline
            row["user_comment"] = row["text"]
            row["business_avg_rating"] = 4.2
            row["lat"] = 37.78
            row["lon"] = -122.41
            row["pics"] = []  # Empty pics for now
            
            # Check if any critical field is NA
            critical_fields = ["user_id", "name", "text", "time", "gmap_id"]
            na_fields = [field for field in critical_fields if row[field] == "NA"]
            
            # Run standard pipeline
            result = run_pipeline_single(row, low_cut, high_cut, clip_thr, model_choice, prompt_key)
            
            # Run business image similarity check if image is provided
            image_result = None
            if business_image and row["text"] != "NA":
                image_result = model6_business_image_similarity(row["text"], business_image, 0.6)  # Fixed threshold
            
            # Override result if image similarity fails
            if image_result and image_result[0] == "invalid":
                result["final"] = {"label": "invalid", "reason": image_result[2]}
                result["image_similarity"] = {"similarity": image_result[1], "status": "failed"}
            elif image_result:
                result["image_similarity"] = {"similarity": image_result[1], "status": "passed"}
            
            # Override result if critical NA fields exist
            if na_fields:
                result["final"] = {"label": "invalid", "reason": f"missing_fields: {', '.join(na_fields)}"}
                result["na_fields"] = na_fields

            # Display NA fields warning if any
            if na_fields:
                st.warning(f"⚠️ Missing fields detected: {', '.join(na_fields)}. This will result in INVALID classification.")

            # Reference Nearest Matches section removed as requested

            # --- Business Image Similarity (if applicable)
            if "image_similarity" in result:
                st.markdown("#### Business Image Similarity")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                img_sim = result["image_similarity"]
                if img_sim["status"] == "passed":
                    st.success(f"✅ Image-Text similarity: {img_sim['similarity']:.3f} (≥ 0.60)")
                else:
                    st.error(f"❌ Image-Text similarity: {img_sim['similarity']:.3f} (< 0.60)")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- Final Result
            st.markdown("#### Final Result")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            final = result["final"]
            pill = '<span class="pill pill-valid">VALID</span>' if final["label"]=="valid" else '<span class="pill pill-invalid">INVALID</span>'
            st.markdown(pill, unsafe_allow_html=True)
            if final["reason"]:
                st.caption("Reason: "+str(final["reason"]))
            
            # Display additional metrics
            s1 = result.get("s1"); s3 = result.get("s3"); bits = []
            if s1: bits.append(f"p_valid={s1[1]:.3f}")
            if s3 and s3[1] is not None: bits.append(f"clip_sim={float(s3[1]):.3f}")
            if "image_similarity" in result and result["image_similarity"]["similarity"] is not None:
                bits.append(f"img_sim={result['image_similarity']['similarity']:.3f}")
            if bits: st.caption(" · ".join(bits))
            st.markdown('</div>', unsafe_allow_html=True)

with tab_batch:
    st.markdown("##### Upload CSV")
    st.caption("Optional columns: user_id, name, time (supports epoch ms), rating, text, pics, resp, gmap_id, business_avg_rating, lat, lon, etc.")
    up = st.file_uploader("CSV", type=["csv"])
    use_demo = st.toggle("Use demo data if no CSV", value=True)

    df_raw=None
    selected_columns = None
    if up is not None:
        try: 
            df_raw=pd.read_csv(up)
            # Show column selection popup
            st.markdown("#### Column Selection")
            st.info("Please select which columns are relevant for this model:")
            
            if df_raw is not None:
                all_columns = df_raw.columns.tolist()
                selected_columns = st.multiselect(
                    "Select relevant columns:",
                    options=all_columns,
                    default=all_columns,
                    help="Choose which columns should be used in the analysis"
                )
                
                if selected_columns:
                    df_raw = df_raw[selected_columns]
                    st.success(f"Selected {len(selected_columns)} columns: {', '.join(selected_columns)}")
                else:
                    st.warning("Please select at least one column to proceed.")
                    df_raw = None
        except Exception as e: st.error(f"Failed to read CSV: {e}")

    if df_raw is None and use_demo:
        df_raw=pd.DataFrame({
            "user_id":[f"u{i%3}" for i in range(12)],
            "name":[f"user{i}" for i in range(12)],
            "time":[1568748357166 + i*3600_000 for i in range(12)],
            "rating":[5,5,4,1,5,2,4,2,4,5,5,1],
            "text":[
                "Amazing dentist, painless procedure, friendly nurses.",
                "Best sushi! Visit www.super-sushi.com for coupons",
                "I love my new phone, anyway the park is big.",
                "Never been here, but my friend said it's terrible",
                "Great ambience, will come again",
                "Discount code SAVE20 at example.com",
                "Cozy vibe, fair prices, clean tables.",
                "Food meh. Long wait and rude staff.",
                "Parking convenient. Staff courteous.",
                "Visit my site for promo deals http://example.com",
                "Loved my experience at the clinic today. Gorgeous office.",
                "I heard it is bad but I didn't go."
            ],
            "pics":[
                "", "", "", "", "", "", "{'url':['http://ex.com/a.jpg']}", "", "", "{'url':['http://ex.com/b.jpg']}", "", ""
            ],
            "business_avg_rating":[4.2]*12,
            "lat":[37.77,40.71,37.77,34.05,37.77,40.71,51.51,48.85,34.05,35.68,37.77,40.71],
            "lon":[-122.41,-74,-122.41,-118.24,-122.41,-74,-0.13,2.35,-118.24,139.69,-122.41,-74],
        })

    if df_raw is not None and st.button("Run Pipeline on Batch", use_container_width=True):
        # recover pics if stringified
        if "pics" in df_raw.columns:
            def _fix(x):
                if isinstance(x,str):
                    try: return ast.literal_eval(x)
                    except: return ""
                return x
            df_raw["pics"]=df_raw["pics"].apply(_fix)
        res=run_pipeline_batch(df_raw, low_cut, high_cut, clip_thr, model_choice, prompt_key)
        st.session_state.batch_res=res

    if "batch_res" in st.session_state:
        res=st.session_state.batch_res
        st.markdown("##### Overview")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        total=len(res); valid_n=int((res["final_label"]=="valid").sum()); invalid_n=total-valid_n
        k1,k2,k3=st.columns(3)
        k1.markdown('<div class="small">Total</div>', unsafe_allow_html=True);   k1.markdown(f'<div class="kpi">{total}</div>', unsafe_allow_html=True)
        k2.markdown('<div class="small">Valid</div>', unsafe_allow_html=True);   k2.markdown(f'<div class="kpi" style="color:#86EFAC">{valid_n}</div>', unsafe_allow_html=True)
        k3.markdown('<div class="small">Invalid</div>', unsafe_allow_html=True); k3.markdown(f'<div class="kpi" style="color:#FECACA">{invalid_n}</div>', unsafe_allow_html=True)

        c1,c2=st.columns([0.48,0.52], gap="large")
        with c1:
            counts=res["final_label"].value_counts()
            fig=px.pie(counts, values=counts.values, names=counts.index, hole=0.62,
                       color=counts.index, color_discrete_map={"valid":"#22C55E","invalid":"#F43F5E"})
            fig.update_layout(margin=dict(l=6,r=6,t=6,b=6), legend_title=None, transition_duration=250)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # ✅ robust: reason categories (includes 'advertisement')
            inv=res[res["final_label"]=="invalid"]["invalid_type"].fillna("unknown")
            cat = inv.replace({"":"unknown"})
            rc=cat.value_counts().sort_values(ascending=False)
            
            # Ensure key categories always appear in the chart
            key_categories = ["irrelevant", "rant_no_visit", "advertisement"]
            for category in key_categories:
                if category not in rc.index:
                    rc[category] = 0
            
            # Sort to ensure consistent order with key categories first
            rc = rc.sort_values(ascending=False)
            figb=px.bar(rc, x=rc.index, y=rc.values, labels={"x":"invalid type","y":"count"})
            figb.update_layout(margin=dict(l=6,r=6,t=6,b=6))
            st.plotly_chart(figb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("##### Details")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download CSV", res.to_csv(index=False).encode("utf-8"),
                           "fusion_results.csv", "text/csv", use_container_width=True)

with tab_screenshot:
    st.markdown("##### Google Review Screenshot Analysis")
    st.caption("Upload Google review screenshots and use AI models to determine if the review is valid")
    
    # Screenshot upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Upload Screenshot**")
    screenshot_image = st.file_uploader("Upload Google Review Screenshot", type=['png', 'jpg', 'jpeg'], key="screenshot_img")
    
    if screenshot_image:
        st.image(screenshot_image, caption="Uploaded Screenshot", width=400)
        
        # Analysis button
        if st.button("Analyze Screenshot", use_container_width=True):
            # Show processing message with extended time
            with st.spinner("Processing screenshot analysis..."):
                import time
                time.sleep(4.0)  # Extended processing time for screenshot
            
            # Run screenshot analysis
            screenshot_result = model5_screenshot_analysis(screenshot_image)
            
            # Display user information and image as requested
            st.markdown("#### User Information")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Display user profile info
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(screenshot_image, width=150)
            with col2:
                st.markdown("**shivani saravanan**")
                st.caption("Local Guide • 16 reviews")
                st.caption("6 days ago")
                st.caption("Dinner")
                st.write("Dining at Ammakase was nothing short of magnificent. Our family chose the 10-course meal menu, and each course was presented with such elegance and artistry that it...")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Analysis Result")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Display result
            if screenshot_result[0] == "valid":
                pill = '<span class="pill pill-valid">VALID</span>'
                st.markdown(pill, unsafe_allow_html=True)
                st.success("✅ Screenshot appears to contain a valid Google review")
            else:
                pill = '<span class="pill pill-invalid">INVALID</span>'
                st.markdown(pill, unsafe_allow_html=True)
                st.error(f"❌ Screenshot is invalid")
                
                # Display specific invalid reason
                reason = screenshot_result[1]
                reason_descriptions = {
                    "fake_review": "Detected fake review content",
                    "spam_content": "Detected spam/advertisement content", 
                    "irrelevant_content": "Detected irrelevant content",
                    "advertisement": "Detected advertisement content",
                    "no_screenshot": "No valid screenshot detected"
                }
                
                reason_text = reason_descriptions.get(reason, reason)
                st.caption(f"Invalid Category: **{reason}** - {reason_text}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📸 Please upload a Google review screenshot to begin analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("##### Instructions")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    **How to use Screenshot Analysis:**
    
    1. **Take a screenshot** of a Google review from Google Maps or Google Search
    2. **Upload the image** using the file uploader above
    3. **Click "Analyze Screenshot"** to run the validation model
    4. **Review the results:**
       - **VALID**: The screenshot contains a legitimate Google review
       - **INVALID**: The screenshot has issues (fake content, spam, advertisements, etc.)
    
    **Invalid Categories:**
    - **fake_review**: Fake review content
    - **spam_content**: Spam/advertisement content  
    - **irrelevant_content**: Irrelevant content
    - **advertisement**: Advertisement content
    """)
    st.markdown('</div>', unsafe_allow_html=True)

