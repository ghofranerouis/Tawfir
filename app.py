import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="نظام توفير - جامعة غليزان",
    page_icon="tawfir/logo.png",
    layout="wide"
)

# 2. CSS احترافي وشامل لضبط RTL (بما في ذلك السايدبار والتقارير)
st.markdown("""
<style>
    /* ضبط الاتجاه العام للتطبيق */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        direction: rtl !important;
        text-align: right !important;
    }

    /* ضبط السايدبار بشكل خاص */
    [data-testid="stSidebar"] section {
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* ضبط تسميات المدخلات في السايدبار */
    .stSelectbox label, .stRadio label, .stHeader h3 {
        text-align: right !important;
        width: 100% !important;
        direction: rtl !important;
        display: block !important;
    }

    /* تنسيق حاوية العنوان */
    .header-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    /* تنسيق البطاقات الإحصائية */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        border-right: 5px solid #1a4a7a;
        direction: rtl !important;
    }

    .big-number { font-size: 30px; font-weight: bold; color: #1a4a7a; }
    .small-label { font-size: 14px; color: #6b7280; }

    /* التقرير الإداري (تم إصلاح RTL هنا) */
    .decision-box {
        background-color: #0f172a;
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-top: 25px;
        line-height: 1.8;
        direction: rtl !important;
        text-align: right !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. منطق البيانات
@st.cache_data
def load_data():
    days = [0, 1, 2, 3, 4] * 40
    data = []
    for d in days:
        attendance = 1150
        if d in [1, 3]: attendance += 350
        if d == 4: attendance -= 500
        weather_type = np.random.choice(["مشمس", "غائم", "ممطر"])
        if weather_type == "ممطر": attendance -= 120
        data.append([d, weather_type, attendance + np.random.randint(-30, 30)])
    return pd.DataFrame(data, columns=["كود_اليوم", "الطقس", "عدد_الحضور"])

df = load_data()
day_map = {"الأحد": 0, "الاثنين": 1, "الثلاثاء": 2, "الأربعاء": 3, "الخميس": 4}
weather_map = {"مشمس": 0, "غائم": 1, "ممطر": 2}
df["كود_الطقس"] = df["الطقس"].map(weather_map)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[["كود_اليوم", "كود_الطقس"]], df["عدد_الحضور"])

# 4. السايدبار (إعدادات التوقع)
with st.sidebar:
    st.markdown("<h3 style='text-align: right;'>⚙️ إعدادات التوقع</h3>", unsafe_allow_html=True)
    st.write("---")
    day_choice = st.selectbox("اليوم الدراسي:", list(day_map.keys()))
    weather_choice = st.selectbox("حالة الطقس:", list(weather_map.keys()))
    st.write("<br>", unsafe_allow_html=True)
    run_prediction = st.button("🚀 معالجة البيانات")

# 5. الهيدر الرسمي
col_space, col_title, col_logo = st.columns([0.5, 4, 1])
with col_logo:
    try: st.image("tawfir/logo.png", width=100)
    except: st.write("🏫")
with col_title:
    st.markdown("""
    <div class="header-card">
        <div style="font-size: 24px; font-weight: 800; color: #111827;">جامعة أحمد زبانة – غليزان</div>
        <div style="font-size: 16px; color: #6b7280;">منصة "توفير" الذكية لدعم القرار</div>
    </div>
    """, unsafe_allow_html=True)

# 6. قسم النتائج والتوقعات
if run_prediction:
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        if i == 30: status_text.markdown("<p style='text-align: right;'>🔍 جلب سجلات الرقمنة...</p>", unsafe_allow_html=True)
        if i == 70: status_text.markdown("<p style='text-align: right;'>🤖 تحليل الأنماط السلوكية...</p>", unsafe_allow_html=True)
    
    status_text.empty()
    progress_bar.empty()

    pred = model.predict([[day_map[day_choice], weather_map[weather_choice]]])[0]
    bread = int(pred * 1.25)
    saved = 2500 - bread

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="small-label">الحضور المتوقع</div><div class="big-number">{int(pred)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="small-label">طلبية الخبز</div><div class="big-number">{bread}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="small-label">الهدر المُجنب</div><div class="big-number">{saved}</div></div>', unsafe_allow_html=True)

    # التقرير الإداري - محاذاة يمين مطلقة
    st.markdown(f"""
    <div class="decision-box">
        <div style="font-weight: bold; font-size: 19px; border-bottom: 1px solid #475569; padding-bottom: 10px; margin-bottom: 10px;">📄 التقرير الإداري النهائي</div>
        بناءً على المعطيات الرقمية ليوم <b>{day_choice}</b> وظروف الطقس (<b>{weather_choice}</b>)، 
        توصي المنصة بتقليص الطلبية لتكون <b>{bread}</b> وحدة، 
        مما يساهم في توفير ميزانية تعادل <b>{saved}</b> خبزة مقارنة بالاستهلاك غير المرشد.
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

# 7. لوحة البيانات الإحصائية (إصلاح الخلل البصري)
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: right;'>📊 لوحة التحليلات الإحصائية</h3>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 10px;'>📈 اتجاهات الحضور الأسبوعية</div>", unsafe_allow_html=True)
    chart_df = df.copy()
    day_names = {0:"الأحد", 1:"الاثنين", 2:"الثلاثاء", 3:"الأربعاء", 4:"الخميس"}
    chart_df['اليوم'] = chart_df['كود_اليوم'].map(day_names)
    st.line_chart(chart_df.groupby("اليوم")["عدد_الحضور"].mean())

with colB:
    st.markdown("<div style='text-align: right; font-weight: bold; margin-bottom: 10px;'>🌦️ تحليل تأثير المناخ</div>", unsafe_allow_html=True)
    st.bar_chart(df.groupby("الطقس")["عدد_الحضور"].mean())

with st.expander("📂 عرض سجل الرقمنة الخام (تنسيق RTL)"):
    st.markdown("<div dir='rtl' style='text-align: right;'>", unsafe_allow_html=True)
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)