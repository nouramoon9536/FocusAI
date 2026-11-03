import streamlit as st
import pandas as pd
import numpy as np
import random, time, os, hashlib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title='FocusAI | بصمة الانتباه', page_icon='🧠', layout='wide', initial_sidebar_state='collapsed')

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = BASE
DATA_FILE = os.path.join(BASE, 'focusai_data.csv')

# ---------- data ----------
DATA_COLS = [
    'participant','date','sleep','bedtime','wake_time','screen','study','breaks',
    'test_time','score','accuracy','errors','avg_reaction_ms',
    'round1','round2','round3','round4'
]

# Features used by the predictive AI. These are lifestyle inputs only, matching the research question.
FEATURES = ['sleep','screen','study','breaks']


def load_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            for c in DATA_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            return df[DATA_COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=DATA_COLS)


def save_row(row):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')


def make_target(df):
    """Binary attention-performance level: 1 = good (>=70), 0 = needs improvement."""
    y = pd.to_numeric(df['score'], errors='coerce')
    return (y >= 70).astype(int)


def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ]),
        'Decision Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')),
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42, class_weight='balanced')),
        ]),
    }


def evaluate_models(df):
    """Train and evaluate the three research models using lifestyle inputs only.
    The target is the observed attention-performance level derived from the actual test score.
    No synthetic data or fabricated metrics are used.
    """
    work = df.copy()
    for c in FEATURES + ['score']:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=FEATURES + ['score']).copy()
    y = make_target(work)
    X = work[FEATURES]

    if len(work) < 8 or y.nunique() < 2 or y.value_counts().min() < 2:
        return {'ready': False, 'reason': 'تحتاج التجربة إلى 8 سجلات على الأقل وتنوعًا كافيًا في مستويات الأداء لتدريب النموذج وتقييمه دون اختلاق بيانات.'}

    from sklearn.model_selection import train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        return {'ready': False, 'reason': f'لا يمكن تقسيم البيانات حاليًا بشكل علمي: {e}'}

    results = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            'النموذج': name,
            'Accuracy': accuracy_score(y_test, pred),
            'F1': f1_score(y_test, pred, zero_division=0),
        })
        fitted[name] = model

    metrics = pd.DataFrame(results).sort_values(['F1','Accuracy'], ascending=False).reset_index(drop=True)
    return {
        'ready': True,
        'metrics': metrics,
        'models': fitted,
        'X': X,
        'y': y,
        'X_test': X_test,
        'y_test': y_test,
        'best': metrics.iloc[0]['النموذج'],
        'sample_size': len(work),
    }


def predict_current(df, features):
    result = evaluate_models(df)
    if not result.get('ready'):
        return result
    best_name = result['best']
    model = result['models'][best_name]
    x = pd.DataFrame([features], columns=FEATURES)
    pred = int(model.predict(x)[0])
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = float(model.predict_proba(x)[0][1])
    result['prediction'] = pred
    result['probability'] = probability
    result['model_name'] = best_name
    return result

# ---------- session ----------
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'participant' not in st.session_state:
    st.session_state.participant = 'FCS-' + hashlib.sha1(str(random.random()).encode()).hexdigest()[:6].upper()
if 'consent' not in st.session_state:
    st.session_state.consent = False
if 'last_score' not in st.session_state:
    st.session_state.last_score = None
if 'last_saved' not in st.session_state:
    st.session_state.last_saved = False
if 'test' not in st.session_state:
    st.session_state.test = {
        'index':0,'phase':'intro','started':None,'question_started':None,
        'answers':[],'results':{},'questions':[]
    }

# ---------- questions ----------
P = lambda name: os.path.join(IMG, name)
QUESTIONS = [
    {'section':'أوجد الاختلاف','badge':'الاختبار 1 • صورة 1','image':P('WhatsApp Image 2026-10-06 at 4.41.06 PM (1).jpeg'),'question':'كم عدد الاختلافات بين الصورتين؟','options':['8','9','10','11'],'correct':'10','duration':10},
    {'section':'أوجد الاختلاف','badge':'الاختبار 1 • صورة 2','image':P('WhatsApp Image 2026-10-06 at 4.41.06 PM (2).jpeg'),'question':'كم عدد الاختلافات بين الصورتين؟','options':['8','9','10','11'],'correct':'9','duration':10},
    {'section':'أوجد الاختلاف','badge':'الاختبار 1 • صورة 3','image':P('WhatsApp Image 2026-10-06 at 4.41.06 PM.jpeg'),'question':'كم عدد الاختلافات بين الصورتين؟','options':['3','2','4','5'],'correct':'3','duration':10},
    {'section':'الذاكرة البصرية','badge':'الاختبار 2 • صورة 1','image':P('WhatsApp Image 2026-10-06 at 4.41.06 PM (3).jpeg'),'question':'أين كانت علبة الأقلام؟','options':['يسار','يمين'],'correct':'يسار','duration':8},
    {'section':'الذاكرة البصرية','badge':'الاختبار 2 • صورة 2','image':P('WhatsApp Image 2026-10-06 at 4.41.06 PM (4).jpeg'),'question':'ما الواجهة الظاهرة على اللابتوب؟','options':['واجهة كود','واجهة ألعاب','واجهة صور','واجهة بريد'],'correct':'واجهة كود','duration':8},
    {'section':'الذاكرة البصرية','badge':'الاختبار 2 • صورة 3','image':P('WhatsApp Image 2026-10-06 at 4.41.07 PM.jpeg'),'question':'كم مجسم كرة أرضية ظهر في الصورة؟','options':['4','5','6','7'],'correct':'6','duration':8},
    {'section':'العدّ والتركيز','badge':'الاختبار 3 • صورة 1','image':P('WhatsApp Image 2026-10-06 at 4.41.07 PM (1).jpeg'),'question':'كم عدد المربعات الملوّنة في الوردة؟','options':['28','30','32','34'],'correct':'32','duration':9},
    {'section':'العدّ والتركيز','badge':'الاختبار 3 • صورة 2','image':P('WhatsApp Image 2026-10-06 at 4.41.07 PM (2).jpeg'),'question':'كم عدد المربعات باللون البني في الصبار؟','options':['26','28','30','32'],'correct':'30','duration':9},
    {'section':'التعرّف البصري','badge':'الاختبار 3 • صورة 3','image':P('whale_half.jpeg'),'question':'ما الصورة الكاملة التي ينتمي إليها هذا الجزء؟','options':['دلفين','نافورة','حوت'],'correct':'حوت','duration':7},
]

# ---------- style ----------
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#07111f;--panel:rgba(15,27,46,.72);--line:rgba(255,255,255,.09);--txt:#f7fbff;--muted:#9eb0c6;--cyan:#59e1ff;--blue:#6e7cff;--pink:#b35cff;--good:#69f0bb;--shadow:0 28px 80px rgba(0,0,0,.32)}
html,body,[class*="css"]{font-family:'Cairo',Arial,sans-serif;direction:rtl}
.stApp{background:radial-gradient(circle at 80% -10%,rgba(94,94,255,.23),transparent 32%),radial-gradient(circle at 10% 18%,rgba(0,209,255,.13),transparent 27%),linear-gradient(180deg,#08111f 0%,#0a1322 100%);color:var(--txt)}
#MainMenu,footer,header{visibility:hidden}.block-container{max-width:1220px;padding:1.1rem 1.4rem 4rem}
.nav{position:sticky;top:0;z-index:50;backdrop-filter:blur(18px);background:rgba(7,17,31,.62);border:1px solid var(--line);border-radius:22px;padding:12px 16px;margin:0 0 28px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 10px 40px rgba(0,0,0,.16)}
.brand{font-size:22px;font-weight:900;display:flex;align-items:center;gap:10px}.logo{width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,var(--cyan),var(--blue) 58%,var(--pink));display:grid;place-items:center;color:#06111f}.navsub{font-size:14px;color:var(--muted)}
.hero{min-height:560px;display:grid;grid-template-columns:1.06fr .94fr;gap:54px;align-items:center;padding:45px 5px}.eyebrow{display:inline-flex;padding:7px 13px;border-radius:999px;border:1px solid rgba(89,225,255,.23);background:rgba(89,225,255,.08);color:#96edff;font-size:12px;font-weight:800}.hero h1{font-size:70px;line-height:1.07;margin:18px 0;letter-spacing:-2.6px}.grad{background:linear-gradient(90deg,#fff 0%,#7aeaff 38%,#8e8fff 72%,#d28cff 100%);-webkit-background-clip:text;background-clip:text;color:transparent}.hero p{font-size:20px;line-height:2;color:var(--muted);max-width:660px}.visual{height:430px;border-radius:34px;border:1px solid var(--line);background:linear-gradient(145deg,rgba(255,255,255,.08),rgba(255,255,255,.02));position:relative;overflow:hidden;box-shadow:var(--shadow)}
.visual:before{content:'';position:absolute;width:330px;height:330px;border-radius:50%;background:radial-gradient(circle,rgba(89,225,255,.33),rgba(110,124,255,.06) 55%,transparent 70%);top:42px;right:70px}.visual:after{content:'FOCUS';position:absolute;inset:0;display:grid;place-items:center;font-size:76px;font-weight:900;letter-spacing:9px;color:rgba(255,255,255,.92);text-shadow:0 0 50px rgba(89,225,255,.4)}
.orbit{position:absolute;border:1px solid rgba(255,255,255,.16);border-radius:50%}.orbit.a{width:280px;height:280px;top:70px;right:90px}.orbit.b{width:190px;height:190px;top:115px;right:135px;border-style:dashed}.dot{position:absolute;width:12px;height:12px;border-radius:50%;background:var(--cyan);box-shadow:0 0 24px var(--cyan)}.dot.d1{top:70px;right:225px}.dot.d2{bottom:78px;left:130px;background:var(--pink);box-shadow:0 0 24px var(--pink)}
.card{height:100%;background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.035));border:1px solid var(--line);border-radius:24px;padding:24px;box-shadow:0 15px 45px rgba(0,0,0,.16)}.card h3{margin:7px 0 8px;font-size:22px}.card p{font-size:16px;line-height:1.9}.muted{color:var(--muted);font-size:16px;line-height:1.9}.section-title{font-size:34px;font-weight:900;margin:48px 0 18px}.mini{font-size:12px;color:#9befff}.feature{width:48px;height:48px;border-radius:15px;background:linear-gradient(135deg,rgba(89,225,255,.18),rgba(179,92,255,.16));border:1px solid rgba(255,255,255,.08);display:grid;place-items:center;font-size:22px;margin-bottom:14px}
.stButton>button{border-radius:14px!important;border:1px solid rgba(255,255,255,.11)!important;background:rgba(255,255,255,.055)!important;color:#fff!important;font-weight:800!important;min-height:48px!important;transition:.18s ease!important}.stButton>button:hover{transform:translateY(-1px);border-color:rgba(89,225,255,.45)!important;box-shadow:0 12px 30px rgba(20,160,210,.12)!important}.primary .stButton>button{background:linear-gradient(135deg,#58defc,#7582ff 60%,#a566ff)!important;border:0!important;color:#07111f!important;box-shadow:0 16px 34px rgba(92,123,255,.28)!important}
.test-shell{max-width:930px;margin:15px auto}.test-top{display:flex;justify-content:space-between;gap:12px;align-items:center}.pill{font-size:14px;display:inline-flex;padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--line);color:#c9d7e7;font-size:12px;font-weight:800}.progress{height:8px;background:rgba(255,255,255,.07);border-radius:999px;margin:13px 0 26px;overflow:hidden}.progress>div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--cyan),var(--blue),var(--pink));box-shadow:0 0 18px rgba(89,225,255,.25)}
.stimulus{background:rgba(255,255,255,.035);border:1px solid var(--line);border-radius:28px;padding:18px;box-shadow:var(--shadow);position:relative;overflow:hidden}.question{font-size:32px;font-weight:900;text-align:center;margin:28px 0 18px}.timer{text-align:center;color:#a9bdd1;font-weight:700;margin-top:12px}.metric{font-size:38px;font-weight:900;background:linear-gradient(90deg,#8af0ff,#8e98ff,#d39aff);-webkit-background-clip:text;background-clip:text;color:transparent}.metric-label{font-size:15px;color:var(--muted)}
.score-wrap{max-width:820px;margin:30px auto;text-align:center}.score-ring{width:210px;height:210px;margin:25px auto;border-radius:50%;background:conic-gradient(#69e8ff var(--score),rgba(255,255,255,.08) 0);display:grid;place-items:center;position:relative;box-shadow:0 0 60px rgba(89,225,255,.15)}.score-ring:after{content:'';position:absolute;inset:14px;border-radius:50%;background:#0b1524}.score-num{position:relative;z-index:2;font-size:50px;font-weight:900}.ok{color:var(--good)}
hr{border-color:rgba(255,255,255,.08)}
[data-testid="stImage"] img{border-radius:18px;max-height:590px;object-fit:contain}
div[data-testid="stWidgetLabel"] p{font-size:16px!important;font-weight:700!important;color:#eef7ff!important}
.stButton>button{font-size:16px!important}
[data-testid="stDataFrame"]{font-size:15px}
@media(max-width:850px){.hero{grid-template-columns:1fr;min-height:auto;padding-top:20px}.hero h1{font-size:50px}.visual{height:320px}.navsub{display:none}}
</style>
''', unsafe_allow_html=True)

# ---------- nav ----------
def nav():
    st.markdown('<div class="nav"><div class="brand"><div class="logo">🧠</div><div>FocusAI<div class="navsub">بصمة الانتباه • AI Research Prototype</div></div></div><div class="navsub">نظام بحثي غير تشخيصي</div></div>', unsafe_allow_html=True)
    cols=st.columns(6)
    for c,(label,page) in zip(cols,[('الرئيسية','home'),('الاختبار','test'),('بياناتي','daily'),('التحليل','dashboard'),('الذكاء الاصطناعي','ai'),('عن المشروع','about')]):
        with c:
            if st.button(label,key='nav_'+page,use_container_width=True):
                st.session_state.page=page
                st.rerun()


def reset_test():
    st.session_state.test={'index':0,'phase':'intro','started':None,'question_started':None,'answers':[],'results':{},'questions':[]}
    st.session_state.last_score=None
    st.session_state.last_saved=False


def build_random_test():
    groups = [
        [q for q in QUESTIONS if q['section']=='أوجد الاختلاف'],
        [q for q in QUESTIONS if q['section']=='الذاكرة البصرية'],
        [q for q in QUESTIONS if q['section'] in ['العدّ والتركيز','التعرّف البصري']]
    ]
    selected=[random.choice(g) for g in groups if g]
    random.shuffle(selected)
    st.session_state.test['questions']=selected


def finish_test():
    answers=st.session_state.test['answers']
    groups={'أوجد الاختلاف':'r1','الذاكرة البصرية':'r2','العدّ والتركيز':'r3','التعرّف البصري':'r3'}
    results={}
    for key in ['r1','r2','r3']:
        vals=[a['correct'] for a in answers if groups.get(a['section'])==key]
        results[key]=round(sum(vals)/len(vals)*100) if vals else 0
    results['r4']=0
    results['overall']=round(sum(a['correct'] for a in answers)/len(answers)*100) if answers else 0
    results['accuracy']=results['overall']/100
    results['errors']=sum(1-a['correct'] for a in answers)
    results['avg_reaction_ms']=round(np.mean([a['reaction_ms'] for a in answers]),1) if answers else 0
    results['total_questions']=len(answers)
    results['test_time']=datetime.now().strftime('%H:%M')
    st.session_state.test['results']=results
    st.session_state.last_score=results['overall']
    st.session_state.last_saved=False
    st.session_state.page='result'

nav()
page=st.session_state.page

# ---------- home ----------
if page=='home':
    st.markdown('''<div class="hero"><div><span class="eyebrow">AI • Machine Learning • Attention Research</span><h1>اختبر تركيزك.<br><span class="grad">واكتشف بصمة انتباهك.</span></h1><p>FocusAI يجمع عوامل يومية منظمة مع اختبار انتباه عملي، ثم يستخدم التعلم الآلي لاكتشاف الأنماط والتنبؤ بمستوى أداء الانتباه عندما تتوفر بيانات كافية.</p></div><div class="visual"><div class="orbit a"></div><div class="orbit b"></div><div class="dot d1"></div><div class="dot d2"></div></div></div>''', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    cards=[('01','اختبار فعلي','يقيس الدقة والأخطاء وزمن الاستجابة بدل الاعتماد على التقييم الشخصي فقط.'),('02','بيانات يومية','النوم، وقت الشاشة، الدراسة، الاستراحات ووقت الاختبار.'),('03','تعلم آلي','مقارنة Logistic Regression وDecision Tree وRandom Forest عند كفاية البيانات.')]
    for c,(n,t,x) in zip([c1,c2,c3],cards):
        with c: st.markdown(f'<div class="card"><div class="feature">{n}</div><h3>{t}</h3><p class="muted">{x}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">جاهز؟</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><span class="mini">3 مهام • اختيار عشوائي • Reaction Time</span><h3>ابدأ من غير اسم</h3><p class="muted">يستخدم النظام معرّفًا عشوائيًا فقط لحفظ النتائج.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="primary">',unsafe_allow_html=True)
    if st.button('ابدأ اختبار التركيز  ←'):
        reset_test(); st.session_state.page='test'; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# ---------- test ----------
elif page=='test':
    t=st.session_state.test
    st.markdown('<div class="test-shell">',unsafe_allow_html=True)
    if t['phase']=='intro':
        st.markdown('<span class="eyebrow">Focus Challenge</span><h1 style="font-size:42px;margin:14px 0 8px">3 مهام قصيرة لقياس أداء الانتباه.</h1><p class="muted" style="font-size:17px;line-height:2">ستظهر صورة لفترة محددة، ثم يظهر السؤال. بعد ظهور السؤال يبدأ قياس زمن الاستجابة حتى اختيار الإجابة.</p>',unsafe_allow_html=True)
        st.markdown('<div class="card" style="margin:22px 0"><h3>مؤشرات القياس</h3><p class="muted">الدقة، عدد الأخطاء، ومتوسط زمن الاستجابة. هذه المؤشرات بحثية ولا تشخّص أي حالة طبية.</p></div>',unsafe_allow_html=True)
        st.checkbox('أوافق على حفظ نتيجة الاختبار داخل المشروع دون اسم أو معلومات تعريفية.',key='consent')
        if st.session_state.consent:
            st.markdown('<div class="primary">',unsafe_allow_html=True)
            if st.button('ابدأ الآن'):
                t['index']=0; t['phase']='show'; t['started']=None; t['question_started']=None; t['answers']=[]; t['results']={}; build_random_test(); st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)
    elif t['phase']=='show':
        i=t['index']; questions=t.get('questions') or []; q=questions[i]; pct=int(i/len(questions)*100)
        st.markdown(f'<div class="test-top"><span class="pill">{q["badge"]}</span><span class="pill">{i+1} / {len(questions)}</span></div><div class="progress"><div style="width:{max(4,pct)}%"></div></div>',unsafe_allow_html=True)
        if not os.path.exists(q['image']): st.error('تعذر العثور على صورة الاختبار.'); st.stop()
        if t['started'] is None: t['started']=time.time()
        st.markdown('<div class="stimulus">',unsafe_allow_html=True); st.image(q['image']); st.markdown('</div>',unsafe_allow_html=True)
        elapsed=time.time()-t['started']; remaining=max(0,q['duration']-elapsed)
        if remaining>0:
            st.markdown(f'<div class="timer">ركز في التفاصيل… {remaining:.1f} ثانية</div>',unsafe_allow_html=True)
            time.sleep(min(.25,remaining)); st.rerun()
        t['phase']='question'; t['started']=None; t['question_started']=time.time(); st.rerun()
    elif t['phase']=='question':
        i=t['index']; questions=t.get('questions') or []; q=questions[i]; pct=int((i+1)/len(questions)*100)
        st.markdown(f'<div class="test-top"><span class="pill">{q["section"]}</span><span class="pill">{i+1} / {len(questions)}</span></div><div class="progress"><div style="width:{pct}%"></div></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="question">{q["question"]}</div>',unsafe_allow_html=True)
        cols=st.columns(len(q['options']) if len(q['options'])<=4 else 2)
        for j,opt in enumerate(q['options']):
            with cols[j%len(cols)]:
                if st.button(opt,key=f'a_{i}_{j}',use_container_width=True):
                    reaction_ms=(time.time()-t['question_started'])*1000 if t['question_started'] else 0
                    t['answers'].append({'index':i,'section':q['section'],'correct':1 if opt==q['correct'] else 0,'answer':opt,'reaction_ms':round(reaction_ms,1)})
                    if i<len(questions)-1:
                        t['index']+=1; t['phase']='show'; t['started']=None; t['question_started']=None
                    else: finish_test()
                    st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# ---------- result ----------
elif page=='result':
    r=st.session_state.test.get('results',{}); score=r.get('overall',0)
    level='ممتاز جدًا' if score>=90 else 'ممتاز' if score>=80 else 'جيد جدًا' if score>=70 else 'جيد' if score>=55 else 'يحتاج تدريب أكثر'
    st.markdown('<div class="score-wrap"><span class="eyebrow">اكتملت بصمة الانتباه</span><h1 style="font-size:42px;margin:14px 0 2px">نتيجتك النهائية</h1><p class="muted">النتيجة تصف أداء هذه المحاولة فقط.</p>',unsafe_allow_html=True)
    st.markdown(f'<div class="score-ring" style="--score:{score}%"><div class="score-num">{score}%</div></div><h2>{level}</h2>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    for c,label,val in zip([c1,c2,c3],['الدقة','الأخطاء','متوسط زمن الاستجابة'],[f'{r.get("accuracy",0)*100:.0f}%',str(r.get('errors',0)),f'{r.get("avg_reaction_ms",0):.0f} ms']):
        with c: st.markdown(f'<div class="card"><div class="metric">{val}</div><div class="metric-label">{label}</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:18px"></div><div class="primary">',unsafe_allow_html=True)
    if st.button('حفظ النتيجة وإضافة بيانات اليوم  ←',use_container_width=True): st.session_state.page='daily'; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)
    if st.button('إعادة الاختبار',use_container_width=True): reset_test(); st.session_state.page='test'; st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# ---------- daily ----------
elif page=='daily':
    st.markdown('<span class="eyebrow">بيانات اليوم</span><h1 style="font-size:40px;margin:14px 0 7px">اربط نتيجتك بعادات يومك</h1><p class="muted">هذه البيانات هي مدخلات البحث التي سيستخدمها النموذج لاكتشاف الأنماط.</p>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sleep=st.number_input('ساعات النوم',3.0,12.0,7.0,.5)
        bedtime=st.time_input('وقت النوم')
        screen=st.number_input('استخدام الأجهزة — ساعة',0.0,14.0,4.0,.5)
    with c2:
        wake_time=st.time_input('وقت الاستيقاظ')
        study=st.number_input('الدراسة — ساعة',0.0,12.0,2.0,.5)
        breaks=st.number_input('عدد الاستراحات',0,12,3,1)
    st.markdown(f'<div class="card" style="margin-top:20px"><span class="pill">معرّف مجهول</span><h3>{st.session_state.participant}</h3><p class="muted">لا نطلب اسمك أو رقم هويتك.</p></div>',unsafe_allow_html=True)
    if st.session_state.last_score is None: st.info('أجرِ الاختبار أولًا حتى تُحفظ النتيجة.')
    elif st.button('حفظ بيانات اليوم'):
        rr=st.session_state.test['results']
        save_row({
            'participant':st.session_state.participant,
            'date':datetime.now().strftime('%Y-%m-%d %H:%M'),
            'sleep':sleep,
            'bedtime':bedtime.strftime('%H:%M'),
            'wake_time':wake_time.strftime('%H:%M'),
            'screen':screen,
            'study':study,
            'breaks':breaks,
            'test_time':rr.get('test_time',''),
            'score':st.session_state.last_score,
            'accuracy':rr.get('accuracy',np.nan),
            'errors':rr.get('errors',np.nan),
            'avg_reaction_ms':rr.get('avg_reaction_ms',np.nan),
            'round1':rr.get('r1',0),'round2':rr.get('r2',0),'round3':rr.get('r3',0),'round4':0
        })
        st.session_state.last_saved=True
        st.success('تم حفظ البيانات بنجاح.'); st.session_state.page='dashboard'; st.rerun()

# ---------- dashboard ----------
elif page=='dashboard':
    df=load_data()
    st.markdown('<span class="eyebrow">Dashboard</span><h1 style="font-size:40px;margin:14px 0 7px">لوحة التحليل</h1><p class="muted">تحليل وصفي للبيانات المسجلة قبل الانتقال إلى نماذج التعلم الآلي.</p>',unsafe_allow_html=True)
    if df.empty:
        st.markdown('<div class="card"><h3>لا توجد بيانات بعد</h3><p class="muted">أكمل اختبارًا واحفظ بيانات اليوم حتى يظهر التحليل هنا.</p></div>',unsafe_allow_html=True)
    else:
        latest=df.iloc[-1]; c1,c2,c3,c4=st.columns(4)
        vals=[f'{pd.to_numeric(df.score,errors="coerce").mean():.0f}%',f'{latest.sleep:.1f}',f'{latest.screen:.1f}',f'{latest.avg_reaction_ms:.0f} ms']
        labs=['متوسط الأداء','ساعات النوم','وقت الشاشة','متوسط الاستجابة']
        for c,v,l in zip([c1,c2,c3,c4],vals,labs):
            with c: st.markdown(f'<div class="card"><div class="metric">{v}</div><div class="metric-label">{l}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">تطور النتيجة</div>',unsafe_allow_html=True)
        chart=df[['date','score']].copy().set_index('date'); st.line_chart(chart,use_container_width=True)
        st.markdown('<div class="section-title">السجل البحثي</div>',unsafe_allow_html=True)
        show=df[['date','sleep','bedtime','wake_time','screen','study','breaks','accuracy','errors','avg_reaction_ms','score']].rename(columns={'date':'التاريخ','sleep':'النوم','bedtime':'وقت النوم','wake_time':'الاستيقاظ','screen':'الشاشة','study':'الدراسة','breaks':'الاستراحات','accuracy':'الدقة','errors':'الأخطاء','avg_reaction_ms':'الاستجابة ms','score':'النتيجة'})
        st.dataframe(show,use_container_width=True,hide_index=True)

# ---------- AI ----------
elif page=='ai':
    st.markdown('<span class="eyebrow">Machine Learning Lab</span><h1 style="font-size:40px;margin:14px 0 7px">مختبر الذكاء الاصطناعي</h1><p class="muted">يقارن النظام ثلاثة نماذج فعلية. لا يعرض أرقامًا وهمية: إذا كانت البيانات غير كافية سيخبرك بذلك.</p>',unsafe_allow_html=True)
    df=load_data()
    result=evaluate_models(df)
    c1,c2,c3=st.columns(3)
    for c,title,desc in zip([c1,c2,c3],['Logistic Regression','Decision Tree','Random Forest'],['نموذج خطي للتصنيف.','نموذج شجري قابل للتفسير.','نموذج تجميعي لاكتشاف أنماط غير خطية.']):
        with c: st.markdown(f'<div class="card"><h3>{title}</h3><p class="muted">{desc}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">جاهزية التجربة</div>',unsafe_allow_html=True)
    if not result.get('ready'):
        st.warning(result['reason'])
        st.info(f'عدد السجلات الحالية: {len(df)}. يُفضّل جمع بيانات يومية من عدد أكبر من المشاركين ولفترة أطول قبل إعلان دقة النموذج.')
    else:
        st.success(f'تم تدريب النماذج فعليًا. النموذج الأفضل في بيانات الاختبار الحالية: {result["best"]}.')
        metrics=result['metrics'].copy(); metrics['Accuracy']=(metrics['Accuracy']*100).round(1).astype(str)+'%'; metrics['F1']=(metrics['F1']*100).round(1).astype(str)+'%'
        st.dataframe(metrics,use_container_width=True,hide_index=True)
        st.caption('التقييم يتم على جزء اختبار لم يُستخدم في تدريب النموذج. هذه الأرقام تتغير مع زيادة البيانات. ومع العينة الحالية تُعرض بوصفها نتيجة استكشافية وليست نتيجة نهائية.')
        best_model=result['models'][result['best']]
        inner=getattr(best_model,'named_steps',{})
        if 'model' in inner and hasattr(inner['model'],'feature_importances_'):
            imp=pd.DataFrame({'العامل':['النوم','وقت الشاشة','مدة الدراسة','الاستراحات'],'الأهمية':inner['model'].feature_importances_}).sort_values('الأهمية',ascending=False)
            st.markdown('<div class="section-title">العوامل الأكثر استخدامًا في قرار النموذج</div>',unsafe_allow_html=True)
            st.dataframe(imp.assign(الأهمية=(imp['الأهمية']*100).round(1).astype(str)+'%'),use_container_width=True,hide_index=True)

        st.markdown('<div class="section-title">التنبؤ لمستخدم جديد</div>',unsafe_allow_html=True)
        p1,p2=st.columns(2)
        with p1:
            psleep=st.number_input('ساعات النوم',3.0,12.0,7.0,.5,key='pred_sleep')
            pscreen=st.number_input('استخدام الأجهزة — ساعة',0.0,14.0,4.0,.5,key='pred_screen')
        with p2:
            pstudy=st.number_input('مدة الدراسة — ساعة',0.0,12.0,2.0,.5,key='pred_study')
            pbreaks=st.number_input('عدد فترات الراحة',0,12,3,1,key='pred_breaks')
        st.caption('هذا التنبؤ يعتمد على العوامل اليومية فقط، ثم يقارن لاحقًا بنتيجة اختبار الانتباه الفعلية. لا يستخدم الدقة أو الأخطاء أو زمن الاستجابة كمدخلات للتنبؤ.')
        if st.button('تشغيل التنبؤ الآن'):
            pred=predict_current(df,[psleep,pscreen,pstudy,pbreaks])
            if pred.get('ready'):
                label='أداء متوقع جيد' if pred['prediction']==1 else 'أداء متوقع يحتاج تحسين'
                conf=pred['probability']*100 if pred.get('probability') is not None else None
                st.success(f'النموذج المختار: {pred["model_name"]} | {label} | احتمال الفئة الجيدة: {conf:.1f}%')
                st.info(f'التنبؤ مبني على {pred["sample_size"]} سجلًا حقيقيًا محفوظًا في المشروع. كلما زاد عدد المشاركين والأيام أصبحت النتيجة أكثر موثوقية.')
            else: st.warning(pred['reason'])

# ---------- about ----------
elif page=='about':
    st.markdown('<span class="eyebrow">عن المشروع</span><h1 style="font-size:40px;margin:14px 0 7px">FocusAI</h1><p class="muted" style="font-size:17px;line-height:2;max-width:900px">نظام بحثي برمجي يجمع عوامل يومية مع قياس عملي لأداء الانتباه، ثم يستخدم التعلم الآلي لاكتشاف الأنماط والتنبؤ بمستوى الأداء. لا يهدف المشروع إلى تشخيص اضطرابات الانتباه أو أي حالة طبية.</p>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    for c,t,x in zip([c1,c2,c3],['اختبار انتباه','بيانات منظمة','3 نماذج ML'],['دقة وأخطاء وزمن استجابة.','نوم، شاشة، دراسة، استراحات وأوقات.','Logistic Regression وDecision Tree وRandom Forest.']):
        with c: st.markdown(f'<div class="card"><h3>{t}</h3><p class="muted">{x}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">الحدود العلمية</div><div class="card"><p class="muted">لا يتم اعتبار أي نتيجة رقمية دليلًا طبيًا. جودة التنبؤ تعتمد على حجم البيانات وتنوع المشاركين ومدة جمع البيانات، لذلك يعرض النظام تحذيرًا بدل اختلاق دقة عندما تكون العينة غير كافية.</p></div>',unsafe_allow_html=True)
