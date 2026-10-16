"""NAFLD Insight — Research Streamlit UI. No TEST, retraining, or database."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.inference import ROOT,load_runtime,predict_case

st.set_page_config(page_title='NAFLD Insight | HUIT Research',page_icon='🧬',layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Be Vietnam Pro',sans-serif;}
.main .block-container{max-width:1230px;padding-top:1.9rem;}
[data-testid="stSidebar"]{background:#0d2546;}[data-testid="stSidebar"] *{color:#ecf3ff!important;}
.hero{border-radius:24px;background:linear-gradient(115deg,#103564 0%,#1b58a5 60%,#228cbd 100%);color:#fff;padding:30px 35px;margin-bottom:23px;box-shadow:0 14px 40px #143f7029;}
.hero h1{font-size:38px;color:#fff!important;margin:8px 0 12px;font-weight:800;letter-spacing:-.03em;}
.hero p{max-width:720px;line-height:1.8;font-size:14px;color:#e6f3ff;}
.kicker{font-size:11px;font-weight:800;letter-spacing:.13em;color:#bee9ff;}
.kpi{padding:20px;border-radius:17px;border:1px solid #e4edf7;background:#fff;margin-bottom:8px;}
.kpi strong{display:block;font-size:28px;color:#134779;}.kpi span{font-size:12px;font-weight:600;color:#64748b;}
.note{padding:16px 18px;border-left:4px solid #3b82f6;border-radius:12px;background:#eaf4ff;color:#1d446f;font-size:13px;line-height:1.8;}
.warn{padding:16px 18px;border:1px solid #f0d2a0;border-radius:12px;background:#fff7e9;color:#775414;font-size:13px;line-height:1.8;}
h1,h2,h3{letter-spacing:-.02em!important;}
[data-testid="stMetric"]{border-radius:16px;border:1px solid #e2eaf5;background:#fff;padding:14px 18px;}
</style>''',unsafe_allow_html=True)
@st.cache_resource(show_spinner='Đang xác minh model...')
def load():return load_runtime()
try:b,groups,weights,thresholds,models=load()
except Exception as e:
    st.error('Không thể nạp/xác minh artifact. Chế độ suy luận bị vô hiệu hóa.');st.code(str(e));st.stop()

with st.sidebar:
    st.markdown('## 🧬 NAFLD Insight')
    st.caption('HUIT · RESEARCH DEMO · 2026')
    st.divider()
    nav=st.radio('Điều hướng',['Tổng quan','Dự đoán nguy cơ','Giải thích SHAP','Về mô hình'],label_visibility='collapsed')
    st.divider()
    st.markdown('**Model đã khóa**')
    st.caption('Fusion (b) · seed 42 · 43 đặc trưng')
    st.caption('Không lưu hồ sơ hoặc đọc TEST.')

st.markdown('''<section class="hero"><div class="kicker">HUIT · KHÓA LUẬN NGHIÊN CỨU</div><h1>NAFLD Insight</h1>
<p>Minh họa dự đoán nguy cơ gan nhiễm mỡ không do rượu từ dữ liệu lâm sàng đa nhóm bằng mô hình Fusion (b), kết hợp giải thích theo nghiên cứu Tuần 9.</p></section>''',unsafe_allow_html=True)

if nav=='Tổng quan':
    st.markdown('### Một quy trình dự đoán có thể kiểm chứng')
    st.write('Ứng dụng sử dụng bộ tiền xử lý đã fit TRAIN, bốn mô hình nhóm và trọng số Fusion (b) seed42. Xác suất được phân loại theo ngưỡng đã khóa.')
    cols=st.columns(4)
    for col,n,text in zip(cols,['26','43','04','05'],['Biến đầu vào','Đặc trưng xử lý','Nhóm đặc trưng','Seed nghiên cứu']):
        col.markdown(f'<div class="kpi"><strong>{n}</strong><span>{text}</span></div>',unsafe_allow_html=True)
    st.image(str(ROOT/'resource/figures/Hinh_KienTruc_Fusion.png'),use_container_width=True)
    st.markdown('<div class="note">Bộ demo chỉ hỗ trợ minh họa phương pháp nghiên cứu, không phải công cụ chẩn đoán hoặc điều trị.</div>',unsafe_allow_html=True)

elif nav=='Dự đoán nguy cơ':
    st.markdown('### Nhập dữ liệu lâm sàng')
    st.caption('Điền các thông tin theo bốn nhóm. Các trường không biết có thể đánh dấu thiếu; phần mềm dùng giá trị điền thiếu đã học từ TRAIN.')
    meta=pd.read_csv(ROOT/'resource/ten_bien_don_vi.csv',encoding='utf-8-sig')
    names={r.bien_goc:(r.ten_bien,str(r.don_vi) if pd.notna(r.don_vi) else '') for r in meta.itertuples()}
    enums={'RIAGENDR':{1:'Nam',2:'Nữ'},'BPQ020':{1:'Có',2:'Không'},'BPQ080':{1:'Có',2:'Không'},'PAQ650':{1:'Có',2:'Không'},
           'RIDRETH3':{1:'Nhóm NHANES 1',2:'Nhóm NHANES 2',3:'Nhóm NHANES 3',4:'Nhóm NHANES 4',6:'Nhóm NHANES 6',7:'Nhóm NHANES 7'},
           'SMOKE_STATUS':{0:'Nhóm 0',1:'Nhóm 1',2:'Nhóm 2',3:'Nhóm 3'},'GLYCEMIC_HX':{0:'Nhóm 0',1:'Nhóm 1',2:'Nhóm 2'}}
    gnames={'G1':'01 · Nhân khẩu & nhân trắc','G2':'02 · Xét nghiệm sinh hóa','G3':'03 · Bệnh nền','G4':'04 · Lối sống & khẩu phần'}
    vargroups={}
    for g,columns in groups.items():
        for c in columns:
            key=c.split('_')[0] if c.startswith(('RIDRETH3_','SMOKE_STATUS_','GLYCEMIC_HX_')) else c
            vargroups[key]=g
    allnums=list(b['numeric_vars']); allcats=list(b['binary_mappings'])+list(b['multi_categorical_vars'])
    with st.form('clinical_form'):
        case={}
        tabs=st.tabs(list(gnames.values()))
        for group,tab in zip(gnames,tabs):
            with tab:
                fields=[c for c in allnums+allcats if vargroups.get(c)==group]
                a,z=st.columns(2)
                for idx,c in enumerate(fields):
                    with (a if idx%2==0 else z):
                        label,unit=names.get(c,(c,'')); label=str(label)
                        if c in allnums:
                            missing=st.checkbox(f'Không có dữ liệu: {label}',key='missing_'+c)
                            default=float(b['num_imputer'].statistics_[allnums.index(c)])
                            minv=18.0 if c=='RIDAGEYR' else 0.0
                            maxv=80.0 if c=='RIDAGEYR' else 100000.0
                            value=st.number_input(label,min_value=minv,max_value=maxv,value=float(max(minv,min(maxv,default))),step=1.0 if c in ('RIDAGEYR','PAD680') else .1,disabled=missing,help=f'{c} · {unit}')
                            case[c]=None if missing else float(value)
                        else:
                            choices=enums[c];case[c]=st.selectbox(label,options=list(choices),format_func=lambda x:choices[x],key='enum_'+c,help=c)
        st.divider()
        mode=st.selectbox('Chế độ ngưỡng',['F1 OOF','0,5 cố định','Recall ≥ 0,80 OOF'])
        submit=st.form_submit_button('Phân tích nguy cơ →',type='primary',use_container_width=True)
    if submit:
        try:st.session_state['prediction']=predict_case(case,mode)
        except Exception as e:st.error(f'Không thể dự đoán: {e}');st.session_state.pop('prediction',None)
    if 'prediction' in st.session_state:
        r=st.session_state['prediction'];st.divider()
        st.markdown('### Kết quả suy luận')
        lc,rc=st.columns([1,1])
        with lc:
            fig=go.Figure(go.Indicator(mode='gauge+number',value=r['probability']*100,number={'suffix':'%','font':{'size':42,'color':'#133965'}},gauge={'axis':{'range':[0,100]},'bar':{'color':'#2874da'},'bgcolor':'#eaf2fc','threshold':{'line':{'color':'#f59e0b','width':5},'value':r['threshold']*100}}))
            fig.update_layout(height=270,margin=dict(l=15,r=15,t=20,b=5),paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
        with rc:
            st.metric('Xác suất dự đoán',f'{r["probability"]*100:.2f}%')
            st.metric('Ngưỡng đã khóa',f'{r["threshold"]*100:.2f}%')
            if r['positive']:st.warning('Dự đoán **đạt/vượt ngưỡng phân loại** của mô hình.')
            else:st.success('Dự đoán **dưới ngưỡng phân loại** của mô hình.')
            st.caption('Đây là ngưỡng thống kê OOF, không phải ngưỡng chẩn đoán y khoa.')
        st.markdown('#### Tỷ lệ đóng góp xác suất từ các mô hình nhóm')
        table=pd.DataFrame([{'Nhóm':gnames[g],'Xác suất nhóm (%)':round(r['groups'][g]*100,2),'Trọng số (%)':round(r['weights'][g]*100,3),'Đóng góp (%)':round(r['groups'][g]*r['weights'][g]*100,3)} for g in groups])
        st.dataframe(table,use_container_width=True,hide_index=True)
        st.markdown('<div class="warn"><b>Thông báo y khoa:</b> Đây là bản demo phục vụ nghiên cứu, không thay thế chẩn đoán hoặc tư vấn y tế. Xác suất mô hình không được xem là nguy cơ tuyệt đối đã được hiệu chuẩn lâm sàng.</div>',unsafe_allow_html=True)

elif nav=='Giải thích SHAP':
    st.markdown('### Điều gì tác động đến dự đoán trong nghiên cứu?')
    st.markdown('<div class="note"><b>Giải thích toàn cục từ validation Tuần 9.</b> Các biểu đồ này không phải SHAP riêng cho bệnh nhân vừa nhập và không tính lại từ TEST.</div>',unsafe_allow_html=True)
    st.image(str(ROOT/'resource/figures/hinh_shap_bar_fusion_b.png'),use_container_width=True)
    aa,bb=st.columns(2)
    with aa:st.image(str(ROOT/'resource/figures/hinh_shap_nhom.png'),caption='Tầm quan trọng theo nhóm',use_container_width=True)
    with bb:st.image(str(ROOT/'resource/figures/hinh_shap_beeswarm_fusion_b.png'),caption='Phân phối SHAP trên validation',use_container_width=True)
    t=pd.read_csv(ROOT/'resource/final_shap_tuan9.csv',encoding='utf-8-sig')
    t=t[t.phan=='top15_fusion_b']
    st.markdown('#### 15 yếu tố nổi bật')
    st.dataframe(t[['muc','nhom','gia_tri','ty_trong_pct']].rename(columns={'muc':'Yếu tố','nhom':'Nhóm','gia_tri':'Mean |SHAP| (điểm %)','ty_trong_pct':'Tỷ trọng (%)'}),hide_index=True,use_container_width=True)

elif nav=='Về mô hình':
    st.markdown('### Thông tin nghiên cứu')
    st.write('**Đề tài:** Dự đoán nguy cơ gan nhiễm mỡ không do rượu từ dữ liệu lâm sàng đa nhóm bằng học máy.')
    st.write('**Dữ liệu:** NHANES 2017–2020 · tiền xử lý 26 biến gốc → 43 đặc trưng.')
    st.write('**Mô hình:** Fusion (b), seed42. G1/G4 Logistic Regression trên scaled; G2/G3 LightGBM trên raw.')
    st.write('**Nguồn SHAP:** Tuần 9, tập validation. **Không đọc TEST** trong ứng dụng.')
    st.markdown('#### Trọng số theo nhóm')
    st.bar_chart(pd.DataFrame({'Nhóm':list(weights),'Trọng số':list(weights.values())}).set_index('Nhóm'),horizontal=True,color='#2563eb')
    st.markdown('#### Ngưỡng suy luận')
    st.dataframe(pd.DataFrame([{'Chế độ':'F1 OOF','Ngưỡng':thresholds['threshold_f1_oof']},{'Chế độ':'0,5','Ngưỡng':thresholds['threshold_0_5']},{'Chế độ':'Recall ≥ 0,80 OOF','Ngưỡng':thresholds['threshold_recall80_oof']}]),hide_index=True,use_container_width=True)
    st.markdown('<div class="warn">Không dùng demo này cho quyết định khám/chữa bệnh. Hiệu năng và khả năng khái quát có giới hạn trong mẫu nghiên cứu.</div>',unsafe_allow_html=True)
