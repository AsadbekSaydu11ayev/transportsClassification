from fastai.vision.all import *
import streamlit as st
import pathlib
import platform
plt  = platform.system()
if plt == "Linux":pathlib.WindowsPath = pathlib.PosixPath
# grafiklarni chiqarish uchun
import plotly.express as px

# title
# title - bu yordamida web dastulash oynamizda modelimizga nom berishimiz mumkin
st.title("Transportni klassifikatsiya qiluvchi model")

# rasmni joylash
# uni ichiga nom beramiz va u web oynada ko'rinadi
# keyin qanaqa rasm turlarini yuklash kerakligi kiritiladi
file = st.file_uploader("rasm yuklash",type=["png",'jpeg','svg','gif'])


# fayl yo'q paytda xatolik yuz bermasligi uchun quyidagi code ni yozamiz
if file:
    # faylimizni ekranga chiqarish
    st.image(file)

    # PIL convert
    # faylimizni rasmga aylantirib olamiz
    img = PILImage.create(file)

    # model
    # endi modelimizni yuklab olamiz
    model = load_learner("transport_model.pkl")

    # prediction
    # rasmimizni bashorat qilib ko'ramiz
    pred, pred_id, probs = model.predict(img)

    # natijamizni ekranga uzatamiz
    st.success(f"Bashorat: {pred}")
    # natijani ekranga chiqarishning 2 usuli
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # maxsulot ko'rinishini yaxshilash uchun esa unga grafiklar qo'shishimiz mumkin:
        # matplotlib, seaborn,
        # interaktiv grafik uchun esa:
            # plotly
    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab )

    # grafikni ekranga chiqaramiz
    st.plotly_chart(fig)
