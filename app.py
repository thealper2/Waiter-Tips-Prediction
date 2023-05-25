import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

sexs = ["Female", "Male"]
answers = ["No", "Yes"]
days = ["Fri", "Sat", "Sun", "Thur"]
times = ["Dinner", "Lunch"]

st.title("Waiter Tips Prediction")
total_bill = st.number_input("Total Bill")
sex = st.selectbox("Sex", sexs)
smokers = st.selectbox("Smoker", answers)
day = st.selectbox("Day", days)
time = st.selectbox("Time", times)
size = st.number_input("Size")

if st.button("Predict"):
	sex = sexs.index(sex)
	smokers = answers.index(smokers)
	day = days.index(day)
	time = times.index(time)
	test = np.array([[total_bill, sex, smokers, day, time, size]])
	res = model.predict(test)
	print(res)
	st.success("Prediction: " + str(res[0]))
