import streamlit as st

st.title("Assignment 1 / 과제 1")

# --- Input numbers (string) / 문자열로 입력 ---
st.header("Enter two numbers / 두 숫자를 입력하세요")

A_str = st.text_input("Enter number A / 숫자 A 입력:", "0")
B_str = st.text_input("Enter number B / 숫자 B 입력:", "0")

# --- Convert to integers safely / 숫자로 변환 (에러 처리 포함) ---
try:
    A = int(A_str)
except ValueError:
    A = 0
    st.warning("A must be a number / A는 숫자여야 합니다.")

try:
    B = int(B_str)
except ValueError:
    B = 0
    st.warning("B must be a number / B는 숫자여야 합니다.")

# --- Sum of A + B / A+B 합계 ---
sum_ab = A + B
st.write(f"Sum of A and B is / A + B 의 합: **{sum_ab}**")

# --- Select which number / A 또는 B 중 선택 ---
st.header("Choose a number to sum up to / 합산할 숫자 선택")
choice = st.selectbox("Select A or B / A 또는 B를 선택:", ("A", "B"))

# --- Sum from 1 to chosen number / 1부터 선택된 숫자까지의 합 ---
if choice == "A":
    total_sum = sum(range(1, A + 1))
else:
    total_sum = sum(range(1, B + 1))

st.write(f"Sum from 1 to {choice} is / 1부터 {choice}까지의 합: **{total_sum}**")
