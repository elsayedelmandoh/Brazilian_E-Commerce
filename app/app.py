import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('../Models/KNN_2_june_3.pkl')
sc = StandardScaler()

def predict_sentiment(data):
    prediction = model.predict(data)
    probabilities = model.predict_proba(data)
    print(prediction)
    print(probabilities)
    return prediction, probabilities

# Define the main function for the Streamlit app
def main():
    # set Streamlit app title and layout
    st.set_page_config(page_title='Brazilian E-Commerce', layout='centered')

    # set title and description
    st.title('Brazilian E-Commerce')
    st.write('Enter your data to predict whether it is Satisfied or Not Satisfied:')

    # Data input
    pri_input = st.text_input("Enter a price:")
    frigh_input = st.text_input("Enter a Frigh:")
    payment_input = st.text_input("Enter a payment:")

    data = []

    if pri_input != "" and frigh_input != "" and payment_input != "":
        try:
            data = [[float(pri_input), float(frigh_input), float(payment_input)]]
        except ValueError:
            st.error("Invalid input. Please enter numeric values.")

    prediction_button = st.button("Predict")

    if prediction_button:
        if data:
            prediction, probabilities = predict_sentiment(data)
            st.subheader("Prediction:")
            
            if prediction == 1:
                st.write("Satisfied: 😂")
            else:
                st.write("Not Satisfied: 😐")
                
            # Display probabilities using sns.barplot and annotate
            class_labels = ['Not Satisfied', 'Satisfied']
            probabilities = probabilities[0]  # Extract probabilities for the single data point
            fig, ax = plt.subplots()
            ax = sns.barplot(x=class_labels, y=probabilities)
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])  # Set y-axis limits to range from 0 to 1
            ax.set_title('Class Probabilities')
            for i, prob in enumerate(probabilities):
                ax.annotate(f'{prob:.2f}', xy=(i, prob), ha='center', va='bottom')
            st.pyplot(fig)
        else:
            st.error("Please enter values for all fields.")

if __name__ == '__main__':
    main()
