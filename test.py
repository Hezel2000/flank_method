import streamlit as st
import pandas as pd

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=[r'$\alpha$', 'B', 'C'])

st.write(df)

st.dataframe(df)


# Create a dataframe with three columns
df = pd.DataFrame({'$\alpha$': [1, 2, 3], '$\beta$': [
                  4, 5, 6], '$\gamma$': [7, 8, 9]})

# Display the dataframe
print(df)
